import os
import torch
import json
from pathlib import Path
import sacrebleu
from typing import Literal, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import (
    SFTTrainer, 
    GRPOTrainer, 
    GRPOConfig
)
from arguments import MainConfig
class TrainerSFT:
    def __init__(self, config: MainConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.callbacks = []

    def setup(self):
        self._load_tokenizer()
        self._load_model()
        self._load_datasets()
        self._setup_trainer()

    def _load_tokenizer(self):
        print("\n[1/4] Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.tokenizer_name_or_path,
            cache_dir=self.config.model.cache_dir,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=self.config.model.use_fast_tokenizer,
            padding_side=self.config.model.tokenizer_padding_side,
        )

        print(f"✓ Tokenizer loaded: {self.config.model.tokenizer_name_or_path}")
        print(f"  - Vocab size: {len(self.tokenizer)}")
        print(f"  - Pad token: {self.tokenizer.pad_token}")
        print(f"  - EOS token: {self.tokenizer.eos_token}")

    def _load_model(self):
        print("\n[2/4] Loading model...")

        # Setup quantization config for QLoRA
        quantization_config = None
        if self.config.peft_method == "qlora":
            print("  - Setting up 4-bit quantization (QLoRA)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.lora.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.lora.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, self.config.lora.bnb_4bit_compute_dtype),
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            cache_dir=self.config.model.cache_dir,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=self.config.model.get_torch_dtype(),
            device_map=self.config.model.device_map,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2"
        )

        print(f"✓ Base model loaded: {self.config.model.model_name_or_path}")

        # Prepare model for k-bit training if using quantization
        if quantization_config is not None:
            print("  - Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.sft.gradient_checkpointing
            )

        # Enable gradient checkpointing
        if self.config.sft.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("  - Gradient checkpointing enabled")

        # Apply PEFT
        if self.config.peft_method != "none":
            print(f"  - Applying {self.config.peft_method.upper()}...")
            peft_config = self.config.get_peft_config()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            print("  - No PEFT method applied (full fine-tuning)")

        print(f"   Model loaded: {type(self.model).__name__}")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_datasets(self):
        print("\n[3/4] Loading datasets...")

        data_config = self.config.data

        # Determine file format
        if data_config.sft_train_file.endswith('.parquet'):
            data_files = {
                "train": data_config.sft_train_file,
                "validation": data_config.sft_val_file,
            }
            dataset = load_dataset("parquet", data_files=data_files)
        elif data_config.sft_train_file.endswith('.jsonl'):
            data_files = {
                "train": data_config.sft_train_file,
                "validation": data_config.sft_val_file,
            }
            dataset = load_dataset("json", data_files=data_files)
        else:
            raise ValueError(f"Unsupported file format: {data_config.sft_train_file}")

        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["validation"]

        print(f"✓ Datasets loaded:")
        print(f"  - Train samples: {len(self.train_dataset):,}")
        print(f"  - Validation samples: {len(self.eval_dataset):,}")
        print(f"  - Features: {self.train_dataset.features}")

        # Show sample
        print("\n  Sample data:")
        sample = self.train_dataset[0]
        if "messages" in sample:
            print(f"    Messages: {str(sample['messages'])[:200]}...")
        elif "text" in sample:
            print(f"    Text: {sample['text'][:200]}...")

    def _setup_trainer(self):
        print("\n[4/4] Setting up trainer...")
        training_config = self.config.get_training_config()

        data_collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            instruction_template="<|im_start|>user\n",
            response_template="<|im_start|>assistant\n",
            mlm=False,
            ignore_index=-100,
        )

        if self.config.data.compute_metrics:
            metrics_callback = TranslationMetricsCallback(
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset,
                num_samples=10,
                max_new_tokens=512,
                system_prompt=self.config.data.system_prompt
            )
            self.callbacks.append(metrics_callback)

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator
            callbacks=self.callbacks
        )
    
    def train(self):
        print("STARTING TRAINING")

        # Train
        self.trainer.train()

        print("TRAINING COMPLETED")

    def evaluate(self):
        """Run evaluation"""
        print("RUNNING EVALUATION")

        metrics = self.trainer.evaluate()

        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        """Save the trained model"""
        if output_dir is None:
            output_dir = self.config.sft.output_dir

        print(f"\n[SAVING] Saving model to {output_dir}...")

        # Save model and tokenizer
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save PEFT adapter if applicable
        if self.config.peft_method != "none":
            adapter_dir = os.path.join(output_dir, "adapter")
            self.model.save_pretrained(adapter_dir)
            print(f"✓ PEFT adapter saved to {adapter_dir}")

        print(f"✓ Model saved to {output_dir}")

def reward_match_chatgpt(prompts, completions, **kwargs):
    """
    Thưởng nếu bản dịch giống với bản dịch mẫu của ChatGPT.
    """
    # Lấy dữ liệu từ cột đã đổi tên
    chatgpt_data = kwargs.get("samples", []) 
    
    rewards = []
    for pred, gpt_samples in zip(completions, chatgpt_data):
        if not gpt_samples:
            rewards.append(0.0)
            continue
            
        # Trong file jsonl, 'completions' là list các dict: [{"text": "...", "score": ...}]
        # Ta lấy text của câu có điểm cao nhất hoặc lấy tất cả làm tham chiếu
        # Ví dụ lấy câu đầu tiên trong list làm chuẩn
        gpt_ref_text = gpt_samples[0]['text'] 
        
        # Tính BLEU score so với câu của ChatGPT
        score = sacrebleu.sentence_bleu(pred, [gpt_ref_text]).score
        rewards.append(score / 20.0)
        
    return rewards
        
class TrainerGRPO:
    def __init__(self, config: MainConfig):     
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
        self.config = config
        
    def _load_checkpoint(self):
        print("\n[1/3] Loading checkpoint...")
        if self.config.peft_method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("  ✓ Using 4-bit quantization (QLoRA)")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            quantization_config=bnb_config if self.config.peft_method == "qlora" else None,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(model, self.config.model.load_adapter_from)
        print(f"✓ Model loaded: {self.config.model.load_adapter_from}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.load_adapter_from, trust_remote_code=True)
        print(f"✓ Tokenizer loaded: {self.config.model.load_adapter_from}")
    
    def _load_dataset(self):
        print("\n[2/3] Loading dataset...")
        
        # Load training data
        train_path = self.config.data.grpo_train_data
        file_ext = Path(train_path).suffix.lower()
        
        if file_ext == '.parquet':
            self.train_dataset = load_dataset('parquet', data_files=train_path)['train']
        elif file_ext in ['.jsonl', '.json']:
            self.train_dataset = load_dataset('json', data_files=train_path)['train']
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        def normalize(sample):
            if 'completions' in sample and len(sample['completions']) > 0:
                first = sample['completions'][0]
                if 'content' in first and 'text' not in first:
                    sample['completions'] = [
                        {'text': c.get('content', ''), 'score': c['score']}
                        for c in sample['completions']
                    ]
            return sample
        
        self.train_dataset = self.train_dataset.map(normalize)
        
        print(f"  Train samples: {len(self.train_dataset)}")
        
        # Load validation data (optional)
        if self.config.data.grpo_validation_data:
            val_path = self.config.data.grpo_validation_data
            if os.path.exists(val_path):
                file_ext = Path(val_path).suffix.lower()
                
                if file_ext == '.parquet':
                    self.eval_dataset = load_dataset('parquet', data_files=val_path)['train']
                elif file_ext in ['.jsonl', '.json']:
                    self.eval_dataset = load_dataset('json', data_files=val_path)['train']
                
                self.eval_dataset = self.eval_dataset.map(normalize)
                print(f"  Eval samples: {len(self.eval_dataset)}")

    def _setup_trainer(self):
        print("\n[4] Setting up trainer...")
        self.trainer = GRPOTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if self.eval_dataset else None,
            args=self.config.grpo,
            reward_funcs=reward_match_chatgpt,
        )

    def setup(self):
        self._load_checkpoint()
        self._load_dataset()
        self._setup_trainer()

    def train(self):
        print("\n[5] Training...")
        self.trainer.train()
        print("✓ Training completed")

    def evaluate(self):
        print("\n[6] Evaluating...")
        metrics = self.trainer.evaluate()
        print("✓ Evaluation completed")
        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        print("\n[7] Saving model...")
        if output_dir is None:
            output_dir = self.config.grpo.output_dir
        self.trainer.save_model(output_dir)
        print("✓ Model saved")