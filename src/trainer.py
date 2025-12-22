import os 
import torch
from typing import Literal, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer
from data_collator import DataCollatorForChatML

from arguments import (
    ModelConfig,
    DataConfig,
    MainConfig
)

os.environ["TOKENIZER_PARALLELISM"] = "False"

def load_tokenizer(
    main_cfg: MainConfig
) -> PreTrainedTokenizer:
    
    model_cfg = main_cfg.model
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        trust_remote_code=model_cfg.trust_remote_code,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_sft_dataset(config: MainConfig) -> DatasetDict:
    """Load preprocessed SFT datasets from parquet/jsonl files"""
    
    print("Loading SFT datasets...")
    
    # Load train dataset
    train_dataset = load_dataset(
        config.data.output_format,
        data_files=config.data.sft_train_file
    )['train']
    
    # Load validation dataset
    eval_dataset = load_dataset(
        config.data.output_format,
        data_files=config.data.sft_val_file
    )['train']
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Eval samples: {len(eval_dataset):,}")
    
    # Apply max samples limit if specified
    if config.data.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), config.data.max_train_samples))
        )
    
    if config.data.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), config.data.max_eval_samples))
        )
    
    print(f"After limits - Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    return {
        'train': train_dataset,
        'eval': eval_dataset
    }

def get_quantization_config(
    main_cfg: MainConfig
):
    if main_cfg.peft_method != "qlora" or not main_cfg.lora.load_in_4bit:
        return None
    
    compute_dtype = getattr(
        torch, main_cfg.lora.bnb_4bit_compute_dtype
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=main_cfg.lora.load_in_4bit,
        load_in_8bit=main_cfg.lora.load_in_8bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=main_cfg.lora.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=main_cfg.lora.bnb_4bit_use_double_quant,
    )

    return bnb_config

def load_model(
    main_cfg: MainConfig
) -> PreTrainedModel:

    bnb_config = get_quantization_config(main_cfg)
    
    # Mixed precision
    model_cfg = main_cfg.model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        torch_dtype=model_cfg.get_torch_dtype(),
        device_map=model_cfg.device_map,
        quantization_config=bnb_config,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    
    if main_cfg.peft_method == "qlora" and main_cfg.lora.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    return model

def setup_peft_model(
    main_cfg: MainConfig,
    model: PreTrainedModel
) -> PreTrainedModel:
    """Apply PEFT methods to model"""
    from peft import LoraConfig, IA3Config, TaskType
    
    if main_cfg.peft_method in ["lora", "qlora"]:
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=main_cfg.lora.r,
            lora_alpha=main_cfg.lora.lora_alpha,
            lora_dropout=main_cfg.lora.lora_dropout,
            target_modules=main_cfg.lora.target_modules,
            bias=main_cfg.lora.bias,
        )
    elif main_cfg.peft_method == "ia3":
        peft_cfg = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=main_cfg.ia3.target_modules,
            feedforward_modules=main_cfg.ia3.feedforward_modules,
        )
    else:
        return model

    return get_peft_model(model, peft_cfg)

def create_trainer(
    main_cfg: MainConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    """Create and return an SFT trainer instance"""
    from data_collator import DataCollatorForChatML
    
    # Setup PEFT if needed
    if main_cfg.peft_method != "none":
        model = setup_peft_model(main_cfg, model)
        model.print_trainable_parameters()
    
    # Create data collator for ChatML format
    data_collator = DataCollatorForChatML(
        tokenizer=tokenizer,
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=main_cfg.sft,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field=main_cfg.data.sft_text_field,
    )
    
    return trainer