import torch
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from transformers import TrainingArguments as HfTrainingArguments
from trl import SFTConfig, GRPOConfig as TRL_GRPOConfig
from peft import PeftType, TaskType

@dataclass
class DataConfig:
    # Directories paths
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"
    
    # File paths
    sft_train_file: Optional[str] = None
    sft_val_file: Optional[str] = None
    grpo_train_file: Optional[str] = None
    grpo_val_file: Optional[str] = None
    
    # Output format
    output_format: Literal["jsonl", "parquet"] = "parquet"
    
    # Field names
    sft_text_field: str = "messages"  
    grpo_prompt_field: str = "prompt"
    grpo_completions_field: str = "completions"
    
    # System prompt
    system_prompt: str = (
        "You are an AI learning support specialist."
        "Your task is to provide accurate translation and explanation of algorithmic terminology"
        "between English and Vietnamese. Always respond professionally, concisely, accurately, and clearly."
        "Rules: Keep abbreviations as-is (e.g., V.A, V.a, PTA, Type B/C/As). "
        "Preserve all numbers, %, ±, ≥, ≤,... parentheses, and punctuation. "
    )
    
    # Prompts
    prompts_en_to_vi: List[str] = field(default_factory=lambda: [
        "Translate the following medical text to Vietnamese:",
        "Convert this content into Vietnamese:",
        "Please translate this medical term into Vietnamese:",
    ])
    prompts_vi_to_en: List[str] = field(default_factory=lambda: [
        "Translate the following medical text to English:",
        "Convert this content into English:",
        "Please translate this medical term into English:",
    ])
    
    # Quality filters
    min_words: int = 2
    max_words: int = 256
    min_length_ratio: float = 0.25
    max_length_ratio: float = 4.0
    
    # Preprocessing
    preprocessing_num_workers: int = 16
    max_trainval_samples: Optional[int] = 100000
    max_eval_samples: Optional[int] = None
    val_ratio: float = 0.1

    def __post_init__(self):
        """Construct file paths if not provided"""
        ext = self.output_format
        
        if self.sft_train_file is None:
            self.sft_train_file = f"{self.processed_dir}/sft_train.{ext}"
        if self.sft_val_file is None:
            self.sft_val_file = f"{self.processed_dir}/sft_validation.{ext}"
        if self.grpo_train_file is None:
            self.grpo_train_file = f"{self.processed_dir}/grpo_train.{ext}"
        if self.grpo_val_file is None:
            self.grpo_val_file = f"{self.processed_dir}/grpo_validation.{ext}"
            
@dataclass
class ModelConfig:
    # Model selection
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    tokenizer_name_or_path: Optional[str] = None
    cache_dir: str = "./model_cache"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    tokenizer_padding_side: str = "right"
    
    # Mixed Precision
    torch_dtype: Literal[
        "auto", "float32", "float16", "bfloat16"
    ] = "bfloat16"
    
    # Device
    device_map: str = "auto"
    
    # Context length
    max_seq_length: int = 512
    
    # Special tokens
    add_eos_token: bool = True
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
            
    def get_torch_dtype(self):
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, "auto")

@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration"""
    type: str = PeftType.LORA
    
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization (for QLoRA)
    use_quantization: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # Task type
    task_type: TaskType = TaskType.CAUSAL_LM
    
    # Bias
    bias: Literal["none", "all", "lora_only"] = "none"
    
    def __post_init__(self):
        if self.load_in_4bit:
            self.use_quantization = True
        if self.load_in_8bit:
            self.use_quantization = True

        self.type = "QLORA" if self.use_quantization else "LORA"

    def to_peft_config(self):
        from peft import LoraConfig
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )
        
@dataclass
class IA3Config:
    """IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) configuration"""
    type: str = PeftType.IA3

    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "k_proj", "v_proj", "down_proj"
    ])
    
    feedforward_modules: List[str] = field(default_factory=lambda: [
        "down_proj"
    ])
    
    # Task type
    task_type: TaskType = TaskType.CAUSAL_LM

    def to_peft_config(self): 
        """Convert to PEFT IA3Config"""
        from peft import IA3Config as PeftIA3Config
        
        return PeftIA3Config(
            target_modules=self.target_modules,
            feedforward_modules=self.feedforward_modules,
            task_type=self.task_type,
        )
        
@dataclass
class SFTTrainingConfig(SFTConfig):
    """Supervised Fine-Tuning configuration"""
    
    output_dir: str = "./outputs/sft"
    run_name: str = "qwen-sft"
    
    # Training schedule
    num_train_epochs: float = 3.0
    max_steps: int = -1
    
    # Batch sizes and gradient accumulation
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 16
    
    # Optimizer
    optim: str = "adamw_torch_fused"  # or "adamw_8bit", "paged_adamw_8bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.998
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    
    # Logging and saving
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 2000
    eval_strategy: str = "steps"
    eval_steps: int = 2000
    report_to: Literal["wandb", "tensorboard", "none"] = "wandb"
    
    # SFT specific
    packing: bool = False
    max_seq_length: int = 512
    dataset_text_field: str = "messages"  # Changed to "messages" for apply_chat_template
    dataset_num_proc: int = 4

    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    tf32: bool = False
    
    # DataLoader settings
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.effective_batch_size = (
            self.per_device_train_batch_size * 
            self.gradient_accumulation_steps
        )

@dataclass
class GRPOConfig:
    pass

@dataclass
class GRPOTrainingConfig(TRL_GRPOConfig):
    pass

@dataclass 
class MainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ia3: Optional[IA3Config] = None
    sft: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    grpo: Optional[GRPOTrainingConfig] = None
    
    training_mode: Literal["sft", "grpo"] = "sft"
    peft_method: Literal["lora", "qlora", "ia3", "none"] = "lora"
    
    # Allow custom output_dir override
    custom_output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize optional configs and set output_dir based on peft_method"""
        # Initialize optional PEFT configs
        if self.peft_method == "ia3" and self.ia3 is None:
            self.ia3 = IA3Config()
        
        if self.training_mode == "grpo" and self.grpo is None:
            self.grpo = GRPOTrainingConfig()
        
        # Set output_dir based on training mode and peft method
        self._set_output_dir()
        self._set_run_name()
    
    def _set_output_dir(self):
        """Set output_dir dynamically based on training mode and PEFT method"""
        if self.custom_output_dir is not None:
            self.sft.output_dir = self.custom_output_dir
            self.sft.run_name = self.custom_output_dir.split("/")[-1]
            return
        
        if self.training_mode == "sft":
            base_name = f"qwen_sft_{self.peft_method}"
        elif self.training_mode == "grpo":
            base_name = f"qwen_grpo_{self.peft_method}"
        else:
            base_name = f"qwen_{self.training_mode}_{self.peft_method}"
        
        self.sft.output_dir = f"./outputs/{base_name}"
        
        self.sft.run_name = base_name
    
    def _set_run_name(self):
        """Set run_name dynamically based on training mode and PEFT method"""
        if self.custom_output_dir is not None:
            self.sft.run_name = self.custom_output_dir.split("/")[-1]
            return
        
        if self.training_mode == "sft":
            base_name = f"qwen_sft_{self.peft_method}"
        elif self.training_mode == "grpo":
            base_name = f"qwen_grpo_{self.peft_method}"
        else:
            base_name = f"qwen_{self.training_mode}_{self.peft_method}"
        
        self.sft.run_name = base_name

    def get_training_config(self):
        """Get the appropriate training config based on mode"""
        if self.training_mode == "sft":
            return self.sft
        elif self.training_mode == "grpo":
            return self.grpo
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def get_peft_config(self):
        """Get the appropriate PEFT config"""
        if self.peft_method in ["lora", "qlora"]:
            return self.lora.to_peft_config()
        elif self.peft_method == "ia3":
            return self.ia3.to_peft_config()
        elif self.peft_method == "none":
            return None
        else:
            raise ValueError(f"Unknown PEFT method: {self.peft_method}")