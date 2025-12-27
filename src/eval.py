import os
import argparse
import logging
import torch
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel

from arguments import MainConfig
from trainer import load_tokenizer, load_model, get_quantization_config
from data_collator import DataCollatorForChatML

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZER_PARALLELISM"] = "False"

def load_eval_dataset(config: MainConfig) -> Dataset:
    """Load evaluation dataset"""
    logger.info("Loading evaluation dataset...")
    
    eval_dataset = load_dataset(
        config.data.output_format,
        data_files=config.data.sft_val_file
    )['train']
    
    logger.info(f"Loaded {len(eval_dataset):,} eval samples")
    
    if config.data.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), config.data.max_eval_samples))
        )
        logger.info(f"Limited to {len(eval_dataset):,} samples")
    
    return eval_dataset


def load_finetuned_model(
    checkpoint_dir: str,
    config: MainConfig
) -> tuple:
    """
    Load fine-tuned model with LoRA/IA3 adapters
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        config: MainConfig instance
    
    Returns:
        (model, tokenizer) tuple
    """
    logger.info(f"\n📦 Loading model from: {checkpoint_dir}")
    
    # Load tokenizer
    logger.info("   Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    
    # Load base model
    logger.info("   Loading base model...")
    model_cfg = config.model
    
    bnb_config = get_quantization_config(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        cache_dir=model_cfg.cache_dir,
        torch_dtype=model_cfg.get_torch_dtype(),
        device_map=model_cfg.device_map,
        quantization_config=bnb_config,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    
    # Load LoRA/IA3 adapters if they exist
    if config.peft_method in ["lora", "qlora"]:
        adapter_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(adapter_path):
            logger.info("   Loading LoRA adapters...")
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            model = model.merge_and_unload()  # Merge adapters into base model
        else:
            logger.warning(f"   LoRA config not found at {adapter_path}")
    
    elif config.peft_method == "ia3":
        adapter_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(adapter_path):
            logger.info("   Loading IA3 adapters...")
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            model = model.merge_and_unload()
        else:
            logger.warning(f"   IA3 config not found at {adapter_path}")
    
    model.eval()
    logger.info(f"   ✅ Model loaded: {type(model).__name__}")
    
    return model, tokenizer


def compute_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    config: MainConfig,
    batch_size: int = 16
) -> dict:
    """
    Compute evaluation metrics on eval dataset
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with metrics
    """
    logger.info("\n📊 Computing evaluation metrics...")
    
    total_loss = 0.0
    num_samples = 0
    num_batches = 0
    
    device = next(model.parameters()).device
    
    # Create data collator
    data_collator = DataCollatorForChatML(
        tokenizer=tokenizer,
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
    )
    
    # Prepare data
    eval_samples = []
    for sample in eval_dataset:
        eval_samples.append({"text": sample["text"]})
    
    # Evaluate in batches
    with torch.no_grad():
        for i in range(0, len(eval_samples), batch_size):
            batch_samples = eval_samples[i:i+batch_size]
            
            # Collate batch
            batch = data_collator(batch_samples)
            
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            
            # Accumulate loss
            total_loss += outputs.loss.item() * len(batch_samples)
            num_samples += len(batch_samples)
            num_batches += 1
            
            if (num_batches % 10) == 0:
                logger.info(f"   Processed {num_samples:,} samples...")
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    metrics = {
        "eval_loss": avg_loss,
        "eval_samples": num_samples,
        "eval_batches": num_batches,
    }
    
    return metrics


def evaluate(
    checkpoint_dir: Optional[str] = None,
    config: Optional[MainConfig] = None,
    peft_strategy: str = "lora"
):
    """
    Main evaluation function
    
    Args:
        checkpoint_dir: Path to checkpoint directory. If None, uses default from config
        config: MainConfig instance. If None, creates default config
        peft_strategy: PEFT strategy if config is None
    """
    
    logger.info("=" * 70)
    logger.info("📊 Model Evaluation")
    logger.info("=" * 70)
    
    # Create config if not provided
    if config is None:
        from config import create_sft_full_config
        config = create_sft_full_config(strategy=peft_strategy)
    
    # Determine checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.sft.output_dir
        logger.info(f"Using output_dir from config: {checkpoint_dir}")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_dir):
        logger.error(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        logger.info(f"   Please ensure training has completed or provide correct path")
        return None
    
    logger.info(f"\n🔧 Configuration:")
    logger.info(f"   Mode: {config.training_mode.upper()}")
    logger.info(f"   PEFT: {config.peft_method.upper()}")
    logger.info(f"   Model: {config.model.model_name_or_path}")
    logger.info(f"   Checkpoint: {checkpoint_dir}")
    
    # Load dataset
    logger.info(f"\n📂 Data:")
    eval_dataset = load_eval_dataset(config)
    
    # Load model and tokenizer
    logger.info(f"\n🤖 Model:")
    model, tokenizer = load_finetuned_model(checkpoint_dir, config)
    
    # Compute metrics
    metrics = compute_metrics(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        config=config,
        batch_size=16
    )
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("📈 Evaluation Results")
    logger.info("=" * 70)
    logger.info(f"   Eval Loss: {metrics['eval_loss']:.4f}")
    logger.info(f"   Eval Samples: {metrics['eval_samples']:,}")
    logger.info(f"   Eval Batches: {metrics['eval_batches']:,}")
    logger.info("=" * 70)
    
    # Save results
    results_file = os.path.join(checkpoint_dir, "eval_results.txt")
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Eval Loss: {metrics['eval_loss']:.4f}\n")
        f.write(f"Eval Samples: {metrics['eval_samples']:,}\n")
        f.write(f"Eval Batches: {metrics['eval_batches']:,}\n")
        f.write(f"Checkpoint: {checkpoint_dir}\n")
        f.write(f"PEFT Method: {config.peft_method}\n")
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    
    return metrics