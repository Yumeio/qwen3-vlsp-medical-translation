import os
import logging
from pathlib import Path
from typing import Optional
from datasets import load_dataset, DatasetDict

from arguments import MainConfig
from config import create_sft_full_config
from trainer import load_tokenizer, load_model, create_trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_datasets(config: MainConfig) -> DatasetDict:
    """Load preprocessed SFT datasets from parquet/jsonl files"""
    
    logger.info("Loading SFT datasets...")
    
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
    
    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Eval samples: {len(eval_dataset):,}")
    
    # Apply max samples limit if specified
    if config.data.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), config.data.max_train_samples))
        )
    
    if config.data.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), config.data.max_eval_samples))
        )
    
    logger.info(f"After limits - Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    return {
        'train': train_dataset,
        'eval': eval_dataset
    }


def setup_training_pipeline(
    config: Optional[MainConfig] = None,
    peft_strategy: str = "lora"
) -> tuple:
    """
    Setup complete training pipeline
    
    Args:
        config: MainConfig instance. If None, creates default config
        peft_strategy: One of "lora", "qlora", "ia3"
    
    Returns:
        (trainer, config, tokenizer, model)
    """
    
    # Create config if not provided
    if config is None:
        config = create_sft_full_config(strategy=peft_strategy)
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Mode: {config.training_mode.upper()}")
    logger.info(f"   PEFT: {config.peft_method.upper()}")
    logger.info(f"   Model: {config.model.model_name_or_path}")
    logger.info(f"   Batch size: {config.sft.per_device_train_batch_size}")
    logger.info(f"   Effective BS: {config.sft.per_device_train_batch_size * config.sft.gradient_accumulation_steps}")
    logger.info(f"   Mixed precision: bf16={config.sft.bf16}")
    logger.info(f"   Output dir: {config.sft.output_dir}")
    logger.info(f"   Run name: {config.sft.run_name}")
    
    # Load tokenizer
    logger.info("\n📋 Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    logger.info(f"   Vocab size: {len(tokenizer):,}")
    
    # Load model
    logger.info("\n🤖 Loading model...")
    model = load_model(config)
    logger.info(f"   Model loaded: {type(model).__name__}")
    logger.info(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    logger.info("\n📊 Loading datasets...")
    datasets = load_datasets(config)
    
    # Create trainer
    logger.info("\n🏋️ Creating trainer...")
    trainer = create_trainer(
        main_cfg=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets['train'],
        eval_dataset=datasets['eval'],
    )
    
    logger.info(f"   Trainer ready with {len(datasets['train']):,} training samples")
    
    return trainer, config, tokenizer, model


def train(
    config: Optional[MainConfig] = None,
    peft_strategy: str = "lora",
    resume_from_checkpoint: Optional[str] = None
):
    """
    Train the model using SFT
    
    Args:
        config: MainConfig instance. If None, creates default config
        peft_strategy: One of "lora", "qlora", "ia3"
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    
    logger.info("=" * 60)
    logger.info("🚀 Starting SFT Training Pipeline")
    logger.info("=" * 60)
    
    # Setup pipeline
    trainer, config, tokenizer, model = setup_training_pipeline(
        config=config,
        peft_strategy=peft_strategy
    )
    
    # Training info
    logger.info("\n📈 Training Info:")
    logger.info(f"   Epochs: {config.sft.num_train_epochs}")
    logger.info(f"   LR: {config.sft.learning_rate}")
    logger.info(f"   LR scheduler: {config.sft.lr_scheduler_type}")
    logger.info(f"   Warmup ratio: {config.sft.warmup_ratio}")
    logger.info(f"   Save strategy: {config.sft.save_strategy}")
    logger.info(f"   Eval strategy: {config.sft.eval_strategy}")
    logger.info(f"   Output dir: {config.sft.output_dir}")
    logger.info(f"   Run name: {config.sft.run_name}")
    
    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("▶️  Starting training...")
    logger.info("=" * 60 + "\n")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info("\n💾 Saving final model...")
    trainer.save_model(config.sft.output_dir)
    
    # Save results
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Final loss: {train_result.training_loss:.4f}")
    logger.info(f"   Checkpoint saved to: {config.sft.output_dir}")
    
    return trainer, config


def evaluate_only(
    config: Optional[MainConfig] = None,
):
    """Evaluate model without training"""
    
    logger.info("=" * 60)
    logger.info("📊 Evaluation Only Mode")
    logger.info("=" * 60)
    
    # Setup pipeline
    trainer, config, tokenizer, model = setup_training_pipeline(config=config)
    
    # Evaluate
    logger.info("\nRunning evaluation...")
    metrics = trainer.evaluate()
    
    logger.info(f"\n✅ Evaluation complete!")
    logger.info(f"   Eval loss: {metrics.get('eval_loss', 'N/A')}")
    
    return metrics, config


# Example usage patterns
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for Qwen Medical Model")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Training or evaluation mode"
    )
    parser.add_argument(
        "--peft",
        choices=["lora", "qlora", "ia3"],
        default="lora",
        help="PEFT method"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        trainer, config = train(
            peft_strategy=args.peft,
            resume_from_checkpoint=args.resume
        )
    else:
        metrics, config = evaluate_only()
    
    logger.info("\n✨ Done!")