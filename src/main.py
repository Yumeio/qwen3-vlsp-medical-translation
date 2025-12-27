import os
import torch
from trainer import Trainer
from config import create_sft_full_config

def train_sft(strategy: str):
    config = create_sft_full_config(strategy=strategy)
    
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    
    if trainer.callbacks and hasattr(trainer.callbacks[0], 'best_scores'):
        print("\n🏆 Best Scores:")
        for metric, info in trainer.callbacks[0].best_scores.items():
            print(f"   {metric.upper()}: {info['score']:.2f} (step {info['step']})")
            
if __name__ == "__main__":
    train_sft("lora")