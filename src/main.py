from config import create_sft_full_config,create_grpo_full_config
from trainer import TrainerSFT, TrainerGRPO

def train_sft(strategy: str):
    config = create_sft_full_config(strategy=strategy)
    
    trainer = TrainerSFT(config)
    trainer.setup()
    trainer.train()
    trainer.evaluate()
    trainer.save_model()

def train_grpo(sft_strategy: str):
    config = create_grpo_full_config(
        f"./outputs/qwen_sft_{sft_strategy}",
        sft_strategy
    )
    config.data.grpo_train_data = "grpo.jsonl"
    trainer = TrainerGRPO(config)
    trainer.setup()
    trainer.train()
    trainer.save_model()

# if __name__ == "__main__":
#     strategy = "lora"
#     train_sft(strategy)
#     train_grpo(strategy)