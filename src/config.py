from arguments import MainConfig
from typing import Literal

def create_sft_full_config(
    strategy: Literal["lora", "qlora", "ia3"] = "lora"
) -> MainConfig:
    config = MainConfig(
        training_mode="sft",
        peft_method=strategy
    )
    return config

def create_grpo_full_config(
    strategy: Literal["lora", "qlora", "ia3"] = "lora"
) -> MainConfig:
    config = MainConfig(
        training_mode="grpo",
        peft_method=strategy
    )
    return config