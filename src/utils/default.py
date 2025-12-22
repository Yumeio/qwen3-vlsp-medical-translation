import os
import logging
import torch
import random 
import bitsandbytes as bnb

def setup_logger():
    """Setup logger configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger
    
SEED_VALUE = 42

def set_seed(seed: int = SEED_VALUE):
    """Set random to ensure result reproducible"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE) # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Current seed: {torch.initial_seed()}")
    
def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def find_all_linear_names(model):
    """Find all linear layer names in the model"""
    linear_names = set()
    for name, module in model.named_modules():
        # Check if the module is an instance of nn.Linear or bnb.nn.Linear4bit
        if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
            names = name.split('.')
            linear_names.add(names[0] if len(names) == 1 else names[-1])
            
        # Bf16 is used in lm_head 
        if "lm_head" in linear_names:
            linear_names.remove("lm_head")
            
        return list(linear_names)
    
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || "
        f"All params: {all_param} || "
        f"Trainable%: {100 * trainable_params / all_param}"
    )