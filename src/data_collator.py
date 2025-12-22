import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass  
class DataCollatorForChatML:
    """
    Most robust version using TRL-style approach
    Only computes loss on completion (assistant response)
    """
    
    tokenizer: PreTrainedTokenizerBase
    instruction_template: str = "<|im_start|>user\n"
    response_template: str = "<|im_start|>assistant\n"
    mlm: bool = False
    ignore_index: int = -100
    
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding=True,
        )
        
        # Create labels
        labels = batch["input_ids"].clone()
        
        # Get response template ids
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )
        
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            
            # Find response template
            for idx in range(len(labels[i]) - len(response_token_ids) + 1):
                if torch.equal(
                    labels[i][idx : idx + len(response_token_ids)],
                    torch.tensor(response_token_ids)
                ):
                    response_token_ids_start_idx = idx
                    break
            
            if response_token_ids_start_idx is None:
                labels[i, :] = self.ignore_index
            else:
                labels[i, : response_token_ids_start_idx + len(response_token_ids)] = self.ignore_index
                
                im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
                end_positions = (labels[i] == im_end_token).nonzero(as_tuple=True)[0]
                
                if len(end_positions) > 0:
                    for end_pos in end_positions:  
                        if end_pos > response_token_ids_start_idx:
                            labels[i, end_pos] = self.ignore_index
                            # Mask everything after end token (padding, next turn, etc)
                            labels[i, end_pos + 1:] = self.ignore_index
                            break
        
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        batch["labels"] = labels
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.torch_call(features)