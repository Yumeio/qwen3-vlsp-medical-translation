import random
import string as str_lib
from typing import List, Dict
from default import set_seed

def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format messages as ChatML"""
    chatml_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        chatml_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return chatml_text.strip()

def create_sft_sample(
    source_text: str,
    target_text: str,
    system_prompt: str,
    prompts: List[str]
) -> Dict[str, str]:
    """
    Create a single SFT training sample
    
    Args:
        source_text: Source language text
        target_text: Target language text
        system_prompt: System prompt
        prompts: List of possible user prompts
    
    Returns:
        {"text": chatml_formatted_string}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{random.choice(prompts)}\n{source_text}"},
        {"role": "assistant", "content": target_text}
    ]
    
    return {"text": format_chatml(messages)}

def create_grpo_sample(
    source_text: str,
    candidates: List[str],
    scores: List[float],
    system_prompt: str,
    prompts: List[str]
) -> Dict:
    """
    Create a single GRPO training sample
    
    Args:
        source_text: Source text to translate
        candidates: List of candidate translations
        scores: List of scores for each candidate
        system_prompt: System prompt
        prompts: List of possible user prompts
    
    Returns:
        {
            "prompt": chatml_prompt,
            "completions": [{"text": ..., "score": ...}, ...]
        }
    """
    
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{random.choice(prompts)}\n{source_text}"},
    ]
    
    # Generation prompt 
    prompt = format_chatml(prompt_messages)
    prompt += "\n<|im_start|>assistant\n"

    completions = [
        {"text": candidate, "score": score}
        for candidate, score in zip(candidates, scores)
    ]
    
    return {
        "prompt": prompt,
        "completions": completions
    }