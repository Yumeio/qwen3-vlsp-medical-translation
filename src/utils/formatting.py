import random
from typing import List, Dict

def format_chatml(messages: List[Dict[str, str]]) -> str:
    chatml_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        chatml_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return chatml_text.strip()

def create_sft_sample(
    tokenizer,
    source_text: str,
    target_text: str,
    system_prompt: str,
    prompts: List[str]
) -> Dict:
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{random.choice(prompts)}\n{source_text}"},
        {"role": "assistant", "content": target_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return {"messages": text}

def create_grpo_sample(
    source_text: str,
    candidates: List[str],
    scores: List[float],
    system_prompt: str,
    prompts: List[str]
) -> Dict:   
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