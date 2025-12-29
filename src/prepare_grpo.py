import os
import time
import json
import random
import numpy as np
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

# OpenAI Import
try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    raise ImportError("Please install: pip install openai")

# Scorer Imports
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    print("⚠️ COMET not found. Using BLEU fallback.")
    COMET_AVAILABLE = False
from sacrebleu import BLEU

class OpenAIGenerator:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", base_url: str = None):
        """
        Khởi tạo OpenAI Client.
        base_url: Có thể dùng để trỏ sang vLLM, DeepSeek, hoặc OpenRouter nếu cần.
        """
        print(f"🔧 Initializing OpenAI Client [{model_name}]...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        num_candidates: int = 4,
        temperature: float = 0.7, 
        max_tokens: int = 512
    ) -> List[str]:
        """
        Sinh nhiều candidates cùng lúc dùng tham số 'n'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                n=num_candidates, 
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )

            candidates = [choice.message.content.strip() for choice in response.choices]
            return candidates

        except RateLimitError:
            print("⚠️ Rate Limit hit. Sleeping 10s...")
            time.sleep(10)
            return []
        except APIError as e:
            print(f"⚠️ OpenAI API Error: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Unexpected Error: {e}")
            return []

class CometScorer:
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        print(f"🔧 Loading COMET {model_name}...")
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
    def score(self, sources: List[str], candidates: List[str], references: List[str]) -> List[float]:
        data = [{"src": s, "mt": c, "ref": r} for s, c, r in zip(sources, candidates, references)]
        output = self.model.predict(data, batch_size=8, gpus=0, progress_bar=False)
        return output.scores

class BLEUScorer:
    def __init__(self):
        self.bleu = BLEU()

    def score(self, sources: List[str], candidates: List[str], references: List[str]) -> List[float]:
        scores = []
        for cand, ref in zip(candidates, references):
            try:
                s = self.bleu.sentence_score(cand, [ref]).score / 100.0
                scores.append(s)
            except:
                scores.append(0.0)
        return scores

def load_parallel_data(en_file: str, vi_file: str, max_samples: int = 2000) -> List[Dict]:
    print(f"\n📂 Loading data...")
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(vi_file, 'r', encoding='utf-8') as f:
        vi_lines = [l.strip() for l in f if l.strip()]

    if len(en_lines) != len(vi_lines):
        min_len = min(len(en_lines), len(vi_lines))
        en_lines, vi_lines = en_lines[:min_len], vi_lines[:min_len]

    # Tạo dataset balanced
    indices = list(range(len(en_lines)))
    random.shuffle(indices)
    indices = indices[:max_samples]

    data = []
    half = len(indices) // 2

    # Nửa đầu En->Vi
    for idx in indices[:half]:
        data.append({
            'source': en_lines[idx], 'target': vi_lines[idx], 'direction': 'en-vi'
        })
    # Nửa sau Vi->En
    for idx in indices[half:]:
        data.append({
            'source': vi_lines[idx], 'target': en_lines[idx], 'direction': 'vi-en'
        })

    print(f"✓ Loaded {len(data)} samples (Balanced En-Vi / Vi-En)")
    return data

def get_medical_prompts(source: str, direction: str) -> Tuple[str, str]:
    """Tạo System Prompt chuyên dụng cho Y tế"""

    system_prompt = (
        "You are a professional medical translator. "
        "Strictly adhere to standard medical terminology (ICD-10, SNOMED). "
        "Do NOT translate proper names of drugs/compounds unless there is a standard local equivalent. "
        "Maintain professional tone, precision. "
        "Rules: Keep abbreviations as-is (e.g., V.A, V.a, PTA, Type B/C/As). "
        "Preserve all numbers, %, ±, ≥, ≤,... parentheses, and punctuation. "
        "Keep numbers, units (mg, ml, %), and vital signs exact."
    )

    if direction == 'en-vi':
        user_prompt = f"Translate the following medical text from English to Vietnamese:\n\n{source}"
    else:
        user_prompt = f"Translate the following medical text from Vietnamese to English:\n\n{source}"

    return system_prompt, user_prompt

def save_jsonl(data: List[Dict], filename: str):
    with open(filename, 'a', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


