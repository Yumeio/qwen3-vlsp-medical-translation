import os
import json
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import Literal, Tuple, List

from config import DataConfig
from utils.default import set_seed
from utils.text_processing import process_pair, check_quality
from utils.formatting import create_sft_sample

class SFTDatasetBuilder:
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_raw_data(self, split: str) -> Tuple[List[str], List[str]]:
        """Load raw parallel data"""
        if split == "train":
            en_file = os.path.join(self.config.raw_dir, "train.en.txt")
            vi_file = os.path.join(self.config.raw_dir, "train.vi.txt")
        elif split == "validation":
            en_file = os.path.join(self.config.raw_dir, "public_test.en.txt")
            vi_file = os.path.join(self.config.raw_dir, "public_test.vi.txt")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        ds = load_dataset(
            "text",
            data_files={"en": en_file, "vi": vi_file},
            streaming=False
        )
        
        return ds["en"]["text"], ds["vi"]["text"]
    
    def process_and_filter(
        self, 
        en_raw: List[str], 
        vi_raw: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Process and filter pairs through all quality checks
        
        Returns:
            List of (en_clean, vi_clean) tuples that passed all filters
        """
        
        total_initial = len(en_raw)
        print(f"Initial Dataset: {total_initial:,}")
        
        # Step 1: clean + basic quality filters
        print("\n[1/3] Cleaning & Quality Filtering...")
        cleaned_pairs = []
        stats = {
            "empty": 0,
            "no_alnum": 0,
            "too_short": 0,
            "too_long": 0,
            "bad_ratio": 0
        }
        
        for en, vi in tqdm(zip(en_raw, vi_raw), total=total_initial, desc="Processing"):
            en_clean, vi_clean, metrics = process_pair(en, vi, self.config)
            
            if metrics.is_valid:
                cleaned_pairs.append((en_clean, vi_clean))
            elif metrics.reason:
                if "Empty" in metrics.reason:
                    stats["empty"] += 1
                elif "alphanumeric" in metrics.reason:
                    stats["no_alnum"] += 1
                elif "short" in metrics.reason:
                    stats["too_short"] += 1
                elif "long" in metrics.reason:
                    stats["too_long"] += 1
                elif "ratio" in metrics.reason:
                    stats["bad_ratio"] += 1
        
        print(f"  ✓ Remaining: {len(cleaned_pairs):,}")
        print(f"  Removed: empty={stats['empty']}, no_alnum={stats['no_alnum']}, "
            f"too_short={stats['too_short']}, too_long={stats['too_long']}, "
            f"bad_ratio={stats['bad_ratio']}")
        
        # Step 2: deduplication
        print("\n[2/3] Deduplication...")
        seen = set()
        deduplicated_pairs = []
        
        for pair in cleaned_pairs:
            if pair not in seen:
                seen.add(pair)
                deduplicated_pairs.append(pair)
        
        removed_dedup = len(cleaned_pairs) - len(deduplicated_pairs)
        print(f"  ✓ Remaining: {len(deduplicated_pairs):,} (Removed: {removed_dedup:,})")
        # clear memory cleaned_pairs
        del cleaned_pairs
        
        # Step 3: Quality-based Filtering
        print("\n[3/3] Quality-based Filtering...")
        quality_based_pairs = []
        
        for en, vi in deduplicated_pairs:
            metrics = check_quality(
                en_text=en,
                vi_text=vi,
                min_words=self.config.min_words,
                max_words=self.config.max_words,
                min_ratio=self.config.min_length_ratio,
                max_ratio=self.config.max_length_ratio
            )
            if metrics.is_valid == True:
                quality_based_pairs.append((en, vi))
            
        removed_quality = len(deduplicated_pairs) - len(quality_based_pairs)
        print(f"  ✓ Remaining: {len(quality_based_pairs):,} (Removed: {removed_quality:,})")
        del deduplicated_pairs
        
        return quality_based_pairs
    
    def build_sft_dataset(
        self, 
        pairs: List[Tuple[str, str]],
    ) -> List[dict]:
        print("\nBuilding SFT dataset...")
        dataset = []
        
        for en_text, vi_text in tqdm(pairs, desc="Formatting"):
            sample_en2vi = create_sft_sample(
                source_text=en_text,
                target_text=vi_text,
                system_prompt=self.config.system_prompt,
                prompts=self.config.prompts_en_to_vi
            )
            
            sample_vi2en = create_sft_sample(
                source_text=vi_text,
                target_text=en_text,
                system_prompt=self.config.system_prompt,
                prompts=self.config.prompts_vi_to_en
            )
            
            dataset.append(sample_en2vi)
            dataset.append(sample_vi2en)
            
        set_seed()
        random.shuffle(dataset)

        print(f"  ✓ Total samples (bidirectional): {len(dataset):,}")
        return dataset
    
    def save(
        self,
        samples: List[dict],
        split: str
    ):
        os.makedirs(self.config.processed_dir, exist_ok=True)
        
        output_file = output_file = os.path.join(
            self.config.processed_dir,
            f"sft_{split}.{self.config.output_format}"
        )
        
        if self.config.output_format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        elif self.config.output_format == "parquet":
            df = pd.DataFrame(samples)
            df.to_parquet(output_file, index=False, compression='snappy')
            
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✅ Saved to: {output_file}")
        print(f"   Format: {self.config.output_format.upper()}")
        print(f"   Samples: {len(samples):,}")
        print(f"   Size: {file_size:.2f} MB")
        
    def build(
        self,
        split: Literal["train", "validation"]
    ): 
        print(f"\n{'='*50}")
        print(f"Building SFT Dataset - Split: {split.upper()}")
        print(f"{'='*50}\n")
        
        # Load
        en_raw, vi_raw = self.load_raw_data(split)
        
        # Process & Filter
        pairs = self.process_and_filter(en_raw, vi_raw)
        
        # Build SFT dataset
        samples = self.build_sft_dataset(pairs)
        
        # Save
        self.save(samples, split)
        
        return samples
    
if __name__ == "__main__":
    config = DataConfig()
    builder = SFTDatasetBuilder(config)
    
    # Build train and validation splits
    builder.build(split="train")
    builder.build(split="validation")