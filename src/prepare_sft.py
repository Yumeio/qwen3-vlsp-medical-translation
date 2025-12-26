import os
import json
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

from arguments import DataConfig
from utils.text_processing import process_pair, check_quality
from utils.formatting import create_sft_sample
from utils.default import set_seed

class SFTDatasetPreparer:
    def __init__(self, config: DataConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        set_seed(seed=seed)

    def load_raw_data(self):
        print("📂 LOADING RAW DATA")

        # Load train data
        train_en_file = os.path.join(self.config.raw_dir, "train.en.txt")
        train_vi_file = os.path.join(self.config.raw_dir, "train.vi.txt")
        print(f"\n[1/2] Loading train data:")
        print(f"  EN: {train_en_file}")
        print(f"  VI: {train_vi_file}")
        train_ds = load_dataset(
            "text",
            data_files={
                "en": train_en_file, 
                "vi": train_vi_file
            },
            streaming=False,
        )
        train_en = train_ds["en"]["text"]
        train_vi = train_ds["vi"]["text"]
        print(f"  ✓ Loaded: {len(train_en):,} pairs")

        # Create bidirectional train samples
        trainval_samples = []
        for en, vi in tqdm(zip(train_en, train_vi), total=len(train_en)):
            trainval_samples.append({
                "source": en,
                "target": vi,
                "direction": "en2vi"
            })
            trainval_samples.append({
                "source": vi,
                "target": en,
                "direction": "vi2en"
            })
        print(f"  ✓ Created: {len(trainval_samples):,} samples (bidirectional)")

        # Load test data
        test_en_file = os.path.join(self.config.raw_dir, "public_test.en.txt")
        test_vi_file = os.path.join(self.config.raw_dir, "public_test.vi.txt")
        print(f"\n[2/2] Loading test data:")
        print(f"  EN: {test_en_file}")
        print(f"  VI: {test_vi_file}")
        test_ds = load_dataset(
            "text",
            data_files={"en": test_en_file, "vi": test_vi_file},
            streaming=False
        )
        test_en = test_ds["en"]["text"]
        test_vi = test_ds["vi"]["text"]
        print(f"  ✓ Loaded: {len(test_en):,} pairs")
        
        # Create bidirectional test samples
        test_samples = []
        for en, vi in zip(test_en, test_vi):
            test_samples.append({
                "source": en,
                "target": vi,
                "direction": "en2vi"
            })
            test_samples.append({
                "source": vi,
                "target": en,
                "direction": "vi2en"
            })
        
        print(f"  ✓ Created: {len(test_samples):,} samples (bidirectional)")
        
        return trainval_samples, test_samples

    def process_and_filter(
        self,
        samples: List[dict],
        split: str,
    ):
        print(f"\n📂 PROCESSING AND FILTERING {split} DATA")
        initial_count = len(samples)
        print(f"\n  Initial count: {initial_count:,}")

        # Count by direction
        direction_counts = {}
        for sample in samples:
            direction = sample["direction"]
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        print(f"  EN→VI: {direction_counts.get('en2vi', 0):,}")
        print(f"  VI→EN: {direction_counts.get('vi2en', 0):,}")

        print(f"\n[1/3] Cleaning & Quality Filtering...")

        cleaned_samples = []
        stats = {
            "empty": 0,
            "no_alnum": 0,
            "too_short": 0,
            "too_long": 0,
            "bad_ratio": 0
        }

        for sample in tqdm(samples, total=initial_count, desc="Cleaning & Quality Filtering"):
            source, target, direction = sample["source"], sample["target"], sample["direction"]
            if direction == "en2vi":
                en_text, vi_text = source, target
            else:
                vi_text, en_text = source, target

            en_clean, vi_clean, metrics = process_pair(en_text, vi_text, self.config)
            
            if metrics.is_valid:
                if direction == "en2vi":
                    cleaned_samples.append({
                        "source": en_clean,
                        "target": vi_clean,
                        "direction": direction
                    })
                else:  # vi2en
                    cleaned_samples.append({
                        "source": vi_clean,
                        "target": en_clean,
                        "direction": direction
                    })
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

        removed_step1 = initial_count - len(cleaned_samples)
        print(f"  ✓ Remaining: {len(cleaned_samples):,} ({removed_step1:,} removed)")
        print(f"    - Empty: {stats['empty']:,}")
        print(f"    - No alphanumeric: {stats['no_alnum']:,}")
        print(f"    - Too short: {stats['too_short']:,}")
        print(f"    - Too long: {stats['too_long']:,}")
        print(f"    - Bad ratio: {stats['bad_ratio']:,}")

        print(f"\n[2/3] Deduplication...")
        seen = set()
        deduplicated_samples = []
        for sample in tqdm(cleaned_samples, total=len(cleaned_samples), desc="Deduplication"):
            key = (sample["source"], sample["target"], sample["direction"])
            
            if key not in seen:
                seen.add(key)
                deduplicated_samples.append(sample)

        removed_step2 = len(cleaned_samples) - len(deduplicated_samples)
        print(f"  ✓ Remaining: {len(deduplicated_samples):,} ({removed_step2:,} removed)")

        del cleaned_samples
        del seen

        print(f"\n[3/3] Check Quality...")
        final_samples = []
        for sample in tqdm(deduplicated_samples, total=len(deduplicated_samples), desc="Quality Check"):
            source, target, direction = sample["source"], sample["target"], sample["direction"]
            
            if direction == "en2vi":
                en_text, vi_text = source, target
            else:  # vi2en
                vi_text, en_text = source, target

            metrics = check_quality(
                en_text=en_text,
                vi_text=vi_text,
                min_words=self.config.min_words,
                max_words=self.config.max_words,
                min_ratio=self.config.min_length_ratio,
                max_ratio=self.config.max_length_ratio
            )
            
            if metrics.is_valid:
                final_samples.append(sample)

        removed_step3 = len(deduplicated_samples) - len(final_samples)
        print(f"  ✓ Remaining: {len(final_samples):,} ({removed_step3:,} removed)")

        del deduplicated_samples

        total_removed = initial_count - len(final_samples)
        retention_rate = (len(final_samples) / initial_count * 100) if initial_count > 0 else 0

        final_direction_counts = {}
        for sample in final_samples:
            direction = sample["direction"]
            final_direction_counts[direction] = final_direction_counts.get(direction, 0) + 1
        
        print(f"\n📊 Summary for {split}:")
        print(f"  Initial:   {initial_count:,}")
        print(f"  Final:     {len(final_samples):,}")
        print(f"  Removed:   {total_removed:,}")
        print(f"  Retention: {retention_rate:.1f}%")
        print(f"\n  By direction:")
        print(f"    EN→VI: {final_direction_counts.get('en2vi', 0):,}")
        print(f"    VI→EN: {final_direction_counts.get('vi2en', 0):,}")
        
        return final_samples
    
    def make_balance_and_limit_samples(
        self,
        samples: List[dict],
        max_samples: Optional[int],
        split_name: str,
    ):
        if max_samples is None:
            print(f"\n✓ No max_samples limit for {split_name}")
            random.shuffle(samples)
            return samples
        
        print(f"⚖️  BALANCING & LIMITING - {split_name.upper()}")
        en2vi_samples = [s for s in samples if s["direction"] == "en2vi"]
        vi2en_samples = [s for s in samples if s["direction"] == "vi2en"]
        
        print(f"\nBefore balancing:")
        print(f"  EN→VI: {len(en2vi_samples):,}")
        print(f"  VI→EN: {len(vi2en_samples):,}")
        print(f"  Total: {len(samples):,}")

        random.shuffle(en2vi_samples)
        random.shuffle(vi2en_samples)

        samples_per_direction = max_samples // 2

        print(f"\nTarget: {max_samples:,} total samples")
        print(f"  → {samples_per_direction:,} per direction")

        en2vi_limited = en2vi_samples[:samples_per_direction]
        vi2en_limited = vi2en_samples[:samples_per_direction]

        balanced_samples = en2vi_limited + vi2en_limited
        random.shuffle(balanced_samples)

        print(f"\nAfter balancing:")
        print(f"  EN→VI: {len(en2vi_limited):,}")
        print(f"  VI→EN: {len(vi2en_limited):,}")
        print(f"  Total: {len(balanced_samples):,}")
        
        return balanced_samples

    def split_train_val(
        self, 
        samples: List[dict],
        val_ratio: float = 0.1,
    ): 
        print(f"✂️  SPLITTING TRAIN/VAL")
        en2vi_samples = [s for s in samples if s["direction"] == "en2vi"]
        vi2en_samples = [s for s in samples if s["direction"] == "vi2en"]
        
        print(f"\nInput samples:")
        print(f"  EN→VI: {len(en2vi_samples):,}")
        print(f"  VI→EN: {len(vi2en_samples):,}")

        en2vi_train, en2vi_val = train_test_split(
            en2vi_samples,
            test_size=val_ratio,
            random_state=self.seed,
            shuffle=True
        )
        vi2en_train, vi2en_val = train_test_split(
            vi2en_samples,
            test_size=val_ratio,
            random_state=self.seed,
            shuffle=True
        )

        train_samples = en2vi_train + vi2en_train
        val_samples = en2vi_val + vi2en_val

        print(f"\nTrain samples:")
        print(f"  EN→VI: {len(en2vi_train):,}")
        print(f"  VI→EN: {len(vi2en_train):,}")
        print(f"  Total: {len(train_samples):,}")

        print(f"\nVal samples:")
        print(f"  EN→VI: {len(en2vi_val):,}")
        print(f"  VI→EN: {len(vi2en_val):,}")
        print(f"  Total: {len(val_samples):,}")

        return train_samples, val_samples

    def convert_to_sft_chatml_format(
        self,
        samples: List[dict],
        split_name: str
    ):
        print(f"🔨 CONVERTING TO SFT FORMAT - {split_name.upper()}")
        sft_samples = []
        
        for sample in tqdm(samples, desc="  Creating SFT samples"):
            source = sample["source"]
            target = sample["target"]
            direction = sample["direction"]

            if direction == "en2vi":
                prompts = self.config.prompts_en_to_vi
            else:  # vi2en
                prompts = self.config.prompts_vi_to_en

            sft_sample = create_sft_sample(
                source_text=source,
                target_text=target,
                system_prompt=self.config.system_prompt,
                prompts=prompts
            )
            
            sft_samples.append(sft_sample)
            
        print(f"  ✓ Created {len(sft_samples):,} clean SFT samples")
        return sft_samples
    
    def save_dataset(
        self, 
        samples: List[dict],
        split: str
    ):
        os.makedirs(self.config.processed_dir, exist_ok=True)
        output_file = os.path.join(
            self.config.processed_dir,
            f"sft_{split}.{self.config.output_format}"
        )
        
        print(f"\n💾 Saving {split} dataset...")
        if len(samples) > 0:
            sample_keys = set(samples[0].keys())
            expected_keys = {"messages"}

            # check if sample_keys is different from expected_keys
            if sample_keys != expected_keys:
                raise ValueError(f"Sample keys {sample_keys} do not match expected keys {expected_keys}")
            else:
                print(f"✓ Sample keys match expected keys")


        if self.config.output_format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        elif self.config.output_format == "parquet":
            df = pd.DataFrame(samples)
            df.to_parquet(output_file, index=False, compression='snappy')
        
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ Saved: {output_file}")
        print(f"    - Format: {self.config.output_format.upper()}")
        print(f"    - Samples: {len(samples):,}")
        print(f"    - Size: {file_size:.2f} MB")
        print(f"    - Structure: {{'messages': [...]}}")

    def build(self):
        print("\nBuilding dataset...")
        
        # Load raw data
        trainval_samples, test_samples = self.load_raw_data()

        # Process and filter
        trainval_processed = self.process_and_filter(trainval_samples, "trainval")
        test_processed = self.process_and_filter(test_samples, "test")

        # Balance and limited 
        trainval_balanced = self.make_balance_and_limit_samples(
            trainval_processed,
            self.config.max_trainval_samples,
            "trainval"
        )
        test_balanced = self.make_balance_and_limit_samples(
            test_processed,
            self.config.max_eval_samples,
            "test"
        )
        
        # Split train/val
        train_samples, val_samples = self.split_train_val(
            trainval_balanced,
            val_ratio=self.config.val_ratio
        )

        # Convert to SFT chatml
        train_sft = self.convert_to_sft_chatml_format(train_samples, "train")
        val_sft = self.convert_to_sft_chatml_format(val_samples, "validation") 
        test_sft = self.convert_to_sft_chatml_format(test_balanced, "test") 
        
        # Save to disk
        self.save_dataset(train_sft, "train")
        self.save_dataset(val_sft, "validation")
        self.save_dataset(test_sft, "test")
        
        # Summary
        print("✅ PIPELINE COMPLETE")
        print(f"\n📁 Output directory: {self.config.processed_dir}")
        print(f"\n📊 Dataset Statistics:")
        print(f"  {'Split':<12} {'Samples':<12} {'File'}")
        print(f"  {'-'*60}")
        print(f"  {'Train':<12} {len(train_sft):>10,}   sft_train.{self.config.output_format}")
        print(f"  {'Validation':<12} {len(val_sft):>10,}   sft_validation.{self.config.output_format}")
        print(f"  {'Test':<12} {len(test_sft):>10,}   sft_test.{self.config.output_format}")
        print("="*70 + "\n")
        
if __name__ == "__main__":
    config = DataConfig()
    
    sft_prepare = SFTDatasetPreparer(config=config, seed=4663)
    sft_prepare.build()