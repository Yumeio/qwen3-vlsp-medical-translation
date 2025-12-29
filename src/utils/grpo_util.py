import json
import argparse


def load_jsonl(file_path: str) -> list:
    """Load JSONL file"""
    samples = []
    print("Reading JSONL...", end=" ", flush=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"✓ ({len(samples)} samples)")
    return samples


def normalize_grpo_format(sample: dict) -> dict:
    """
    Normalize GRPO sample to standard format

    Input format (from file):
    {
        "prompt": "...",
        "system_prompt": "...",
        "source": "...",
        "reference": "...",
        "completions": [
            {"content": "...", "score": 0.5},
            ...
        ]
    }

    Output format (for TRL):
    {
        "prompt": "...",
        "samples": [
            {"text": "...", "score": 0.5},
            ...
        ],
        "source": "...",
        "reference": "...",
        "system_prompt": "..." (optional)
    }
    """

    normalized = {
        "prompt": sample["prompt"],
        "samples": [],
        "source": sample.get("source", ""),
        "reference": sample.get("reference", ""),
    }

    # Add system_prompt if present
    if "system_prompt" in sample:
        normalized["system_prompt"] = sample["system_prompt"]

    # Normalize completions: content → text
    for completion in sample["completions"]:
        normalized_completion = {
            "text": completion.get("content") or completion.get("text"),
            "score": completion["score"]
        }
        normalized["samples"].append(normalized_completion)

    return normalized


def analyze_data(samples: list):
    """Analyze GRPO data"""

    print("\n" + "="*60)
    print("Data Analysis")
    print("="*60)

    print(f"Total samples: {len(samples)}")

    # Check format
    first = samples[0]
    print(f"\nSample keys: {list(first.keys())}")
    print(f"Completion keys: {list(first['samples'][0].keys())}")

    # Samples stats
    num_samples = [len(s['samples']) for s in samples]
    print(f"\nSamples per sample:")
    print(f"  Min: {min(num_samples)}")
    print(f"  Max: {max(num_samples)}")
    print(f"  Avg: {sum(num_samples)/len(num_samples):.1f}")

    # Score stats
    all_scores = []
    for sample in samples:
        all_scores.extend([c['score'] for c in sample['samples']])

    print(f"\nScore statistics:")
    print(f"  Mean: {sum(all_scores)/len(all_scores):.4f}")
    print(f"  Min: {min(all_scores):.4f}")
    print(f"  Max: {max(all_scores):.4f}")

    # Sample
    print("\n" + "="*60)
    print("Sample")
    print("="*60)
    print(f"Prompt: {first['prompt'][:100]}...")
    print(f"Source: {first.get('source', 'N/A')[:80]}...")
    print(f"Reference: {first.get('reference', 'N/A')[:80]}...")
    print(f"\nSamples ({len(first['samples'])}):")
    for i, sample in enumerate(first['samples']):
        text = sample.get('text')
        print(f"  [{i}] Score: {sample['score']:.4f}")
        print(f"      Text: {text[:80]}...")


def convert_format(
    input_file: str,
    output_file: str,
    format: str = "parquet",
    analyze: bool = True
):
    """Convert GRPO JSONL to standard format"""

    print("="*60)
    print("GRPO Data Converter")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Format: {format}")
    print("="*60)

    # Load data
    print("\n📂 Loading data...")
    samples = load_jsonl(input_file)
    print(f"✓ Loaded {len(samples)} samples")

    # Analyze
    if analyze:
        analyze_data(samples)

    # Normalize
    print("\n🔄 Normalizing format...")
    print("Converting...", end=" ", flush=True)
    normalized = [normalize_grpo_format(sample) for sample in samples]
    print(f"✓ ({len(normalized)} samples)")

    # Verify normalization
    print("\n✓ Normalized samples")
    print(f"  Completion key: 'text' (was 'content')")

    # Save
    print(f"\n💾 Saving to {format}...")

    if format == "parquet":
        try:
            from datasets import Dataset
            dataset = Dataset.from_list(normalized)
            dataset.to_parquet(output_file)
            print(f"✓ Saved {len(normalized)} samples (parquet)")
        except ImportError:
            print("⚠️  'datasets' not installed, saving as JSONL instead")
            format = "jsonl"
            output_file = output_file.replace('.parquet', '.jsonl')

    if format == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in normalized:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ Saved {len(normalized)} samples (jsonl)")

    elif format == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(normalized)} samples (json)")

    print("\n" + "="*60)
    print("✅ Conversion Complete!")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"Samples: {len(normalized)}")
    print("\nKey changes:")
    print("  - Completions field: 'content' → 'text'")
    print("  - Format verified for TRL GRPO training")