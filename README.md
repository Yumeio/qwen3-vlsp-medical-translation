# Qwen3 VLSP Medical Translation

This project focuses on fine-tuning **Qwen3 (1.7B)** models for **English-Vietnamese medical translation** using the VLSP dataset. It leverages efficient training techniques like SFT and GRPO with PEFT (LoRA, QLoRA, IA3).

[Tiếng Việt](README.vi.md)

## Features

- **Data Processing**: Automated cleaning, filtering, deduplication, and balancing.
- **Training Strategies**:
  - **SFT**: Supervised Fine-Tuning.
  - **GRPO**: Generative Reward Policy Optimization.
- **Efficient Training**: LoRA, QLoRA, IA3 adapters, Flash Attention 2, and BF16 precision.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place your raw data in `dataset/raw` and run:

```bash
python src/prepare_sft.py
```

### 3. Training

Start the training pipeline (SFT or GRPO):

```bash
python src/main.py
```

## Model Checkpoints

Download trained checkpoints here: [Google Drive Folder](https://drive.google.com/drive/folders/1EEF71obQJ__149HmzRtWSycmJbyWIzVr?usp=sharing)