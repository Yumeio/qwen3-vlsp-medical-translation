# Dịch Y tế VLSP với Qwen3

Dự án này tập trung vào việc tinh chỉnh (fine-tune) mô hình **Qwen3 (1.7B)** cho tác vụ **dịch thuật y tế Anh-Việt** sử dụng dữ liệu VLSP. Dự án áp dụng các kỹ thuật huấn luyện hiệu quả như SFT và GRPO cùng với PEFT (LoRA, QLoRA, IA3).

[English](README.md)

## Tính năng

- **Xử lý Dữ liệu**: Tự động làm sạch, lọc, khử trùng lặp và cân bằng dữ liệu.
- **Chiến lược Huấn luyện**:
  - **SFT**: Supervised Fine-Tuning.
  - **GRPO**: Generative Reward Policy Optimization.
- **Huấn luyện Hiệu quả**: Sử dụng adapter LoRA, QLoRA, IA3, Flash Attention 2 và độ chính xác BF16.

## Bắt đầu nhanh

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị Dữ liệu

Đặt dữ liệu thô vào thư mục `dataset/raw` và chạy lệnh:

```bash
python src/prepare_sft.py
```

### 3. Huấn luyện

Bắt đầu quy trình huấn luyện (SFT hoặc GRPO):

```bash
python src/main.py
```

## Checkpoint Mô hình

Tải xuống các checkpoint đã huấn luyện tại đây: [Thư mục Google Drive](https://drive.google.com/drive/folders/1EEF71obQJ__149HmzRtWSycmJbyWIzVr?usp=sharing)
