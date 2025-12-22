from typing import List, Literal
from dataclasses import dataclass

@dataclass
class DataConfig:
    # Directories
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"
    
    # System prompt
    system_prompt: str = (
        "You are Qwen-Med, an AI learning support specialist."
        "Your task is to provide accurate translation and explanation of algorithmic terminology"
        "between English and Vietnamese. Always respond professionally, concisely, accurately, and clearly."
    )
    
    # Prompts
    prompts_en_to_vi: List[str] = None
    prompts_vi_to_en: List[str] = None
    
    # Quality filters
    min_words: int = 2
    max_words: int = 256
    min_length_ratio: float = 0.25
    max_length_ratio: float = 4.0
    
    # Output format
    output_format: Literal["jsonl", "parquet"] = "parquet"
    
    def __post_init__(self):
        if self.prompts_en_to_vi is None:
            self.prompts_en_to_vi = [
                "Dịch câu văn y học sau sang tiếng Việt:",
                "Chuyển ngữ nội dung này sang tiếng Việt:",
                "Hãy dịch thuật ngữ chuyên ngành này sang tiếng Việt:",
                "Phiên dịch đoạn văn y học sau sang tiếng Việt:",
                "Nghĩa tiếng Việt của đoạn văn này là gì:",
                "Dịch sang Tiếng Việt đoạn văn sau:",
                "Translate the following medical text to Vietnamese:",
                "Convert this content into Vietnamese:",
                "Please translate this medical term into Vietnamese:",
            ]
        
        if self.prompts_vi_to_en is None:
            self.prompts_vi_to_en = [
                "Dịch câu văn y học sau sang tiếng Anh:",
                "Chuyển ngữ nội dung này sang tiếng Anh:",
                "Hãy dịch thuật ngữ chuyên ngành này sang tiếng Anh:",
                "Phiên dịch đoạn văn y học sau sang tiếng Anh:",
                "Nghĩa tiếng Anh của đoạn văn này là gì:",
                "Dịch sang Tiếng Anh đoạn văn sau:",
                "Translate the following medical text to English:",
                "Convert this content into English:",
                "Please translate this medical term into English:",
            ]