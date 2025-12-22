import re
import unicodedata
from typing import Tuple, Optional
from dataclasses import dataclass
@dataclass
class QualityMetrics:
    en_words: int
    vi_words: int
    length_ratio: float
    is_valid: bool
    reason: Optional[str] = None

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove unwanted characters
    text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[\s\t\n\r\xa0]+', ' ', text)
    
    # Remove special chars at start/end
    text = re.sub(r'^[\-\*\•\>]\s+', '', text)
    text = re.sub(r'\s+[\-\*\•\>]$', '', text)
    
    # Normalize punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])(?=[a-zA-Z0-9])', r'\1 ', text)
    
    return text.strip()

def check_quality(
    en_text: str, 
    vi_text: str,
    min_words: int = 2,
    max_words: int = 256,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0
) -> QualityMetrics:
    en_words_list = en_text.split()
    vi_words_list = vi_text.split()
    
    len_en = len(en_words_list)
    len_vi = len(vi_words_list)
    
    # Check empty lines
    if not en_text or not vi_text:
        return QualityMetrics(
            en_words=len_en,
            vi_words=len_vi,
            length_ratio=0.0,
            is_valid=False,
            reason="Empty text"
        )
        
    # Alphanumeric check
    if not re.search(r'[a-zA-Z0-9]', en_text) or not re.search(r'[a-zA-Z0-9]', vi_text):
        return QualityMetrics(
            en_words=len_en,
            vi_words=len_vi,
            length_ratio=0.0,
            is_valid=False,
            reason="No alphanumeric characters"
        )
        
    # Min/max length
    if len_en < min_words or len_vi < min_words:
        return QualityMetrics(
            en_words=len_en,
            vi_words=len_vi,
            length_ratio=len_en / len_vi if len_vi > 0 else 0,
            is_valid=False,
            reason=f"Too short (min {min_words} words)"
        )
    if len_en > max_words or len_vi > max_words:
        return QualityMetrics(
            en_words=len_en,
            vi_words=len_vi,
            length_ratio=len_en / len_vi if len_vi > 0 else 0,
            is_valid=False,
            reason=f"Too long (max {max_words} words)"
        )
        
    # Length ratio check
    ratio = len_en / len_vi if len_vi > 0 else 0
    if ratio < min_ratio or ratio > max_ratio:
        return QualityMetrics(
            en_words=len_en,
            vi_words=len_vi,
            length_ratio=ratio,
            is_valid=False,
            reason=f"Bad length ratio: {ratio:.2f}"
        )
    
    return QualityMetrics(
        en_words=len_en,
        vi_words=len_vi,
        length_ratio=ratio,
        is_valid=True
    )
    
def process_pair(
    en_raw: str,
    vi_raw: str,
    config
) -> Tuple[Optional[str], Optional[str], QualityMetrics]:
    """
    Process a single pair: clean + quality check
    
    Returns:
        (en_clean, vi_clean, metrics) or (None, None, metrics) if invalid
    """
    en_clean = clean_text(en_raw)
    vi_clean = clean_text(vi_raw)
    
    metrics = check_quality(
        en_clean, vi_clean,
        min_words=config.min_words,
        max_words=config.max_words,
        min_ratio=config.min_length_ratio,
        max_ratio=config.max_length_ratio
    )
    
    if metrics.is_valid:
        return en_clean, vi_clean, metrics
    else:
        return None, None, metrics