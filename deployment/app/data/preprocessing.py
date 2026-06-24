import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.errors import ArtifactError

try:
    from langdetect import detect_langs, LangDetectException
    _LANGDETECT_AVAILABLE = True
except Exception:
    detect_langs = None
    LangDetectException = Exception
    _LANGDETECT_AVAILABLE = False


def load_vocabs(vocab_path: str | Path) -> tuple[dict[str, int], dict[int, str], int]:
    """Load vocabulary mappings and metadata from the exported JSON artifact."""
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ArtifactError(
            "VOCAB_NOT_FOUND",
            "Vocabulary file not found.",
            {"path": str(vocab_path)},
        ) from exc
    except JSONDecodeError as exc:
        raise ArtifactError(
            "VOCAB_INVALID",
            "Vocabulary file is not valid JSON.",
            {"path": str(vocab_path)},
        ) from exc
    word2idx = data.get('stoi', {})
    idx2word = {int(k): v for k, v in data.get('itos', {}).items()}
    vocab_size = data.get('vocab_size')
    if vocab_size is None:
        vocab_size = len(word2idx)
    
    return word2idx, idx2word, vocab_size


def tokenize_and_pad(
    text: str, word2idx: dict[str, int], config: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cleaned text to model input ids and the pre-padding sequence length."""
    tokens = text.split()
    
    actual_length = min(len(tokens), config.MAX_LENGTH)
    # ONNX and PyTorch LSTM paths both require a positive sequence length.
    actual_length = max(actual_length, 1)

    ids = [word2idx.get(token, config.UNK_IDX) for token in tokens]

    if len(ids) < config.MAX_LENGTH:
        ids += [config.PAD_IDX] * (config.MAX_LENGTH - len(ids))
    else:
        ids = ids[:config.MAX_LENGTH]

    input_ids = np.asarray([ids], dtype=np.int64)
    input_length = np.asarray([actual_length], dtype=np.int64)
    
    return input_ids, input_length


def is_empty_text(text: str | None) -> bool:
    """Return True when text is missing or only whitespace."""
    return not text or not text.strip()


def detect_language(text: str) -> tuple[str | None, float | None]:
    """Detect the most likely language, or return empty values when unavailable."""
    if not _LANGDETECT_AVAILABLE:
        return None, None
    sample = text.strip()
    if not sample:
        return None, None
    try:
        langs = detect_langs(sample)
        if not langs:
            return None, None
        top = langs[0]
        return top.lang, float(top.prob)
    except LangDetectException:
        return None, None


def clean_text(text: str, config: Any) -> str:
    """Normalize tweet text using the same rules as the training pipeline."""
    if config.LOWERCASE:
        text = text.lower()
    elif config.UPPERCASE:
        text = text.upper()

    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'%20', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Preserve repeated punctuation as stable tokens because urgency can matter.
    text = re.sub(r'!{2,}', ' !! ', text)
    text = re.sub(r'\?{2,}', ' ?? ', text)

    if config.STRIP_MULTIPLE_WHITESPACE:
        text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'#\s+(\w+)', r'#\1', text)

    return text


def build_final_text(text: str, keyword: str | None = None, config: Any = None) -> str:
    """Build model-ready inference text from the raw tweet and optional keyword."""
    cleaned_text = clean_text(text, config)

    if config.USE_KEYWORD and keyword:
        cleaned_keyword = clean_text(keyword, config)
        if cleaned_keyword:
            return f"{cleaned_keyword} : {cleaned_text}"

    return cleaned_text
