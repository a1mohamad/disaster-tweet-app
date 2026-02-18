import json
import re
import torch

from json import JSONDecodeError
from app.utils.errors import ArtifactError

try:
    from langdetect import detect_langs, LangDetectException
    _LANGDETECT_AVAILABLE = True
except Exception:
    detect_langs = None
    LangDetectException = Exception
    _LANGDETECT_AVAILABLE = False

def load_vocabs(vocab_path):
    '''
    Load vocabulary mappings and metadata from JSON.
    '''
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
    
    # Extract mappings
    word2idx = data.get('stoi', {})
    # Convert JSON string keys ("0", "1") to integers (0, 1)
    idx2word = {int(k): v for k, v in data.get('itos', {}).items()}
    
    # Extract metadata saved during training
    vocab_size = data.get('vocab_size')
    
    # Fallback if vocab_size is missing for some reason
    if vocab_size is None:
        vocab_size = len(word2idx)
    
    return word2idx, idx2word, vocab_size

def tokenize_and_pad(text, word2idx, config):
    '''
    Returns the tensor AND the actual length (before padding).
    '''
    tokens = text.split()
    
    # Calculate length (capped at MAX_LENGTH)
    actual_length = min(len(tokens), config.MAX_LENGTH)
    # Ensure length is at least 1 (to avoid LSTM errors)
    actual_length = max(actual_length, 1)

    ids = [word2idx.get(token, config.UNK_IDX) for token in tokens]

    if len(ids) < config.MAX_LENGTH:
        ids += [config.PAD_IDX] * (config.MAX_LENGTH - len(ids))
    else:
        ids = ids[:config.MAX_LENGTH]

    input_ids = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)
    input_length = torch.tensor([actual_length], dtype=torch.long)
    
    return input_ids, input_length

def is_empty_text(text):
    return not text or not text.strip()

def detect_language(text):
    """
    Returns (lang, prob) or (None, None) if unavailable or undetected.
    """
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

def clean_text(text, config):
    '''
    Professional cleaning for Deep Learning:
    - Removes URLs and HTML tags
    - Standardizes whitespace
    - Keeps punctuation and stopwords for context
    '''
    if config.LOWERCASE:
        # Lowercase
        text = text.lower()
    elif config.UPPERCASE:
        text = text.upper()

    # URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)

    # HTML entities
    text = re.sub(r'&amp;', ' and ', text)

    # Mentions
    text = re.sub(r'@\w+', ' <USER> ', text)

    # Split URL-style joined words (general)
    text = re.sub(r'%20', ' ', text)


    # Remove broken unicode artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Normalize repeated punctuation
    text = re.sub(r'!{2,}', ' !! ', text)
    text = re.sub(r'\?{2,}', ' ?? ', text)

    if config.STRIP_MULTIPLE_WHITESPACE:    
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

    # Normalize numbers
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)

    # Reduce elongated words: cooool -> cool
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    text = re.sub(r'#\s+(\w+)', r'#\1', text)


    return text

def build_final_text(text, keyword=None, config=None):
    """
    Build model-ready text for inference.
    """
    cleaned_text = clean_text(text, config)

    if config.USE_KEYWORD and keyword:
        cleaned_keyword = clean_text(keyword, config)
        if cleaned_keyword:
            return f"{cleaned_keyword} : {cleaned_text}"

    return cleaned_text
