import re
from typing import Any

import pandas as pd


def clean_text(text: str, config: Any) -> str:
    """Normalize tweet text while preserving useful disaster-reporting context."""
    if config.LOWERCASE:
        text = text.lower()
    elif config.UPPERCASE:
        text = text.upper()

    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'%20', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Keep the fact that punctuation was repeated, but cap it to stable tokens.
    text = re.sub(r'!{2,}', ' !! ', text)
    text = re.sub(r'\?{2,}', ' ?? ', text)

    if config.STRIP_MULTIPLE_WHITESPACE:
        text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'#\s+(\w+)', r'#\1', text)

    return text


def final_text_with_keyword(row: pd.Series, config: Any) -> str:
    """Combine keyword and tweet text into the model's final training input."""
    keyword = str(row["keyword"]).strip() if pd.notna(row["keyword"]) else ""
    keyword = clean_text(keyword, config)
    cleaned_text = clean_text(row["text"], config)

    if config.USE_KEYWORD and keyword != "":
        return f"{keyword} : {cleaned_text}"

    return cleaned_text


def preprocess_df(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Create the final text column and remove raw columns no longer needed."""
    if config.DROP_LOCATION and 'location' in df.columns:
        df = df.drop(columns=['location'])

    df['final_text'] = df.apply(final_text_with_keyword, axis=1)
    df = df.drop(columns=['keyword', 'text'])

    return df
