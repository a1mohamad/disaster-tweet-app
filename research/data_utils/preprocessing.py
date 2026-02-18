import pandas as pd
import re

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

def final_text_with_keyword(row, config):
    keyword = str(row["keyword"]).strip() if pd.notna(row["keyword"]) else ""
    keyword = clean_text(keyword, config)
    cleaned_text = clean_text(row["text"], config)

    if config.USE_KEYWORD and keyword != "":
        return f"{keyword} : {cleaned_text}"

    return cleaned_text


def preprocess_df(df, config):
    if config.DROP_LOCATION and 'location' in df.columns:
        df = df.drop(columns=['location'])

    df['final_text'] = df.apply(final_text_with_keyword, axis=1)
    df = df.drop(columns=['keyword', 'text'])

    return df
