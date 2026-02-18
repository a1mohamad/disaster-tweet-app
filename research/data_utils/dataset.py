from collections import Counter
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

def build_vocab(texts, config):
    counter = Counter()
    vocab_size = config.VOCAB_SIZE
    PAD_TOKEN = config.PAD_TOKEN
    PAD_IDX = config.PAD_IDX
    UNK_TOKEN = config.UNK_TOKEN
    UNK_IDX = config.UNK_IDX
    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    most_common = counter.most_common(vocab_size - 2)

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX
    }

    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx

    return vocab

def save_vocabs(stoi: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    itos = {str(v): k for k, v in stoi.items()}
    payload = {
        "vocab_size": len(stoi),
        "special_tokens": {
            "pad": "<PAD>",
            "unk": "<UNK>"
        },
        "stoi": stoi,
        "itos": itos
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Vocab and inverse Vocab saved as a JSON file")

def encode_text(text, vocab, maxlen, config):
    PAD_TOKEN = config.PAD_TOKEN
    UNK_TOKEN = config.UNK_TOKEN
    tokens = text.split()
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    if len(ids) > maxlen:
        ids = ids[:maxlen]

    if len(ids) < maxlen:
        ids += [vocab[PAD_TOKEN]] * (maxlen - len(ids))

    return ids

def decode(ids, idx2word, config):
    return " ".join([idx2word.get(i, "<UNK>") for i in ids if i != config.PAD_IDX])

class DisasterDataset(Dataset):
    def __init__(self, df, vocab, config):
        self.texts = df["final_text"].values
        self.targets = df["target"].values
        self.vocab = vocab
        self.maxlen = config.MAX_LENGTH
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded_text = encode_text(
            self.texts[idx],
            self.vocab,
            self.maxlen,
            self.config
        )
        length = min(len(self.texts[idx].split()), self.maxlen)
        return {
            "input_ids": torch.LongTensor(encoded_text),
            "labels": torch.tensor(self.targets[idx], dtype=torch.float),
            "length": torch.tensor(length, dtype=torch.long)
        }