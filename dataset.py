# ============================================================
# dataset.py — KONE LLM Dataset Pipeline
# Loads kone_train.txt, tokenizes, creates batches
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tokenizer import KONETokenizer
from config import BLOCK_SIZE, BATCH_SIZE, TOKENIZER_PATH


class KONEDataset(Dataset):

    def __init__(self, split="train", val_ratio=0.1):
        self.block_size = BLOCK_SIZE

        # load tokenizer
        self.tokenizer = KONETokenizer()
        if os.path.exists(TOKENIZER_PATH):
            self.tokenizer.load(TOKENIZER_PATH)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_PATH}\n"
                "Run: python tokenizer.py first"
            )

        # load and tokenize text
        with open("data/kone_train.txt", "r", encoding="utf-8") as f:
            texts = f.readlines()
        texts = texts[:25000]

        print(f"Loaded {len(texts)} training texts")

        # encode all texts
        all_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text.strip())
            if ids:
                all_ids.extend(ids)
                all_ids.append(self.tokenizer.vocab.get("<EOS>", 2))

        print(f"Total tokens: {len(all_ids):,}")

        # split train/val
        tokens  = torch.tensor(all_ids, dtype=torch.long)
        n       = int(len(tokens) * (1 - val_ratio))

        if split == "train":
            self.data = tokens[:n]
        else:
            self.data = tokens[n:]

        print(f"{split} split: {len(self.data):,} tokens")

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk  = self.data[idx: idx + self.block_size + 1]
        x      = chunk[:-1]
        y      = chunk[1:]
        return x, y


def get_dataloaders():
    train_ds = KONEDataset(split="train")
    val_ds   = KONEDataset(split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    x, y = next(iter(train_loader))
    print(f"Batch x shape : {x.shape}")
    print(f"Batch y shape : {y.shape}")
