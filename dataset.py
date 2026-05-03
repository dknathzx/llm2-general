# ============================================================
# dataset.py — General LLM Dataset
# Sources: OpenWebText + Stack Overflow (free on Kaggle)
# No confidential data — fully public datasets
# ============================================================
import sys
sys.path.append('/kaggle/working/llm2-general')
import json
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    BLOCK_SIZE, BATCH_SIZE, TOKENIZER_PATH,
    MAX_SAMPLES, DEVICE
)
from tokenizer import GeneralTokenizer

def load_tokenizer():
    tok = GeneralTokenizer()
    tok.load(TOKENIZER_PATH)
    return tok

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size=BLOCK_SIZE):
        self.data       = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        print(f"  Dataset tokens : {len(self.data):,}")
        print(f"  Block size     : {block_size}")
        print(f"  Total batches  : {len(self):,}")

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x     = chunk[:-1]
        y     = chunk[1:]
        return x, y

def load_data(tok):
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"  Loading Datasets")
    print(f"{'='*60}")

    all_texts = []

    print("\n[1/2] Loading OpenWebText...")
    try:
        wiki = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
        wiki_sample = min(MAX_SAMPLES // 2, len(wiki))
        for i in range(wiki_sample):
            text = wiki[i]["text"]
            if text and len(text) > 100:
                all_texts.append(text[:1000])
            if (i + 1) % 50000 == 0:
                print(f"  OpenWebText: loaded {i+1:,} articles")
        print(f"  OpenWebText total: {wiki_sample:,} articles ✅")
    except Exception as e:
        print(f"  OpenWebText failed: {e}")

    print("\n[2/2] Loading Stack Overflow...")
    try:
        so = load_dataset("koutch/stackoverflow_python", split="train", trust_remote_code=True)
        so_sample = min(MAX_SAMPLES // 2, len(so))
        for i in range(so_sample):
            row  = so[i]
            text = str(row.get("question_body", "")) + " " + str(row.get("answer_body", ""))
            if text.strip() and len(text) > 50:
                all_texts.append(text[:1000])
            if (i + 1) % 50000 == 0:
                print(f"  StackOverflow: loaded {i+1:,} posts")
        print(f"  StackOverflow total: {so_sample:,} posts ✅")
    except Exception as e:
        print(f"  StackOverflow failed: {e}")

    print(f"\nTotal texts loaded : {len(all_texts):,}")

    DATASET_CKPT = "/kaggle/working/dataset_checkpoint.json"
    start_idx    = 0
    all_ids      = []

    if os.path.exists(DATASET_CKPT):
        with open(DATASET_CKPT) as f:
            ckpt = json.load(f)
        all_ids   = ckpt["ids"]
        start_idx = ckpt["done"]
        print(f"  ✅ Resumed from text {start_idx:,}")

    print(f"\nTokenizing texts {start_idx:,} → {len(all_texts):,}...")
    print(f"{'='*60}")

    from tokenizer import push_to_kaggle

    start_time = time.time()

    for i, text in enumerate(all_texts[start_idx:], start=start_idx):
        t0       = time.time()
        ids      = tok.encode(text)
        t1       = time.time()
        all_ids.extend(ids)

        elapsed  = time.time() - start_time
        speed    = (i - start_idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(all_texts) - i - 1) / speed if speed > 0 else 0
        pct      = (i + 1) / len(all_texts) * 100

        print(
            f"  [{pct:5.1f}%] "
            f"text {i+1:,}/{len(all_texts):,}  "
            f"tokens: +{len(ids):3d} = {len(all_ids):,}  "
            f"speed: {speed:.1f} texts/s  "
            f"ETA: {remaining/60:.1f} min  "
            f"took: {(t1-t0)*1000:.0f}ms"
        )

        if (i + 1) % 5000 == 0:
            with open(DATASET_CKPT, "w") as f:
                json.dump({"ids": all_ids, "done": i + 1}, f)
            print(f"\n  💾 checkpoint saved at text {i+1:,}")
            push_to_kaggle(f"dataset checkpoint text {i+1}")
            print(f"{'='*60}\n")

    print(f"\n✅ Tokenization complete!")
    print(f"Total tokens : {len(all_ids):,}")
    return all_ids

def get_dataloaders(tok):
    all_ids = load_data(tok)

    split    = int(0.9 * len(all_ids))
    train_ds = TextDataset(all_ids[:split])
    val_ds   = TextDataset(all_ids[split:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n  Train batches : {len(train_dl):,}")
    print(f"  Val batches   : {len(val_dl):,}")
    return train_dl, val_dl
