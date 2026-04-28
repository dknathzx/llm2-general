# ============================================================
# tokenizer.py — General BPE Tokenizer
# Trains on Wikipedia + StackOverflow data
# Vocab size: 50,000
# ============================================================

import sys
sys.path.append('/kaggle/working/llm2-general')

from journey_log import log

import shutil
import json
import os
import re
import subprocess
import time
from collections import Counter
from config import VOCAB_SIZE, TOKENIZER_PATH

# ── Special tokens ────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# ── Kaggle Push Setup ─────────────────────────────────────
KAGGLE_DATASET     = "dwarakanathk/llm2-checkpoints-placeholder-file"
KAGGLE_DATASET_DIR = "/kaggle/working/llm2-checkpoints"
AUTO_PUSH          = True

def setup_kaggle_credentials():
    os.environ["KAGGLE_USERNAME"] = "dwarakanathk"
    os.environ["KAGGLE_KEY"]      = "KGAT_97ff8b4a8d918070c5209ec1e5c84858"
    # also write kaggle.json so CLI works
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({
            "username": "dwarakanathk",
            "key": "KGAT_97ff8b4a8d918070c5209ec1e5c84858"
        }, f)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)


def push_to_kaggle(label="update"):
    """Push ALL important files to Kaggle dataset permanently"""
    if not AUTO_PUSH:
        return
    try:
        setup_kaggle_credentials()
        print(f"\n  📤 Pushing to Kaggle dataset... [{label}]")

        os.makedirs(KAGGLE_DATASET_DIR, exist_ok=True)

        files_to_save = [
            "/kaggle/working/GENERAL_tokenizer.json",
            "/kaggle/working/tokenizer_checkpoint.json",
            "/kaggle/working/journey_log.json",
            "/kaggle/working/journey_backup.json",
        ]

        # Also grab any model checkpoints
        ckpt_dir = "/kaggle/working/checkpoints"
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                files_to_save.append(os.path.join(ckpt_dir, f))

        copied = []
        for fpath in files_to_save:
            if os.path.exists(fpath):
                shutil.copy(fpath, KAGGLE_DATASET_DIR)
                size_mb = os.path.getsize(fpath) / 1024**2
                copied.append(f"{os.path.basename(fpath)} ({size_mb:.1f} MB)")

        if not copied:
            print("  ⚠️ No files found to push yet.")
            return

        meta = {
            "title": "LLM2 Checkpoints - placeholder file",
            "id": KAGGLE_DATASET,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(f"{KAGGLE_DATASET_DIR}/dataset-metadata.json", "w") as f:
            json.dump(meta, f)

        result = subprocess.run([
            "kaggle", "datasets", "version",
            "-p", KAGGLE_DATASET_DIR,
            "-m", label,
            "--dir-mode", "skip"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ Permanently saved {len(copied)} files:")
            for c in copied:
                print(f"     → {c}")
        else:
            print(f"  ⚠️ Push failed: {result.stderr}")

    except Exception as e:
        print(f"  ⚠️ Push error: {e}")


# ── Tokenizer Class ───────────────────────────────────────
class GeneralTokenizer:

    def __init__(self):
        self.vocab      = {}
        self.merges     = {}
        self.vocab_size = 0

    def train(self, texts, target_vocab=VOCAB_SIZE):
        print(f"\n{'='*60}")
        print(f" Tokenizer Training")
        print(f" Texts : {len(texts):,}")
        print(f" Target vocab : {target_vocab:,}")
        print(f"{'='*60}")

        # Step 1 — word frequencies
        print("\n[1/4] Building word frequencies...")
        word_freq = Counter()
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"  processed {i:,}/{len(texts):,} texts")
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_freq.update(words)
        print(f"  unique words : {len(word_freq):,}")

        # Step 2 — character vocabulary
        print("\n[2/4] Building character vocabulary...")
        char_vocab = set()
        for word in word_freq:
            char_vocab.update(list(word))
        print(f"  unique chars : {len(char_vocab)}")

        # Step 3 — BPE merges
        n_merges = max(0, target_vocab - len(SPECIAL_TOKENS) - len(char_vocab))
        print(f"\n[3/4] Running {n_merges:,} BPE merges...")

        TOK_CKPT   = "/kaggle/working/tokenizer_checkpoint.json"
        start_merge = 0
        vocab       = {word: list(word) for word in word_freq}
        merges      = {}

        # Resume if checkpoint exists
        if os.path.exists(TOK_CKPT):
            with open(TOK_CKPT) as f:
                ckpt = json.load(f)
            merges      = {tuple(k.split("|||")): v for k, v in ckpt["merges"].items()}
            vocab       = ckpt["vocab"]
            start_merge = ckpt["done"]
            print(f"  Resumed tokenizer from merge {start_merge:,}")

        start = time.time()

        for i in range(start_merge, n_merges):

            # Count pairs
            pairs = Counter()
            for word, chars in vocab.items():
                freq = word_freq[word]
                for a, b in zip(chars, chars[1:]):
                    pairs[(a, b)] += freq

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            merges[best] = "".join(best)

            # Apply merge
            new_vocab = {}
            for word, chars in vocab.items():
                new_chars = []
                j = 0
                while j < len(chars):
                    if j < len(chars) - 1 and (chars[j], chars[j+1]) == best:
                        new_chars.append("".join(best))
                        j += 2
                    else:
                        new_chars.append(chars[j])
                        j += 1
                new_vocab[word] = new_chars
            vocab = new_vocab

            # Progress log every 500 merges
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                eta     = elapsed / (i + 1) * (n_merges - i - 1)
                print(f"  merge {i+1:5,}/{n_merges:,} "
                      f"({100*(i+1)/n_merges:.1f}%) "
                      f"best: {''.join(best):12s} "
                      f"elapsed: {elapsed:.0f}s ETA: {eta:.0f}s")

            # ✅ Save checkpoint + push to Kaggle every 1000 merges
            if (i + 1) % 1000 == 0:
                with open(TOK_CKPT, "w") as f:
                    json.dump({
                        "done"  : i + 1,
                        "merges": {"|||".join(k): v for k, v in merges.items()},
                        "vocab" : vocab
                    }, f)
                print(f"  💾 checkpoint saved — merge {i+1:,}")
                shutil.copy("/kaggle/working/journey_log.json",
                            "/kaggle/working/journey_backup.json")

                # ── PUSH TO KAGGLE PERMANENTLY ──
                push_to_kaggle(f"tokenizer merge {i+1}")

        # Step 4 — build final vocab
        print("\n[4/4] Building final vocabulary...")
        all_tokens = set()
        all_tokens.update(char_vocab)
        all_tokens.update(merges.values())
        for chars in vocab.values():
            all_tokens.update(chars)

        token_list      = SPECIAL_TOKENS + sorted(all_tokens)
        token_list      = token_list[:target_vocab]
        self.vocab      = {tok: i for i, tok in enumerate(token_list)}
        self.merges     = {str(k): v for k, v in merges.items()}
        self.vocab_size = len(self.vocab)

        elapsed = time.time() - start
        print(f"\nTokenizer Training Complete!")
        print(f"  Vocab size : {self.vocab_size:,}")
        print(f"  Total time : {elapsed:.0f}s ({elapsed/60:.1f} mins)")

    def encode(self, text):
        words  = re.findall(r'\w+|[^\w\s]', text.lower())
        ids    = []
        unk_id = self.vocab.get(UNK_TOKEN, 1)
        for word in words:
            chars = list(word)
            for pair, merged in self.merges.items():
                 a, b = pair if isinstance(pair, tuple) else tuple(pair.split("|||"))
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i+1] == b:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            for tok in chars:
                ids.append(self.vocab.get(tok, unk_id))
        return ids

    def decode(self, ids):
        id_to_tok = {v: k for k, v in self.vocab.items()}
        tokens    = [id_to_tok.get(i, UNK_TOKEN) for i in ids]
        return " ".join(t for t in tokens if t not in SPECIAL_TOKENS)

    def save(self, path=TOKENIZER_PATH):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)
        print(f"Tokenizer saved → {path}")

    def load(self, path=TOKENIZER_PATH):
        with open(path) as f:
            d           = json.load(f)
            self.vocab  = d["vocab"]
            self.merges = d["merges"]
        self.vocab_size = len(self.vocab)
        print(f"Tokenizer loaded! Vocab size: {self.vocab_size:,}")


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    from journey_log import log
    log("tokenizer.py", "RUNNING", "started")

    from datasets import load_dataset

    print("Loading Wikipedia dataset from Kaggle...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    print(f"Total articles: {len(dataset):,}")

    print("Sampling 100,000 articles for tokenizer training...")
    texts = []
    for i, item in enumerate(dataset):
        if i >= 100000:
            break
        texts.append(item["text"][:500])

    tok = GeneralTokenizer()
    tok.train(texts, target_vocab=VOCAB_SIZE)
    tok.save(TOKENIZER_PATH)

    # ✅ PUSH FINAL TOKENIZER TO KAGGLE PERMANENTLY
    push_to_kaggle("tokenizer COMPLETE - final")

    # Test
    sample  = "the quick brown fox jumps over the lazy dog"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"\nencode('{sample}')")
    print(f"  → {encoded[:20]}...")
    print(f"decode → '{decoded}'")

    log("tokenizer.py", "OK", "completed successfully")
