# ============================================================
# tokenizer.py — KONE Domain BPE Tokenizer
# ============================================================

import json
import os
import re
import time
from collections import Counter
from config import VOCAB_SIZE, TOKENIZER_PATH


class KONETokenizer:

    def __init__(self):
        self.vocab       = {}
        self.id_to_token = {}
        self.merges      = {}
        self.vocab_size  = VOCAB_SIZE
        self.PAD_TOKEN   = "<PAD>"
        self.UNK_TOKEN   = "<UNK>"
        self.BOS_TOKEN   = "<BOS>"
        self.EOS_TOKEN   = "<EOS>"

    def train(self, texts: list):
        print("=" * 60)
        print("KONE BPE Tokenizer Training")
        print("=" * 60)
        t_start = time.time()

        # Step 1
        print("\n[1/4] Building word frequencies...")
        word_freqs = Counter()
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                word_freqs[word] += 1
            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1:,}/{len(texts):,} texts — {len(word_freqs):,} unique words so far")
        print(f"  DONE — {len(word_freqs):,} unique words")
        print(f"  time so far: {time.time()-t_start:.1f}s")

        # Step 2
        print("\n[2/4] Building character vocabulary...")
        vocab = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + ["</w>"])
            vocab[chars] = freq
        print(f"  DONE — {len(vocab):,} word entries built")

        # Step 3
        num_merges = self.vocab_size - 256 - 4
        print(f"\n[3/4] Running {num_merges} BPE merges...")
        t_merge = time.time()

        for i in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                print(f"  no more pairs at merge {i} — stopping")
                break
            best_pair = max(pairs, key=pairs.get)
            vocab     = self._merge_vocab(best_pair, vocab)
            self.merges[best_pair] = "".join(best_pair)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_merge
                pct     = (i + 1) / num_merges * 100
                eta     = (elapsed / (i + 1)) * (num_merges - i - 1)
                print(
                    f"  merge {i+1:>4}/{num_merges}  "
                    f"({pct:.1f}%)  "
                    f"best_pair: {''.join(best_pair):<15}  "
                    f"freq: {pairs[best_pair]:>5}  "
                    f"elapsed: {elapsed:.0f}s  "
                    f"ETA: {eta:.0f}s"
                )

        print(f"  DONE — {len(self.merges):,} merges completed")

        # Step 4
        print("\n[4/4] Building final vocabulary...")
        all_tokens = set()
        all_tokens.update([self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN])
        for chars in vocab:
            for ch in chars:
                all_tokens.add(ch)
        for merged in self.merges.values():
            all_tokens.add(merged)

        self.vocab       = {tok: i for i, tok in enumerate(sorted(all_tokens))}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        total = time.time() - t_start
        print(f"  DONE — vocab size: {len(self.vocab):,}")
        print(f"\n{'='*60}")
        print(f"Tokenizer Training Complete!")
        print(f"  Total time : {total:.1f}s  ({total/60:.1f} mins)")
        print(f"  Vocab size : {len(self.vocab):,}")
        print(f"  Merges     : {len(self.merges):,}")
        print(f"{'='*60}")

    def _get_pairs(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram    = re.escape(" ".join(pair))
        pattern   = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in vocab.items():
            word_str     = " ".join(word)
            new_word_str = pattern.sub("".join(pair), word_str)
            new_word     = tuple(new_word_str.split())
            new_vocab[new_word] = freq
        return new_vocab

    def encode(self, text: str) -> list:
        tokens = []
        words  = text.lower().split()
        for word in words:
            chars = list(word) + ["</w>"]
            while len(chars) > 1:
                pairs     = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]
                if not mergeable:
                    break
                best   = min(mergeable, key=lambda x: list(self.merges.keys()).index(x[0]))
                pair   = best[0]
                merged = best[1]
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars)-1 and (chars[i], chars[i+1]) == pair:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            for ch in chars:
                tokens.append(self.vocab.get(ch, self.vocab.get(self.UNK_TOKEN, 1)))
        return tokens

    def decode(self, ids: list) -> str:
        tokens = [self.id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        text   = " ".join(tokens)
        text   = text.replace("</w>", "").replace("  ", " ").strip()
        return text

    def save(self, path: str = TOKENIZER_PATH):
        print(f"\nSaving tokenizer to {path}...")
        data = {
            "vocab":  self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"Tokenizer saved! File size: {size_mb:.2f} MB")

    def load(self, path: str = TOKENIZER_PATH):
        print(f"Loading tokenizer from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab       = data["vocab"]
        self.id_to_token = {int(i): tok for tok, i in self.vocab.items()}
        self.merges      = {tuple(k.split("|||")): v for k, v in data["merges"].items()}
        print(f"Tokenizer loaded! Vocab size: {len(self.vocab):,}")

    def __len__(self):
        return len(self.vocab)


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("Step 1 — Loading data")
    print("=" * 60)
    t0 = time.time()

    with open("data/kone_train.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()

    print(f"Total lines in file : {len(texts):,}")
    texts = texts[:10000]
    print(f"Using for training  : {len(texts):,} samples")
    print(f"Load time           : {time.time()-t0:.1f}s")

    print("\nStep 2 — Training tokenizer")
    tok = KONETokenizer()
    tok.train(texts)

    print("\nStep 3 — Saving tokenizer")
    tok.save()

    print("\nStep 4 — Testing encode/decode")
    test_texts = [
        "elevator fault code F23",
        "vpn not working cannot connect",
        "password reset request new employee",
        "laptop screen black after update",
    ]
    print("-" * 60)
    for text in test_texts:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        print(f"  input   : {text}")
        print(f"  encoded : {encoded[:8]}... ({len(encoded)} tokens)")
        print(f"  decoded : {decoded}")
        print()

    print("Tokenizer ready! Run python train.py next.")
