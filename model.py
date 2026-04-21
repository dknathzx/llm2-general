# ============================================================
# model.py — General LLM (~117M Parameter Transformer)
# Scaled up for Kaggle P100 GPU (16GB VRAM)
# Same architecture as before — just bigger dimensions
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import (
    VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS,
    FFN_DIM, BLOCK_SIZE, DROPOUT, DEVICE
)

# ── Single Attention Head ─────────────────────────────────
class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.Wq = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.Wk = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.Wv = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, C = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ V

# ── Multi-Head Attention ──────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = EMBED_DIM // N_HEADS
        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(N_HEADS)])
        self.proj  = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))

# ── FeedForward Layer ─────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),                        # GELU instead of ReLU — better for LLMs
            nn.Linear(FFN_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

# ── Transformer Block ─────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffn  = FeedForward()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.ln2  = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ── Full General LLM ──────────────────────────────────────
class GeneralLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop      = nn.Dropout(DROPOUT)
        self.blocks    = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_final  = nn.LayerNorm(EMBED_DIM)
        self.head      = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

        # weight tying
        self.token_emb.weight = self.head.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        device = idx.device
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    model = GeneralLLM().to(DEVICE)
    total = model.count_params()
    print(f"Device         : {DEVICE}")
    print(f"Model params   : {total:,}")
    print(f"Model size     : ~{total * 4 / 1024**2:.1f} MB")
    x = torch.randint(0, VOCAB_SIZE, (2, 64)).to(DEVICE)
    logits, loss = model(x, x)
    print(f"Input shape    : {x.shape}")
    print(f"Output shape   : {logits.shape}")
    print(f"Loss           : {loss.item():.4f}")
