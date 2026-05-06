# ============================================================
# config.py — General LLM Configuration
# ============================================================

import torch

# ── Device ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Tokenizer ─────────────────────────────────────────────
VOCAB_SIZE     = 50000
TOKENIZER_PATH = "/kaggle/working/general_tokenizer.json"

# ── Model ─────────────────────────────────────────────────
EMBED_DIM      = 512
N_LAYERS       = 8
N_HEADS        = 8
FFN_DIM        = 2048   # ← ADD THIS (4 x EMBED_DIM)
BLOCK_SIZE     = 512
DROPOUT        = 0.1

# ── Training ──────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
EPOCHS         = 2
MAX_SAMPLES    = 50000

# ── Checkpointing ─────────────────────────────────────────
CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
CHECKPOINT_PATH = "/kaggle/working/checkpoints/general_llm.pt"
