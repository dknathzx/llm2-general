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
EMBED_DIM      = 512    # reduced from 768
N_LAYERS       = 8      # reduced from 12
N_HEADS        = 8      # reduced from 12
BLOCK_SIZE     = 512    # reduced from 1024
DROPOUT        = 0.1

# ── Training ──────────────────────────────────────────────
BATCH_SIZE     = 32     # increased (faster on GPU)
LEARNING_RATE  = 3e-4
EPOCHS         = 2      # reduced from 3
MAX_SAMPLES    = 50000  # reduced from 500000

# ── Checkpointing ─────────────────────────────────────────
CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
CHECKPOINT_PATH = "/kaggle/working/checkpoints/general_llm.pt"
