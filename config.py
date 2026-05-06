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
FFN_DIM        = 2048
BLOCK_SIZE     = 512
DROPOUT        = 0.1

# ── Training ──────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 3e-4
EPOCHS         = 2
MAX_SAMPLES    = 50000
GRAD_CLIP      = 1.0
WARMUP_STEPS   = 100
USE_AMP        = True
EVAL_INTERVAL  = 500
SAVE_INTERVAL  = 500

# ── Paths ──────────────────────────────────────────────────
CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
CHECKPOINT_PATH = "/kaggle/working/checkpoints/general_llm.pt"
MODEL_PATH      = "/kaggle/working/checkpoints/general_llm_final.pt"
LOG_PATH        = "/kaggle/working/journey_log.json"
