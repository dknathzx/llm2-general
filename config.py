# ============================================================
# config.py — General LLM maximized for Kaggle P100 GPU
# Target: ~117M parameters (GPT2 size)
# GPU: NVIDIA Tesla P100 (16GB VRAM)
# ============================================================

import torch

# ── Model Architecture ────────────────────────────────────
VOCAB_SIZE   = 50000   # large general vocabulary
EMBED_DIM    = 768     # embedding dimension (was 256)
N_HEADS      = 12      # attention heads    (was 4)
N_LAYERS     = 12      # transformer layers (was 4)
FFN_DIM      = 3072    # feedforward dim = 4x embed_dim
BLOCK_SIZE   = 1024    # context window    (was 256)
DROPOUT      = 0.1     # dropout

# ── Training ──────────────────────────────────────────────
BATCH_SIZE        = 16        # P100 16GB handles this well
LEARNING_RATE     = 3e-4
EPOCHS            = 3
EVAL_INTERVAL     = 200
SAVE_INTERVAL     = 500
GRAD_CLIP         = 1.0
WARMUP_STEPS      = 1000      # learning rate warmup
USE_AMP           = True      # mixed precision FP16 — faster on P100

# ── Data ──────────────────────────────────────────────────
MAX_SAMPLES       = 500000    # 500K samples from Wikipedia + StackOverflow

# ── Paths ─────────────────────────────────────────────────
DATA_DIR          = "/kaggle/working/data"
CHECKPOINT_DIR    = "/kaggle/working/checkpoints"
TOKENIZER_PATH    = "/kaggle/working/general_tokenizer.json"
MODEL_PATH        = "/kaggle/working/checkpoints/general_llm.pt"
LOG_PATH          = "/kaggle/working/training_log.json"

# ── Device ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Parameter estimate ────────────────────────────────────
# 12 layers x 768 dim x 12 heads = ~117M params
# Fits comfortably in P100 16GB VRAM
