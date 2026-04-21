# ============================================================
# config.py — KONE LLM Configuration
# ============================================================

# ── Model Architecture ────────────────────────────────────
VOCAB_SIZE    = 8000   # KONE domain vocabulary
EMBED_DIM     = 256    # embedding dimension
N_HEADS       = 4      # attention heads
N_LAYERS      = 4      # transformer layers
FFN_DIM       = 1024   # feedforward hidden dim (4x embed)
BLOCK_SIZE    = 256    # context window
DROPOUT       = 0.1    # dropout

# ── Training ──────────────────────────────────────────────
BATCH_SIZE    = 4      # small for CPU
LEARNING_RATE = 3e-4
EPOCHS        = 3
EVAL_INTERVAL = 200
SAVE_INTERVAL = 500
GRAD_CLIP     = 1.0

# ── Paths ─────────────────────────────────────────────────
DATA_DIR       = "data"
CHECKPOINT_DIR = "checkpoints"
TOKENIZER_PATH = "kone_tokenizer.json"
MODEL_PATH     = "checkpoints/phase2_kone_llm.pt"

# ── Device ────────────────────────────────────────────────
DEVICE = "cpu"  # HP EliteBook — no CUDA GPU
