# ============================================================
# train.py — General LLM Training Loop
# Optimized for Kaggle P100 GPU (16GB VRAM)
# Features:
#   - Mixed precision FP16 (2x faster on P100)
#   - Checkpoint save every N steps
#   - Auto resume from last checkpoint
#   - Detailed progress output
#   - Learning rate warmup + cosine decay
# ============================================================

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
import math

from config import (
    VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS,
    FFN_DIM, BLOCK_SIZE, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    EVAL_INTERVAL, SAVE_INTERVAL, GRAD_CLIP,
    WARMUP_STEPS, USE_AMP,
    CHECKPOINT_DIR, MODEL_PATH, LOG_PATH, DEVICE
)
from model import GeneralLLM
from tokenizer import GeneralTokenizer
from dataset import get_dataloaders

# ── Helpers ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, epoch, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
    }, path)
    size_mb = os.path.getsize(path) / 1024**2
    print(f"  ✅ checkpoint saved → {path}  ({size_mb:.1f} MB)")

def load_checkpoint(model, optimizer, scaler, path):
    if not os.path.exists(path):
        return 0, 0
    print(f"  Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt["epoch"]
    step  = ckpt["step"]
    print(f"  Resumed from checkpoint — epoch {epoch}  step {step:,}")
    return epoch, step

def get_lr(step, warmup_steps, max_steps, base_lr):
    """Warmup then cosine decay"""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, val_dl, device, use_amp):
    model.eval()
    total_loss = 0
    count      = 0
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
        count      += 1
        if count >= 50:  # limit eval batches for speed
            break
    model.train()
    return total_loss / max(count, 1)

# ── Main training ─────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  General LLM Training")
    print(f"  Device   : {DEVICE}")
    print(f"  AMP FP16 : {USE_AMP}")
    print(f"{'='*60}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # [1/5] Model
    print("\n[1/5] Building model...")
    model = GeneralLLM().to(DEVICE)
    total_params = model.count_params()
    print(f"  Parameters : {total_params:,}")
    print(f"  Model size : ~{total_params * 4 / 1024**2:.1f} MB")

    # [2/5] Optimizer
    print("\n[2/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 0.1,
        betas        = (0.9, 0.95)
    )
    scaler = GradScaler() if USE_AMP and DEVICE == "cuda" else None
    print(f"  Optimizer  : AdamW")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Grad clip  : {GRAD_CLIP}")

    # [3/5] Load checkpoint if exists
    print("\n[3/5] Checking for checkpoint...")
    start_epoch, global_step = load_checkpoint(model, optimizer, scaler, MODEL_PATH)

    # [4/5] Tokenizer
    print("\n[4/5] Loading tokenizer...")
    tok = GeneralTokenizer()
    tok.load()

    # [5/5] Data
    print("\n[5/5] Loading data...")
    train_dl, val_dl = get_dataloaders(tok)

    total_steps = len(train_dl) * EPOCHS
    print(f"\n  Epochs       : {EPOCHS}")
    print(f"  Steps/epoch  : {len(train_dl):,}")
    print(f"  Total steps  : {total_steps:,}")

    # ── Training loop ─────────────────────────────────────
    log = []
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        epoch_start = time.time()
        running_loss = 0
        count        = 0

        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # learning rate schedule
            lr = get_lr(global_step, WARMUP_STEPS, total_steps, LEARNING_RATE)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # forward + backward
            optimizer.zero_grad()
            if USE_AMP and DEVICE == "cuda":
                with autocast():
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item()
            count        += 1
            global_step  += 1

            # progress
            if global_step % 10 == 0:
                avg_loss  = running_loss / count
                elapsed   = time.time() - epoch_start
                steps_done = step + 1
                steps_left = len(train_dl) - steps_done
                eta        = elapsed / steps_done * steps_left if steps_done > 0 else 0
                pct        = 100 * steps_done / len(train_dl)

                print(f"  step {global_step:6,}  "
                      f"epoch {epoch+1}/{EPOCHS}  "
                      f"({pct:5.1f}%)  "
                      f"loss: {loss.item():.4f}  "
                      f"avg: {avg_loss:.4f}  "
                      f"lr: {lr:.2e}  "
                      f"elapsed: {elapsed/3600:.1f}h  "
                      f"ETA: {eta/3600:.1f}h")

            # eval
            if global_step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_dl, DEVICE, USE_AMP)
                avg_loss = running_loss / count
                print(f"\n  --- EVAL at step {global_step:,} ---")
                print(f"  train_loss : {avg_loss:.4f}")
                print(f"  val_loss   : {val_loss:.4f}\n")
                log.append({
                    "step": global_step, "epoch": epoch + 1,
                    "train_loss": avg_loss, "val_loss": val_loss
                })

            # checkpoint
            if global_step % SAVE_INTERVAL == 0:
                avg_loss = running_loss / count
                print(f"\n  --- CHECKPOINT at step {global_step:,} ---")
                save_checkpoint(model, optimizer, scaler,
                                epoch + 1, global_step, avg_loss, MODEL_PATH)
                # save log
                with open(LOG_PATH, "w") as f:
                    json.dump(log, f, indent=2)
                print()

        # end of epoch checkpoint
        avg_loss = running_loss / max(count, 1)
        epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        print(f"\n  --- END OF EPOCH {epoch+1} ---")
        print(f"  avg train loss : {avg_loss:.4f}")
        save_checkpoint(model, optimizer, scaler,
                        epoch + 1, global_step, avg_loss, epoch_path)
        save_checkpoint(model, optimizer, scaler,
                        epoch + 1, global_step, avg_loss, MODEL_PATH)

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "general_llm_final.pt")
    save_checkpoint(model, optimizer, scaler, EPOCHS, global_step, avg_loss, final_path)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Final model → {final_path}")
    print(f"  Next step   : python inference.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
