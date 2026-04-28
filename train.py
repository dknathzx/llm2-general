# ============================================================
# train.py — General LLM Training Loop
# Optimized for Kaggle GPU
# Features:
#   - Mixed precision FP16 (2x faster on GPU)
#   - Checkpoint save every N steps
#   - Auto push to Kaggle dataset permanently
#   - Auto resume from dataset checkpoint
#   - Detailed progress output
#   - Learning rate warmup + cosine decay
# ============================================================
import sys
sys.path.append('/kaggle/working/llm2-general')
import shutil
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
import math
import subprocess

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

# ── Kaggle Dataset Config ─────────────────────────────────
KAGGLE_DATASET     = "dwarakanathk/llm2-checkpoints-placeholder-file"
KAGGLE_DATASET_DIR = "/kaggle/working/llm2-train-checkpoints"
AUTO_PUSH          = True

# ── Setup Kaggle API ──────────────────────────────────────
def setup_kaggle_credentials():
    os.environ["KAGGLE_USERNAME"] = "dwarakanathk"
    os.environ["KAGGLE_KEY"]      = "KGAT_97ff8b4a8d918070c5209ec1e5c84858"
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({
            "username": "dwarakanathk",
            "key":      "KGAT_97ff8b4a8d918070c5209ec1e5c84858"
        }, f)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)

# ── Push checkpoints to Kaggle dataset ───────────────────
def push_to_kaggle(label="update"):
    if not AUTO_PUSH:
        return
    try:
        setup_kaggle_credentials()
        print(f"\n  📤 Pushing to Kaggle dataset... [{label}]")

        os.makedirs(KAGGLE_DATASET_DIR, exist_ok=True)

        # files to save
        files_to_save = [
            MODEL_PATH,
            LOG_PATH,
            "/kaggle/working/journey_log.json",
            "/kaggle/working/journey_backup.json",
        ]

        # also grab epoch checkpoints
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                files_to_save.append(os.path.join(CHECKPOINT_DIR, f))

        copied = []
        for fpath in files_to_save:
            if os.path.exists(fpath):
                shutil.copy(fpath, KAGGLE_DATASET_DIR)
                size_mb = os.path.getsize(fpath) / 1024**2
                copied.append(f"{os.path.basename(fpath)} ({size_mb:.1f} MB)")

        if not copied:
            print("  ⚠️ No files found to push yet.")
            return

        # dataset metadata
        meta = {
            "title": "LLM2 Checkpoints - placeholder file",
            "id":    KAGGLE_DATASET,
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

# ── Download checkpoint from dataset ─────────────────────
def download_from_kaggle():
    try:
        setup_kaggle_credentials()
        print("  📥 Checking dataset for existing checkpoint...")
        result = subprocess.run([
            "kaggle", "datasets", "download",
            KAGGLE_DATASET,
            "-p", "/kaggle/working/",
            "--unzip"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Downloaded from dataset!")
        else:
            print("  ⚠️ No checkpoint in dataset yet — starting fresh!")
    except Exception as e:
        print(f"  ⚠️ Download error: {e}")

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
        print("  No checkpoint found — starting fresh")
        return 0, 0
    print(f"  Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt["epoch"]
    step  = ckpt["step"]
    print(f"  ✅ Resumed from epoch {epoch} step {step:,} loss {ckpt['loss']:.4f}")
    return epoch, step

def get_lr(step, warmup_steps, max_steps, base_lr):
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
        if count >= 50:
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

    # Download checkpoint from dataset first
    print("\n[0/5] Checking dataset for checkpoint...")
    download_from_kaggle()

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

    # [3/5] Load checkpoint
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

        epoch_start  = time.time()
        running_loss = 0
        count        = 0

        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)

            lr = get_lr(global_step, WARMUP_STEPS, total_steps, LEARNING_RATE)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

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
                avg_loss   = running_loss / count
                elapsed    = time.time() - epoch_start
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

            # checkpoint + push to dataset
            if global_step % SAVE_INTERVAL == 0:
                avg_loss = running_loss / count
                print(f"\n  --- CHECKPOINT at step {global_step:,} ---")
                save_checkpoint(model, optimizer, scaler,
                                epoch + 1, global_step, avg_loss, MODEL_PATH)
                with open(LOG_PATH, "w") as f:
                    json.dump(log, f, indent=2)
                # ✅ PUSH TO KAGGLE PERMANENTLY
                push_to_kaggle(f"step {global_step}")
                print()

        # end of epoch
        avg_loss   = running_loss / max(count, 1)
        epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        print(f"\n  --- END OF EPOCH {epoch+1} ---")
        print(f"  avg train loss : {avg_loss:.4f}")
        save_checkpoint(model, optimizer, scaler,
                        epoch + 1, global_step, avg_loss, epoch_path)
        save_checkpoint(model, optimizer, scaler,
                        epoch + 1, global_step, avg_loss, MODEL_PATH)
        # ✅ PUSH EPOCH CHECKPOINT TO KAGGLE
        push_to_kaggle(f"epoch {epoch+1} complete")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "general_llm_final.pt")
    save_checkpoint(model, optimizer, scaler, EPOCHS, global_step, avg_loss, final_path)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    # ✅ PUSH FINAL MODEL TO KAGGLE
    push_to_kaggle("FINAL MODEL COMPLETE")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Final model → {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
