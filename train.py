# ============================================================
# train.py — GENERAL LLM Training Loop
# Full detailed progress shown in terminal
# ============================================================

import torch
import torch.nn as nn
import os
import json
import time
from datetime import datetime
from model import GENERALModel
from dataset import get_dataloaders
from tokenizer import GENERALTokenizer
from config import (
    LEARNING_RATE, EPOCHS, EVAL_INTERVAL,
    SAVE_INTERVAL, GRAD_CLIP, DEVICE,
    CHECKPOINT_DIR, MODEL_PATH, TOKENIZER_PATH
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_lr(step, warmup_steps=100):
    if step < warmup_steps:
        return LEARNING_RATE * step / warmup_steps
    return LEARNING_RATE


@torch.no_grad()
def evaluate(model, val_loader, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y    = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def save_checkpoint(model, optimizer, epoch, step, loss, path):
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✅ checkpoint saved → {path}  ({size_mb:.1f} MB)")


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print("  No checkpoint found — starting fresh")
        return 0, 0
    print(f"  Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  ✅ Resumed from epoch {ckpt['epoch']} step {ckpt['step']} loss {ckpt['loss']:.4f}")
    return ckpt["epoch"], ckpt["step"]


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def train():
    print("=" * 60)
    print("GENERAL LLM — Training Started")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}")
    print("=" * 60)

    # ── Model ─────────────────────────────────────────────
    print("\n[1/5] Building model...")
    model = GENERALModel().to(DEVICE)
    total_params = model.count_params()
    print(f"  Parameters : {total_params:,}")
    print(f"  Model size : ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # ── Optimizer ─────────────────────────────────────────
    print("\n[2/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    print(f"  Optimizer  : AdamW")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Grad clip  : {GRAD_CLIP}")

    # ── Data ──────────────────────────────────────────────
    print("\n[3/5] Loading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    total_steps = len(train_loader) * EPOCHS
    print(f"  Total steps   : {total_steps:,}")

    # ── Resume ────────────────────────────────────────────
    print("\n[4/5] Checking for checkpoint...")
    start_epoch, global_step = load_checkpoint(model, optimizer, MODEL_PATH)

    # ── Training ──────────────────────────────────────────
    print("\n[5/5] Starting training loop...")
    print("=" * 60)

    log        = []
    t_train    = time.time()
    best_loss  = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS} — {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        t_epoch    = time.time()
        t_step     = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # learning rate warmup
            for g in optimizer.param_groups:
                g["lr"] = get_lr(global_step)

            # forward + backward
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            # print every 10 steps
            if step % 10 == 0:
                steps_done     = step + 1
                steps_total    = len(train_loader)
                pct            = steps_done / steps_total * 100
                elapsed        = time.time() - t_epoch
                eta_epoch      = (elapsed / steps_done) * (steps_total - steps_done)
                avg_loss_so_far = epoch_loss / steps_done
                lr_now         = optimizer.param_groups[0]["lr"]

                print(
                    f"  step {steps_done:>5}/{steps_total}  "
                    f"({pct:>5.1f}%)  "
                    f"loss: {loss.item():.4f}  "
                    f"avg: {avg_loss_so_far:.4f}  "
                    f"lr: {lr_now:.6f}  "
                    f"elapsed: {format_time(elapsed)}  "
                    f"ETA: {format_time(eta_epoch)}"
                )

            # eval
            if global_step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_loader)
                print(f"\n  --- EVAL at step {global_step} ---")
                print(f"  train_loss : {loss.item():.4f}")
                print(f"  val_loss   : {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"  🎯 New best val loss: {best_loss:.4f}")
                print()

                log.append({
                    "epoch":      epoch + 1,
                    "step":       global_step,
                    "train_loss": round(loss.item(), 4),
                    "val_loss":   round(val_loss, 4),
                    "time":       datetime.now().isoformat(),
                })

            # checkpoint
            if global_step % SAVE_INTERVAL == 0:
                print(f"\n  --- CHECKPOINT at step {global_step} ---")
                save_checkpoint(model, optimizer, epoch, global_step, loss.item(), MODEL_PATH)
                print()

        # epoch summary
        avg_loss   = epoch_loss / len(train_loader)
        epoch_time = time.time() - t_epoch
        total_time = time.time() - t_train

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} COMPLETE")
        print(f"  Avg loss   : {avg_loss:.4f}")
        print(f"  Best loss  : {best_loss:.4f}")
        print(f"  Epoch time : {format_time(epoch_time)}")
        print(f"  Total time : {format_time(total_time)}")
        print(f"  Finished   : {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # save epoch checkpoint
        epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch+1, global_step, avg_loss, epoch_path)

    # final save
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    save_checkpoint(model, optimizer, EPOCHS, global_step, avg_loss, MODEL_PATH)
    total_time = time.time() - t_train
    print(f"  Total time : {format_time(total_time)}")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # save log
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("\nTraining log saved → training_log.json")
    print("\nNext step: python inference.py")


if __name__ == "__main__":
    train()
