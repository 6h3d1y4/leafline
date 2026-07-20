#!/usr/bin/env python3
"""
Fine-tune DeepTreesModel on 6-channel 7.5 cm Kiel data.
Plain PyTorch training loop — no Lightning dependency.

Prerequisites:
    python 3_Model/src/prepare_data.py --config 3_Model/configs/finetune_v1.yaml

Usage:
    cd ~/leafline
    # With nDOM (6 channels, default)
    .venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_v1.yaml
    # Without nDOM (5 channels — height ablation)
    .venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_v1_no_ndom.yaml

Which variant runs is controlled entirely by model.in_channels in the config
(5 or 6) — see dataset.py::channel_indices_for. Both variants train with the
small-crown oversampling fix (dataset.py::KielPatchDataset(oversample_small=True)).

Resumes automatically from last.pt if it exists in the output directory.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6750 XT (gfx1031) has no prebuilt ROCm kernels; use gfx1030

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import KielPatchDataset, channel_indices_for
from model_utils import load_pretrained_6ch


def compute_pixel_f1(model, val_dl, device, n_thresholds=19):
    """Pixel-level PR over 19 thresholds → best F1 (mirrors DeepTreesModel.validation_step)."""
    thresholds = torch.linspace(0.05, 0.95, n_thresholds)
    tp = torch.zeros(n_thresholds)
    fp = torch.zeros(n_thresholds)
    fn = torch.zeros(n_thresholds)
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, loss_mask, loss_outline, loss_dist, *_ = model.shared_step((x, y), output=output)
            loss = loss_mask + loss_outline + 2.0 * loss_dist
            total_loss += loss.item()
            n_batches += 1

            mask_prob = torch.sigmoid(output[:, 0]).cpu()
            mask_true = (y[:, 0] > 0.5).cpu()
            for i, t in enumerate(thresholds):
                pred = mask_prob > t
                tp[i] += (pred & mask_true).sum()
                fp[i] += (pred & ~mask_true).sum()
                fn[i] += (~pred & mask_true).sum()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    best_idx = int(f1.argmax())
    return (
        float(f1[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(thresholds[best_idx]),
        total_loss / max(n_batches, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base        = Path(cfg["paths"]["base"])
    stacked_dir = base / "stacked_6ch"
    gt_dir      = base / "gt_bands"
    out_dir     = Path(cfg["paths"]["output"])
    ckpt_dir    = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CUDA context before allocating large dataset arrays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1, device=device)  # warm-up
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    patch_size  = cfg["data"]["patch_size"]
    bs          = cfg["training"]["batch_size"]
    in_channels = cfg["model"]["in_channels"]
    channel_indices = channel_indices_for(in_channels)
    print(f"in_channels={in_channels}  channel_indices={channel_indices if channel_indices is not None else 'all (0-5)'}")

    train_areas = [item["area"] for item in cfg["data"]["train_areas"]]
    valid_areas = [item["area"] for item in cfg["data"]["valid_areas"]]

    train_ds = KielPatchDataset(
        train_areas, stacked_dir, gt_dir, patch_size, augment=True,
        channel_indices=channel_indices, oversample_small=True,
    )
    val_ds = KielPatchDataset(
        valid_areas, stacked_dir, gt_dir, patch_size, augment=False,
        channel_indices=channel_indices, oversample_small=False,
    )
    print(f"Train patches/epoch : {len(train_ds)}")
    print(f"Valid patches/epoch : {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=bs, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=bs, num_workers=0)

    last_ckpt = ckpt_dir / "last.pt"
    pretrained = Path(cfg["paths"]["pretrained"]).expanduser()

    model = load_pretrained_6ch(str(pretrained), lr=cfg["model"]["lr"], apply_sigmoid=False, in_channels=in_channels)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"])

    start_epoch   = 0
    best_f1       = 0.0
    patience_left = cfg["training"]["patience"]

    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_f1       = ckpt.get("best_f1", 0.0)
        patience_left = ckpt.get("patience_left", cfg["training"]["patience"])
        print(f"  Resumed at epoch {start_epoch}, best F1={best_f1:.4f}")

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall", "val_threshold"
    ])
    if start_epoch == 0:
        log_writer.writeheader()

    max_epochs = cfg["training"]["max_epochs"]

    for epoch in range(start_epoch, max_epochs):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train    = 0
        for batch_idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss_mask, loss_outline, loss_dist, *_ = model.shared_step((x, y))
            # Distance transform is season-independent (tree height, not spectral appearance)
            # → weight it 2× vs mask/outline to handle spring domain gap
            loss = loss_mask + loss_outline + 2.0 * loss_dist
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train    += 1
            if batch_idx % 50 == 0:
                print(f"  epoch {epoch:3d}  batch {batch_idx:4d}/{len(train_ds)//bs}  loss={loss.item():.4f}", flush=True)

        train_loss /= max(n_train, 1)

        # ── Validation ──────────────────────────────────────────────────────
        val_f1, val_p, val_r, val_thresh, val_loss = compute_pixel_f1(model, val_dl, device)

        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_F1={val_f1:.4f}  (P={val_p:.3f} R={val_r:.3f} @ t={val_thresh:.2f})"
        )

        log_writer.writerow({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_f1": val_f1, "val_precision": val_p, "val_recall": val_r,
            "val_threshold": val_thresh,
        })
        log_file.flush()

        # ── Checkpoints ─────────────────────────────────────────────────────
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "patience_left": patience_left,
        }
        torch.save(state, ckpt_dir / "last.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = cfg["training"]["patience"]
            torch.save(state | {"best_f1": best_f1}, ckpt_dir / "best.pt")
            print(f"  ✓ New best F1={best_f1:.4f} — saved best.pt")
        else:
            patience_left -= 1
            print(f"  No improvement. Patience: {patience_left}/{cfg['training']['patience']}")
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    log_file.close()
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pt'}")
    print(f"Training log    : {log_path}")


if __name__ == "__main__":
    main()
