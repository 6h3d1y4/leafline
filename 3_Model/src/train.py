#!/usr/bin/env python3
"""
Fine-tune DeepTreesModel on stacked Kiel data (5- or 6-channel).
Plain PyTorch training loop — no Lightning dependency.

Prerequisites:
    python 3_Model/src/prepare_data.py --config <config>

Usage:
    cd ~/leafline
    # Single run (fixed train/valid split from the config):
    .venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_step1_spring75.yaml

    # Optional cross-validation over a learning-rate grid (Schedule Schritt 1
    # "cross-fold to get best hyperparam setup"). Pools train+valid areas, runs
    # k folds per LR, reports mean±std val_F1, writes runs/<out>/cv_results.csv.
    # No checkpoints are saved in CV mode — it only measures hyperparameter effect.
    .venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_step1_spring75.yaml \
        --cv-folds 5 --lr-grid 5e-5,1e-4,2e-4

What is controlled by the config:
  - model.in_channels (5 = RGBI+NDVI, 6 = +nDOM)          → dataset.channel_indices_for
  - data.augment.{hflip,vflip,brightness,contrast,        → dataset.DEFAULT_AUGMENT
                  nir_scale,ndvi_scale,spectral_noise}       (omitted section = all on)
  - training.oversample_small (default False)              → small-crown oversampling

Single-run mode resumes automatically from last.pt if present in the output dir.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6750 XT (gfx1031) has no prebuilt ROCm kernels; use gfx1030

import numpy as np
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


def build_model(cfg, in_channels, lr, device):
    """Fresh model from the pretrained checkpoint (re-created per CV fold)."""
    pretrained = Path(cfg["paths"]["pretrained"]).expanduser()
    model = load_pretrained_6ch(str(pretrained), lr=lr, apply_sigmoid=False, in_channels=in_channels)
    return model.to(device)


def make_loaders(cfg, train_areas, valid_areas, stacked_dir, gt_dir, channel_indices):
    """Build train (augmented) and valid (clean) DataLoaders from area-name lists."""
    patch_size = cfg["data"]["patch_size"]
    bs         = cfg["training"]["batch_size"]
    aug_cfg    = cfg["data"].get("augment", None)
    oversample = cfg["training"].get("oversample_small", False)

    train_ds = KielPatchDataset(
        train_areas, stacked_dir, gt_dir, patch_size, augment=True,
        channel_indices=channel_indices, oversample_small=oversample,
        augment_config=aug_cfg,
    )
    val_ds = KielPatchDataset(
        valid_areas, stacked_dir, gt_dir, patch_size, augment=False,
        channel_indices=channel_indices, oversample_small=False,
    )
    train_dl = DataLoader(train_ds, batch_size=bs, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=bs, num_workers=0)
    return train_ds, val_ds, train_dl, val_dl


def train_evaluate(cfg, model, train_ds, train_dl, val_dl, device, *,
                   lr, max_epochs, patience, ckpt_dir=None, log_path=None, resume=False, tag=""):
    """Run the epoch loop with early stopping; return best val_f1.

    If ckpt_dir is given, best.pt/last.pt are saved (single-run mode). If None, no
    checkpoints are written (CV mode). resume only applies when ckpt_dir is set.
    """
    bs = cfg["training"]["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch, best_f1, patience_left = 0, 0.0, patience
    if resume and ckpt_dir is not None and (ckpt_dir / "last.pt").exists():
        print(f"Resuming from {ckpt_dir / 'last.pt'}")
        ckpt = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_f1       = ckpt.get("best_f1", 0.0)
        patience_left = ckpt.get("patience_left", patience)
        print(f"  Resumed at epoch {start_epoch}, best F1={best_f1:.4f}")

    log_writer = log_file = None
    if log_path is not None:
        log_file = open(log_path, "a", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=[
            "epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall", "val_threshold"
        ])
        if start_epoch == 0:
            log_writer.writeheader()

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
                print(f"  {tag}epoch {epoch:3d}  batch {batch_idx:4d}/{len(train_ds)//bs}  loss={loss.item():.4f}", flush=True)

        train_loss /= max(n_train, 1)

        # ── Validation ──────────────────────────────────────────────────────
        val_f1, val_p, val_r, val_thresh, val_loss = compute_pixel_f1(model, val_dl, device)
        print(
            f"{tag}Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_F1={val_f1:.4f}  (P={val_p:.3f} R={val_r:.3f} @ t={val_thresh:.2f})"
        )

        if log_writer is not None:
            log_writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_f1": val_f1, "val_precision": val_p, "val_recall": val_r,
                "val_threshold": val_thresh,
            })
            log_file.flush()

        # ── Checkpoints ─────────────────────────────────────────────────────
        if ckpt_dir is not None:
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
            patience_left = patience
            if ckpt_dir is not None:
                torch.save(state | {"best_f1": best_f1}, ckpt_dir / "best.pt")
                print(f"  ✓ New best F1={best_f1:.4f} — saved best.pt")
        else:
            patience_left -= 1
            print(f"  {tag}No improvement. Patience: {patience_left}/{patience}")
            if patience_left <= 0:
                print(f"{tag}Early stopping at epoch {epoch}")
                break

    if log_file is not None:
        log_file.close()
    return best_f1


def run_single(cfg, stacked_dir, gt_dir, channel_indices, in_channels, device):
    out_dir  = Path(cfg["paths"]["output"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_areas = [item["area"] for item in cfg["data"]["train_areas"]]
    valid_areas = [item["area"] for item in cfg["data"]["valid_areas"]]

    train_ds, val_ds, train_dl, val_dl = make_loaders(
        cfg, train_areas, valid_areas, stacked_dir, gt_dir, channel_indices)
    print(f"Train patches/epoch : {len(train_ds)}")
    print(f"Valid patches/epoch : {len(val_ds)}")

    lr = cfg["model"]["lr"]
    model = build_model(cfg, in_channels, lr, device)
    best_f1 = train_evaluate(
        cfg, model, train_ds, train_dl, val_dl, device,
        lr=lr, max_epochs=cfg["training"]["max_epochs"], patience=cfg["training"]["patience"],
        ckpt_dir=ckpt_dir, log_path=out_dir / "train_log.csv", resume=True,
    )
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pt'}")


def run_cv(cfg, stacked_dir, gt_dir, channel_indices, in_channels, device, n_folds, lr_grid):
    """k-fold CV over a learning-rate grid. Measures hyperparameter effect only —
    trains fresh from the pretrained checkpoint per fold, saves no model checkpoints."""
    out_dir = Path(cfg["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pool all labelled areas (train + valid); fold over area names.
    all_areas = ([item["area"] for item in cfg["data"]["train_areas"]]
                 + [item["area"] for item in cfg["data"]["valid_areas"]])
    all_areas = list(dict.fromkeys(all_areas))  # dedupe, preserve order
    if n_folds > len(all_areas):
        print(f"--cv-folds {n_folds} > {len(all_areas)} areas; clamping to {len(all_areas)}")
        n_folds = len(all_areas)

    rng = np.random.default_rng(42)
    shuffled = list(rng.permutation(all_areas))
    folds = [shuffled[i::n_folds] for i in range(n_folds)]  # round-robin, balanced sizes

    print(f"CV: {n_folds} folds over {len(all_areas)} areas, LR grid = {lr_grid}")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {f}")

    cv_rows = []
    for lr in lr_grid:
        fold_f1s = []
        for fold_idx in range(n_folds):
            val_areas   = folds[fold_idx]
            train_areas = [a for j, f in enumerate(folds) if j != fold_idx for a in f]
            tag = f"[lr={lr:g} fold={fold_idx}] "
            print(f"\n── CV {tag.strip()} — train {len(train_areas)} / val {len(val_areas)} areas ──")

            train_ds, _, train_dl, val_dl = make_loaders(
                cfg, train_areas, val_areas, stacked_dir, gt_dir, channel_indices)
            model = build_model(cfg, in_channels, lr, device)
            best_f1 = train_evaluate(
                cfg, model, train_ds, train_dl, val_dl, device,
                lr=lr, max_epochs=cfg["training"]["max_epochs"],
                patience=cfg["training"]["patience"], ckpt_dir=None, log_path=None, resume=False,
                tag=tag,
            )
            fold_f1s.append(best_f1)
            cv_rows.append({"lr": lr, "fold": fold_idx, "best_val_f1": best_f1})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean, std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
        print(f"\n══ lr={lr:g}: mean val_F1={mean:.4f} ± {std:.4f}  (folds: {[round(x,3) for x in fold_f1s]})")

    cv_path = out_dir / "cv_results.csv"
    with open(cv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "fold", "best_val_f1"])
        writer.writeheader()
        writer.writerows(cv_rows)

    # Best LR by mean fold F1
    by_lr = {}
    for r in cv_rows:
        by_lr.setdefault(r["lr"], []).append(r["best_val_f1"])
    best_lr = max(by_lr, key=lambda k: np.mean(by_lr[k]))
    print(f"\nCV done. Results: {cv_path}")
    print(f"Best LR by mean val_F1: {best_lr:g}  (mean {np.mean(by_lr[best_lr]):.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="If >0, run k-fold cross-validation instead of a single run.")
    parser.add_argument("--lr-grid", type=str, default="",
                        help="Comma-separated LRs for CV (e.g. 5e-5,1e-4,2e-4). "
                             "Defaults to just model.lr from the config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base        = Path(cfg["paths"]["base"])
    stacked_dir = base / "stacked_6ch"
    gt_dir      = base / "gt_bands"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1, device=device)  # warm-up CUDA context before large allocations
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    in_channels = cfg["model"]["in_channels"]
    channel_indices = channel_indices_for(in_channels)
    print(f"in_channels={in_channels}  channel_indices={channel_indices if channel_indices is not None else 'all (0-5)'}")
    print(f"augment={{**DEFAULT, **{cfg['data'].get('augment', {})}}}  "
          f"oversample_small={cfg['training'].get('oversample_small', False)}")

    if args.cv_folds and args.cv_folds > 0:
        lr_grid = ([float(x) for x in args.lr_grid.split(",")] if args.lr_grid
                   else [cfg["model"]["lr"]])
        run_cv(cfg, stacked_dir, gt_dir, channel_indices, in_channels, device, args.cv_folds, lr_grid)
    else:
        run_single(cfg, stacked_dir, gt_dir, channel_indices, in_channels, device)


if __name__ == "__main__":
    main()
