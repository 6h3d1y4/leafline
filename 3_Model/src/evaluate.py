#!/usr/bin/env python3
"""
Evaluate a fine-tuned checkpoint on any split (train/valid/test).

Output CSV matches the baseline format:
  gebiet,aufloesung,pred,gt,tp,fp,fn,precision,recall,f1

Usage:
    cd ~/leafline
    # Evaluate on all three baseline-comparable resolutions (default):
    .venv/bin/python 3_Model/src/evaluate.py \
        --config 3_Model/configs/finetune_v1.yaml \
        --checkpoint 3_Model/runs/v1/checkpoints/last.pt \
        --split test

    # Restrict to a single resolution category:
    .venv/bin/python 3_Model/src/evaluate.py \
        --config 3_Model/configs/finetune_v1.yaml \
        --checkpoint 3_Model/runs/v1/checkpoints/last.pt \
        --split test --resolutions 20cm-spring
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
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import yaml

from deeptrees.model.deeptrees_model import DeepTreesModel
from deeptrees.modules.utils import predict_on_tile
from deeptrees.modules.postprocessing import extract_polygons

sys.path.insert(0, str(Path(__file__).parent))
from dataset import channel_indices_for


# Maps a baseline-comparable resolution category to the stacked-TIF filename suffix.
# "7.5cm" = native spring (existing {area}.tif). "20cm" = native summer (DOP20, no
# reprojection). "20cm-spring" = native spring at 20cm (DOP20-spring, no reprojection).
# Labels match baseline_model.ipynb's EVAL_PLAN exactly for direct comparability.
RESOLUTION_FILE_SUFFIX = {
    "7.5cm": "",
    "20cm": "_native20_summer",
    "20cm-spring": "_native20_spring",
}
RESOLUTION_LABEL = {
    "7.5cm": "7.5cm",
    "20cm": "20cm",
    "20cm-spring": "20cm-spring",
}


# ── Polygon-level evaluation (identical to baseline) ────────────────────────

def iou_polygon(p1: Polygon, p2: Polygon) -> float:
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def match_polygons(pred: list, gt: list, threshold: float = 0.5):
    matched = set()
    tp = 0
    for p in pred:
        best_iou, best_i = 0.0, -1
        for i, g in enumerate(gt):
            if i in matched:
                continue
            iou = iou_polygon(p, g)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= threshold:
            tp += 1
            matched.add(best_i)
    fp = len(pred) - tp
    fn = len(gt) - len(matched)
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--resolutions", nargs="+", default=["7.5cm", "20cm", "20cm-spring"],
                        choices=["7.5cm", "20cm", "20cm-spring"],
                        help="Which baseline-comparable resolution categories to evaluate - "
                             "defaults to all three so every checkpoint is tested on the full "
                             "matrix (7.5cm spring, 20cm summer, 20cm spring) to avoid bias.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(cfg["paths"]["base"])
    stacked_dir = base / "stacked_6ch"
    pp = cfg["postprocessing"]
    in_channels = cfg["model"]["in_channels"]
    channel_indices = channel_indices_for(in_channels)

    # Load model from plain PyTorch checkpoint (saved by train.py)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DeepTreesModel(
        in_channels=in_channels,
        apply_sigmoid=True,  # sigmoid needed for extract_polygons
        lr=cfg["model"]["lr"],
        num_backbones=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)

    results = []
    area_list = cfg["data"][f"{args.split}_areas"]

    for resolution in args.resolutions:
        suffix = RESOLUTION_FILE_SUFFIX[resolution]
        for item in area_list:
            area = item["area"]
            stacked_path = stacked_dir / f"{area}{suffix}.tif"
            gt_shp = base / item["gt_file"]

            if not stacked_path.exists():
                print(f"  {area} [{resolution}]: stacked TIF not found ({stacked_path.name}), skipping")
                continue

            with rasterio.open(stacked_path) as src:
                img = src.read().astype(np.float32)  # (6, H, W)
                transform = src.transform
            if channel_indices is not None:
                img = img[channel_indices]

            tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = predict_on_tile(
                    model,
                    tensor,
                    patch_size=cfg["data"]["patch_size"],
                    local_batch_size=cfg["training"]["batch_size"],
                    stride=cfg["data"]["patch_size"] // 2,
                )

            mask    = output[0, 0].cpu().numpy()
            outline = output[0, 1].cpu().numpy()
            dist    = output[0, 2].cpu().numpy()

            pred_polys = extract_polygons(mask, outline, dist, transform=transform, **pp)

            gt_gdf = gpd.read_file(gt_shp)
            gt_polys = list(gt_gdf.geometry)

            tp, fp, fn = match_polygons(pred_polys, gt_polys, args.iou_threshold)
            precision, recall, f1 = compute_metrics(tp, fp, fn)

            row = {
                "gebiet": area, "aufloesung": RESOLUTION_LABEL[resolution],
                "pred": len(pred_polys), "gt": len(gt_polys),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            }
            results.append(row)
            print(f"  {area:20s} [{resolution:11s}] pred={len(pred_polys):4d}  gt={len(gt_polys):4d}  F1={f1:.3f}")

    if not results:
        print("No results - check that stacked TIFs exist and split is correct")
        return

    out_path = Path(cfg["paths"]["output"]) / f"eval_{args.split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Per-resolution micro-average (helps spot which category drives the overall number)
    for resolution in args.resolutions:
        res_rows = [r for r in results if r["aufloesung"] == RESOLUTION_LABEL[resolution]]
        if not res_rows:
            continue
        r_tp = sum(r["tp"] for r in res_rows)
        r_fp = sum(r["fp"] for r in res_rows)
        r_fn = sum(r["fn"] for r in res_rows)
        rp, rr, rf1 = compute_metrics(r_tp, r_fp, r_fn)
        print(f"  [{resolution:11s}] Precision={rp:.3f}  Recall={rr:.3f}  F1={rf1:.3f}")

    # Micro-average summary
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p, r, f1 = compute_metrics(total_tp, total_fp, total_fn)
    print(f"\nMicro-average  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
