#!/usr/bin/env python3
"""
Evaluate a fine-tuned checkpoint on any split (train/valid/test).

Output CSV matches the baseline format:
  gebiet,aufloesung,pred,gt,tp,fp,fn,precision,recall,f1

Usage:
    cd ~/leafline
    .venv/bin/python 3_Model/src/evaluate.py \
        --config 3_Model/configs/finetune_v1.yaml \
        --checkpoint 3_Model/runs/v1/checkpoints/last.ckpt \
        --split test

    # Evaluate on the lower-resolution 20cm summer imagery instead of 7.5cm spring
    # (requires stacked_6ch/{area}_summer.tif, see prepare_data.py --summer):
    .venv/bin/python 3_Model/src/evaluate.py \
        --config 3_Model/configs/finetune_v1.yaml \
        --checkpoint 3_Model/runs/v1/checkpoints/last.ckpt \
        --split test --season summer
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
    parser.add_argument("--season", default="spring", choices=["spring", "summer"],
                        help="spring = 7.5cm DOP7-5 (default), summer = 20cm DOP20 resampled to 7.5cm grid")
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

    suffix = "_summer" if args.season == "summer" else ""

    for item in area_list:
        area = item["area"]
        stacked_path = stacked_dir / f"{area}{suffix}.tif"
        gt_shp = base / item["gt_file"]

        if not stacked_path.exists():
            print(f"  {area}: stacked TIF not found ({stacked_path.name}), skipping")
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
            "gebiet": area, "aufloesung": "20cm" if args.season == "summer" else "7.5cm",
            "pred": len(pred_polys), "gt": len(gt_polys),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
        results.append(row)
        print(f"  {area:30s} pred={len(pred_polys):4d}  gt={len(gt_polys):4d}  F1={f1:.3f}")

    if not results:
        print("No results — check that stacked TIFs exist and split is correct")
        return

    season_suffix = "_summer" if args.season == "summer" else ""
    out_path = Path(cfg["paths"]["output"]) / f"eval_{args.split}{season_suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Micro-average summary
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p, r, f1 = compute_metrics(total_tp, total_fp, total_fn)
    print(f"\nMicro-average  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
