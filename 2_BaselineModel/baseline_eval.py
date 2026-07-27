#!/usr/bin/env python3
"""
Baseline-Evaluation als eigenständiges Skript — reproduziert exakt die EVAL_PLAN-
Schleife aus baseline_model.ipynb (un-finetuntes freudenberg2022-Modell), damit die
Baseline-Zahlen gespeichert und direkt neben den Finetuning-Runs abgelegt werden können.

Deckungsgleich mit 3_Model/src/evaluate.py:
  - dieselbe iou_polygon / match_polygons / compute_metrics (IoU-Schwelle 0.5),
  - dieselbe Auflösungs-Matrix (7.5cm / 20cm / 20cm-spring) und CSV-Spaltenstruktur,
  - dasselbe Postprocessing-Default.
Unterschied zu evaluate.py (bewusst, weil es die *Baseline* ist):
  - Modell = DeepTreesModel(in_channels=5, apply_sigmoid=True), Gewichte aus
    freudenberg2022.pt via torch.jit.load → tcd_backbone (KEIN Finetune-Checkpoint),
  - Input wird direkt aus den DOP-RGBI-Kacheln + berechnetem NDVI gebaut (nicht aus
    stacked_6ch), nativ ohne Resampling — exakt wie im Baseline-Notebook,
  - Tiling je Auflösung wie im Notebook (7.5cm: 682/341, 20cm*: 256/128).

Muss als `nda` laufen (Datenzugriff auf Data/Kiel). Beispiel:
    cd ~/leafline
    .venv/bin/python 2_BaselineModel/baseline_eval.py
    # getuntes Postprocessing gegentesten (vgl. debug_predictions.ipynb):
    .venv/bin/python 2_BaselineModel/baseline_eval.py --min-dist 30 --sigma 2 \
        --out 3_Model/runs/baseline_eval_pp30.csv
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6750 XT: gfx1031 → gfx1030

import numpy as np
import torch
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

from deeptrees.model.deeptrees_model import DeepTreesModel
from deeptrees.modules.utils import predict_on_tile
from deeptrees.modules.postprocessing import extract_polygons

# 1-basierte Bandreihenfolge im RGBI-GeoTIFF
BAND_RED, BAND_GRN, BAND_BLU, BAND_NIR = 1, 2, 3, 4
IOU_THRESHOLD = 0.5
LOCAL_BATCH_SIZE = 16

# Auflösungs-Matrix (Reihenfolge + Tiling wie baseline_model.ipynb EVAL_PLAN)
RES_SPEC = {
    "7.5cm":       dict(subdir="DOP7-5",       patch_size=682, stride=341),
    "20cm":        dict(subdir="DOP20",        patch_size=256, stride=128),
    "20cm-spring": dict(subdir="DOP20-spring", patch_size=256, stride=128),
}
EVAL_PLAN = [
    ("BotGarten", "7.5cm"), ("BotGarten", "20cm"), ("BotGarten", "20cm-spring"),
    ("HoernNord", "7.5cm"), ("HoernNord", "20cm"), ("HoernNord", "20cm-spring"),
]

# Postprocessing-Default identisch zum Baseline-Notebook / finetune-Configs
PP_DEFAULT = dict(
    mask_exp=2, outline_multiplier=5, outline_exp=1, dist_exp=0.5,
    sigma=1, binary_threshold=0.10, min_dist=10, label_threshold=0.10,
    area_min=3, simplify=0.3,
)


# ── Polygon-Metriken (identisch zu evaluate.py / baseline_model.ipynb) ──────

def iou_polygon(p1: Polygon, p2: Polygon) -> float:
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def match_polygons(pred: list, gt: list, threshold: float = IOU_THRESHOLD):
    matched_gt = set()
    tp = 0
    for p in pred:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            iou = iou_polygon(p, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= threshold:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_baseline_model(model_path: Path, device):
    """DeepTreesModel(in_channels=5) mit freudenberg2022-Gewichten — wie baseline_model.ipynb."""
    model = DeepTreesModel(
        in_channels=5, architecture="Unet", backbone="resnet18",
        apply_sigmoid=True, num_backbones=1,
    )
    jit_model = torch.jit.load(str(model_path), map_location="cpu")
    model.tcd_backbone.load_state_dict(jit_model.state_dict())
    model.eval()
    return model.to(device)


def load_rgbi_ndvi(rgbi_path: Path):
    """RGBI (nativ, /255) + berechnetes NDVI → (5, H, W), plus transform. Kein Resampling."""
    with rasterio.open(rgbi_path) as src:
        rgbi = src.read(
            indexes=[BAND_RED, BAND_GRN, BAND_BLU, BAND_NIR],
        ).astype(np.float32) / 255.0
        transform = src.transform
    nir, red = rgbi[3], rgbi[0]
    ndvi = ((nir - red) / (nir + red + 1e-8))[np.newaxis]  # (1, H, W)
    img = np.concatenate([rgbi, ndvi], axis=0)             # (5, H, W)
    return img, transform


def main():
    parser = argparse.ArgumentParser(description="Baseline-Evaluation (freudenberg2022) über die Auflösungs-Matrix")
    parser.add_argument("--base", default="/home/leafline/leafline/Data/Kiel/TrainingAreas",
                        help="Wurzel mit DOP7-5/ DOP20/ DOP20-spring/ test/")
    parser.add_argument("--model", default="~/pretrained_models/freudenberg2022.pt")
    # Default in 3_Model/runs/ — das Verzeichnis gehört nda und ist beschreibbar;
    # 2_BaselineModel/ gehört leafline (als nda nicht schreibbar). Liegt damit auch
    # direkt neben den Finetune-Eval-CSVs.
    parser.add_argument("--out", default="3_Model/runs/baseline_eval.csv")
    parser.add_argument("--resolutions", nargs="+", default=list(RES_SPEC.keys()),
                        choices=list(RES_SPEC.keys()))
    parser.add_argument("--min-dist", type=int, default=None,
                        help="Postprocessing min_dist überschreiben (Default 10; getunt: 30)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Postprocessing sigma überschreiben (Default 1; getunt: 2)")
    args = parser.parse_args()

    base = Path(args.base)
    gt_dir = base / "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pp = dict(PP_DEFAULT)
    if args.min_dist is not None:
        pp["min_dist"] = args.min_dist
    if args.sigma is not None:
        pp["sigma"] = args.sigma
    print(f"Postprocessing: min_dist={pp['min_dist']} sigma={pp['sigma']}")

    model = load_baseline_model(Path(args.model).expanduser(), device)
    print("Baseline-Modell geladen (freudenberg2022, in_channels=5).")

    results = []
    plan = [(a, r) for (a, r) in EVAL_PLAN if r in args.resolutions]
    for area, res in plan:
        spec = RES_SPEC[res]
        rgbi_path = base / spec["subdir"] / f"{area}.tif"
        gt_shp = gt_dir / f"{area}_GroundTruth.shp"

        if not rgbi_path.exists():
            print(f"  {area} [{res}]: RGBI-Kachel fehlt ({rgbi_path}), übersprungen")
            continue

        img, transform = load_rgbi_ndvi(rgbi_path)
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 5, H, W)
        with torch.no_grad():
            output = predict_on_tile(
                model, tensor,
                patch_size=spec["patch_size"], local_batch_size=LOCAL_BATCH_SIZE,
                stride=spec["stride"],
            )
        mask    = output[0, 0].cpu().numpy()
        outline = output[0, 1].cpu().numpy()
        dist    = output[0, 2].cpu().numpy()

        pred_polys = extract_polygons(mask, outline, dist, transform=transform, **pp)
        gt_polys = list(gpd.read_file(gt_shp).geometry)

        tp, fp, fn = match_polygons(pred_polys, gt_polys)
        precision, recall, f1 = compute_metrics(tp, fp, fn)
        results.append({
            "gebiet": area, "aufloesung": res,
            "pred": len(pred_polys), "gt": len(gt_polys),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
        print(f"  {area:20s} [{res:11s}] pred={len(pred_polys):4d}  gt={len(gt_polys):4d}  F1={f1:.3f}")

    if not results:
        print("Keine Ergebnisse — Pfade/Auflösungen prüfen.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Per-Auflösungs-Mikroschnitt + Gesamt (wie evaluate.py)
    for res in args.resolutions:
        rows = [r for r in results if r["aufloesung"] == res]
        if not rows:
            continue
        tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
        p, r_, f1 = compute_metrics(tp, fp, fn)
        print(f"  [{res:11s}] Precision={p:.3f}  Recall={r_:.3f}  F1={f1:.3f}")

    tp = sum(r["tp"] for r in results); fp = sum(r["fp"] for r in results); fn = sum(r["fn"] for r in results)
    p, r_, f1 = compute_metrics(tp, fp, fn)
    print(f"\nMicro-average  Precision={p:.3f}  Recall={r_:.3f}  F1={f1:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
