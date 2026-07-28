#!/usr/bin/env python3
"""
Segmentierungs-Diagnose: quantifiziert Über-Segmentierung vs. Merging (Unter-Segmentierung).

Die 'partial'-Kronen (überlappt, aber IoU<0.5) sind eine Mischung aus:
  - Über-Segmentierung : 1 GT-Krone → mehrere Vorhersage-Polygone
  - Merging            : mehrere GT-Kronen → 1 Vorhersage-Polygon
Beide senken die IoU. Welcher Anteil überwiegt, entscheidet den Hebel:
  viel Merging  → outline-Kanal stärken (PP outline_multiplier/exp, oder outline-Loss-Gewicht)
  viel Über-Seg → Watershed-Marker (min_dist/sigma)

Zählt je GT die überlappenden Vorhersagen und je Vorhersage die überlappten GT
(Überlappung = IoU >= --overlap-iou). Läuft als `nda`. Beispiel:
    .venv/bin/python 3_Model/src/seg_diagnostic.py \
        --config 3_Model/configs/finetune_step1_ndom_spring75.yaml \
        --checkpoint 3_Model/runs/step1_ndom_spring75/checkpoints/best.pt --resolution 7.5cm
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import numpy as np
import torch
import geopandas as gpd
import rasterio
import yaml

from deeptrees.model.deeptrees_model import DeepTreesModel
from deeptrees.modules.utils import predict_on_tile
from deeptrees.modules.postprocessing import extract_polygons

sys.path.insert(0, str(Path(__file__).parent))
from dataset import channel_indices_for
from evaluate import iou_polygon, RESOLUTION_FILE_SUFFIX


def _bbox_disjoint(a, b):
    return (a.bounds[0] > b.bounds[2] or a.bounds[2] < b.bounds[0]
            or a.bounds[1] > b.bounds[3] or a.bounds[3] < b.bounds[1])


def main():
    ap = argparse.ArgumentParser(description="Über-Segmentierung vs. Merging quantifizieren")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--resolution", default="7.5cm", choices=["7.5cm", "20cm", "20cm-spring"])
    ap.add_argument("--overlap-iou", type=float, default=0.1,
                    help="Ab welcher IoU ein GT/Pred-Paar als 'überlappend' zählt (Default 0.1)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(cfg["paths"]["base"]); stacked_dir = base / "stacked_6ch"
    pp = cfg["postprocessing"]; in_ch = cfg["model"]["in_channels"]
    ci = channel_indices_for(in_ch); suffix = RESOLUTION_FILE_SUFFIX[args.resolution]

    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DeepTreesModel(in_channels=in_ch, apply_sigmoid=True, lr=cfg["model"]["lr"], num_backbones=1)
    model.load_state_dict(ck["model_state_dict"]); model.eval(); model.to(device)

    preds_per_gt, gts_per_pred = [], []   # gepoolt über alle Kacheln
    for item in cfg["data"][f"{args.split}_areas"]:
        area = item["area"]; sp = stacked_dir / f"{area}{suffix}.tif"
        if not sp.exists():
            print(f"  {area}: {sp.name} fehlt, übersprungen"); continue
        with rasterio.open(sp) as src:
            img = src.read().astype(np.float32); transform = src.transform
        xin = img[ci] if ci is not None else img
        with torch.no_grad():
            out = predict_on_tile(model, torch.from_numpy(xin).unsqueeze(0).to(device),
                                  patch_size=cfg["data"]["patch_size"],
                                  local_batch_size=cfg["training"]["batch_size"],
                                  stride=cfg["data"]["patch_size"] // 2)
        preds = extract_polygons(out[0, 0].cpu().numpy(), out[0, 1].cpu().numpy(),
                                 out[0, 2].cpu().numpy(), transform=transform, **pp)
        gts = list(gpd.read_file(base / item["gt_file"]).geometry)

        # Überlappungsmatrix (IoU >= overlap_iou)
        ov = [[False] * len(gts) for _ in preds]
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                if _bbox_disjoint(p, g):
                    continue
                if iou_polygon(p, g) >= args.overlap_iou:
                    ov[pi][gi] = True
        preds_per_gt += [sum(ov[pi][gi] for pi in range(len(preds))) for gi in range(len(gts))]
        gts_per_pred += [sum(ov[pi]) for pi in range(len(preds))]
        print(f"  {area}: {len(preds)} Vorhersagen, {len(gts)} GT-Kronen")

    ppg = np.array(preds_per_gt); gpp = np.array(gts_per_pred)
    n_gt, n_pred = len(ppg), len(gpp)

    print(f"\n── Über-Segmentierung: Vorhersagen pro GT-Krone (Überlappung IoU>={args.overlap_iou}) ──")
    b0 = int((ppg == 0).sum()); b1 = int((ppg == 1).sum()); b2 = int((ppg >= 2).sum())
    print(f"  0 (verpasst)      : {b0:4d}  ({100*b0/n_gt:4.1f}%)")
    print(f"  1 (sauber)        : {b1:4d}  ({100*b1/n_gt:4.1f}%)")
    print(f"  >=2 (über-segm.)  : {b2:4d}  ({100*b2/n_gt:4.1f}%)   mean/GT (bei >=1): "
          f"{ppg[ppg>=1].mean():.2f}")

    print(f"\n── Merging: GT-Kronen pro Vorhersage (Überlappung IoU>={args.overlap_iou}) ──")
    m0 = int((gpp == 0).sum()); m1 = int((gpp == 1).sum()); m2 = int((gpp >= 2).sum())
    print(f"  0 (Fehlalarm)     : {m0:4d}  ({100*m0/n_pred:4.1f}%)")
    print(f"  1 (sauber)        : {m1:4d}  ({100*m1/n_pred:4.1f}%)")
    print(f"  >=2 (Merging)     : {m2:4d}  ({100*m2/n_pred:4.1f}%)   mean GT/pred (bei >=1): "
          f"{gpp[gpp>=1].mean():.2f}")

    print(f"\nDeutung: >=2-Vorhersagen/GT = Über-Segmentierung; >=2-GT/Vorhersage = Merging. "
          f"Der größere Block bestimmt den Hebel (Merging → outline stärken).")

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"seg_diagnostic_{args.resolution}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "bucket", "count", "pct"])
        w.writerow(["preds_per_gt", "0", b0, round(100*b0/n_gt, 2)])
        w.writerow(["preds_per_gt", "1", b1, round(100*b1/n_gt, 2)])
        w.writerow(["preds_per_gt", ">=2", b2, round(100*b2/n_gt, 2)])
        w.writerow(["gts_per_pred", "0", m0, round(100*m0/n_pred, 2)])
        w.writerow(["gts_per_pred", "1", m1, round(100*m1/n_pred, 2)])
        w.writerow(["gts_per_pred", ">=2", m2, round(100*m2/n_pred, 2)])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
