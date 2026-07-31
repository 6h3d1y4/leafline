#!/usr/bin/env python3
"""
Postprocessing-Sweep gegen Über-Segmentierung.

Läuft die (teure) Netz-Inferenz EINMAL je Testkachel und cached mask/outline/dist;
danach wird nur das (billige) Watershed-Postprocessing `extract_polygons` über ein
`min_dist × sigma`-Raster variiert und je Kombination die kronenweise Mikro-F1 gemessen.
Das sind die beiden Haupthebel gegen zu viele Marker pro Krone (siehe
debug_predictions.ipynb, Abschnitte 8-9); die übrigen PP-Parameter bleiben auf den
Config-Werten.

Zweck: für ein Modell/eine Auflösung ein sinnvolles PP finden (v.a. 7.5cm, wo die
Finetune-Modelle über-segmentieren), damit spätere Vergleiche - insbesondere der
nDOM-Effekt - nicht durch schlechtes PP verzerrt werden.

Muss als `nda` laufen (Datenzugriff). Beispiel:
    cd ~/leafline
    .venv/bin/python 3_Model/src/pp_sweep.py \
        --config 3_Model/configs/finetune_step1_spring75.yaml \
        --checkpoint 3_Model/runs/step1_spring75/checkpoints/best.pt \
        --resolution 7.5cm
    # eigenes Raster:
    #   --min-dist 10 20 30 40 50  --sigma 1 2 3 4
"""

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6750 XT: gfx1031 → gfx1030

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
from evaluate import (
    match_polygons, compute_metrics,
    RESOLUTION_FILE_SUFFIX, RESOLUTION_LABEL,
)


def main():
    parser = argparse.ArgumentParser(description="min_dist × sigma Postprocessing-Sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--resolution", default="7.5cm",
                        choices=["7.5cm", "20cm", "20cm-spring"],
                        help="Eine Auflösung (Standard 7.5cm - dort über-segmentieren die Modelle).")
    parser.add_argument("--min-dist", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40],
                        help="Watershed-Marker-Mindestabstand: größer = weniger, größere Marker/Krone.")
    parser.add_argument("--sigma", type=float, nargs="+", default=[1, 2, 3],
                        help="Gaussian-Glättung vor der Maxima-Suche: größer = weniger Spurious-Marker.")
    parser.add_argument("--outline-mult", type=float, nargs="+", default=None,
                        help="outline_multiplier-Werte: höher = tiefere Watershed-Barrieren an "
                             "Kronengrenzen (trennt gemergte Nachbarkronen). Default: nur Config-Wert.")
    parser.add_argument("--outline-exp", type=float, nargs="+", default=None,
                        help="outline_exp-Werte: höher = schärferer Grenzen-Kontrast. Default: nur Config-Wert.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold2", type=float, default=0.3,
                        help="Zweite IoU-Schwelle als Ko-Metrik (Default 0.3).")
    parser.add_argument("--out", default=None,
                        help="Ausgabe-CSV (Default: runs/<output>/pp_sweep_<resolution>.csv)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(cfg["paths"]["base"])
    stacked_dir = base / "stacked_6ch"
    base_pp = dict(cfg["postprocessing"])
    in_channels = cfg["model"]["in_channels"]
    channel_indices = channel_indices_for(in_channels)
    suffix = RESOLUTION_FILE_SUFFIX[args.resolution]

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DeepTreesModel(in_channels=in_channels, apply_sigmoid=True,
                           lr=cfg["model"]["lr"], num_backbones=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)

    # ── Inferenz einmal je Kachel, Ergebnisse cachen ────────────────────────
    tiles = []
    total_gt = 0
    for item in cfg["data"][f"{args.split}_areas"]:
        area = item["area"]
        stacked_path = stacked_dir / f"{area}{suffix}.tif"
        if not stacked_path.exists():
            print(f"  {area} [{args.resolution}]: stacked TIF fehlt ({stacked_path.name}), übersprungen")
            continue
        with rasterio.open(stacked_path) as src:
            img = src.read().astype(np.float32)
            transform = src.transform
        if channel_indices is not None:
            img = img[channel_indices]
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = predict_on_tile(
                model, tensor,
                patch_size=cfg["data"]["patch_size"],
                local_batch_size=cfg["training"]["batch_size"],
                stride=cfg["data"]["patch_size"] // 2,
            )
        mask    = output[0, 0].cpu().numpy()
        outline = output[0, 1].cpu().numpy()
        dist    = output[0, 2].cpu().numpy()
        gt_polys = list(gpd.read_file(base / item["gt_file"]).geometry)
        total_gt += len(gt_polys)
        tiles.append((area, mask, outline, dist, transform, gt_polys))
        print(f"  Inferenz gecached: {area}  (GT-Kronen: {len(gt_polys)})")

    if not tiles:
        print("Keine Kacheln - Pfade/Auflösung prüfen.")
        return

    om_grid = args.outline_mult if args.outline_mult is not None else [base_pp["outline_multiplier"]]
    oe_grid = args.outline_exp if args.outline_exp is not None else [base_pp["outline_exp"]]
    print(f"\nSweep min_dist={args.min_dist} × sigma={args.sigma} × "
          f"outline_mult={om_grid} × outline_exp={oe_grid}  "
          f"(Baseline-PP: min_dist={base_pp['min_dist']}, sigma={base_pp['sigma']}, "
          f"outline_multiplier={base_pp['outline_multiplier']}, outline_exp={base_pp['outline_exp']}) - "
          f"GT gesamt: {total_gt}\n")

    # ── Sweep (nur Postprocessing, keine erneute Inferenz) ──────────────────
    rows = []
    for md, sg, om, oe in itertools.product(args.min_dist, args.sigma, om_grid, oe_grid):
        pp = {**base_pp, "min_dist": md, "sigma": sg,
              "outline_multiplier": om, "outline_exp": oe}
        tp = fp = fn = tp2 = fp2 = fn2 = preds = 0
        for area, mask, outline, dist, transform, gt_polys in tiles:
            polys = extract_polygons(mask, outline, dist, transform=transform, **pp)
            a, b, c = match_polygons(polys, gt_polys, args.iou_threshold)
            a2, b2, c2 = match_polygons(polys, gt_polys, args.iou_threshold2)
            tp += a; fp += b; fn += c; preds += len(polys)
            tp2 += a2; fp2 += b2; fn2 += c2
        p, r, f1 = compute_metrics(tp, fp, fn)
        p2, r2, f12 = compute_metrics(tp2, fp2, fn2)
        rows.append({"min_dist": md, "sigma": sg,
                     "outline_multiplier": om, "outline_exp": oe,
                     "pred": preds, "gt": total_gt, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                     "precision_03": round(p2, 4), "recall_03": round(r2, 4),
                     "f1_03": round(f12, 4)})
        is_base = (md == base_pp["min_dist"] and sg == base_pp["sigma"]
                   and om == base_pp["outline_multiplier"] and oe == base_pp["outline_exp"])
        flag = "  ← Baseline-PP" if is_base else ""
        print(f"  min_dist={md:3d}  sigma={sg:<4g}  om={om:<4g} oe={oe:<4g}  pred={preds:5d}  "
              f"F1@.5={f1:.3f} (R={r:.3f})  F1@.3={f12:.3f}{flag}")

    rows.sort(key=lambda x: x["f1"], reverse=True)

    out_path = Path(args.out) if args.out else \
        Path(cfg["paths"]["output"]) / f"pp_sweep_{args.resolution}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print(f"\nBestes PP: min_dist={best['min_dist']} sigma={best['sigma']} "
          f"outline_multiplier={best['outline_multiplier']} outline_exp={best['outline_exp']}  "
          f"→ F1@.5={best['f1']:.4f}  F1@.3={best['f1_03']:.4f}  "
          f"(pred={best['pred']} vs GT={total_gt})")
    print(f"Saved: {out_path}")
    print("\nZum Übernehmen in den Config (postprocessing:):")
    print(f"  min_dist: {best['min_dist']}")
    print(f"  sigma: {best['sigma']}")
    print(f"  outline_multiplier: {best['outline_multiplier']}")
    print(f"  outline_exp: {best['outline_exp']}")


if __name__ == "__main__":
    main()
