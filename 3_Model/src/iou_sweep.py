#!/usr/bin/env python3
"""
Recall-Diagnose: wo geht der Recall verloren — Nicht-Erkennung oder Form/Größe?

Läuft die Inferenz EINMAL je Testkachel, wendet das (getunte) Config-Postprocessing an
und wertet die predizierten Polygone gegen die GT aus — ohne Training, ohne neue Daten.
Zwei Auswertungen:

  1. Beste IoU PRO GT-Krone (schwellen-unabhängig): teilt jede GT-Krone ein in
       zero    : keine überlappende Vorhersage (best IoU == 0)  → echte Nicht-Erkennung
       partial : überlappt, aber best IoU < match-Schwelle       → Form/Größe daneben
       hit     : best IoU >= match-Schwelle                       → Treffer
     Viele 'zero' → Erkennungsproblem (Loss/Sampling). Viele 'partial' → Form-/PP-Problem.

  2. Precision/Recall/F1 über mehrere IoU-Schwellen (0.1 … 0.5): zeigt, wie stark der
     Recall bei lockererer Schwelle steigt (= wie viele Near-Miss die 0.5-Grenze frisst).

Muss als `nda` laufen. Beispiel (bestes 7.5cm-Modell):
    cd ~/leafline
    .venv/bin/python 3_Model/src/iou_sweep.py \
        --config 3_Model/configs/finetune_step1_ndom_spring75.yaml \
        --checkpoint 3_Model/runs/step1_ndom_spring75/checkpoints/best.pt \
        --resolution 7.5cm
"""

import argparse
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
from evaluate import iou_polygon, match_polygons, compute_metrics, RESOLUTION_FILE_SUFFIX


def best_iou_per_gt(preds, gts):
    """Für jede GT-Krone die höchste IoU zu irgendeiner Vorhersage (0 wenn keine Überlappung)."""
    out = []
    for g in gts:
        best = 0.0
        for p in preds:
            # billiger Bounding-Box-Vorabtest spart teure Polygon-Schnitte
            if p.bounds[0] > g.bounds[2] or p.bounds[2] < g.bounds[0] \
               or p.bounds[1] > g.bounds[3] or p.bounds[3] < g.bounds[1]:
                continue
            iou = iou_polygon(p, g)
            if iou > best:
                best = iou
        out.append(best)
    return out


def main():
    parser = argparse.ArgumentParser(description="Recall-Diagnose: IoU-Verteilung + Schwellen-Sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--resolution", default="7.5cm", choices=["7.5cm", "20cm", "20cm-spring"])
    parser.add_argument("--iou-thresholds", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--partial-thresh", type=float, default=0.5,
                        help="Schwelle für die zero/partial/hit-Einteilung (Default 0.5)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(cfg["paths"]["base"])
    stacked_dir = base / "stacked_6ch"
    pp = cfg["postprocessing"]
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

    # ── Inferenz + Postprocessing einmal je Kachel cachen ───────────────────
    tiles = []
    for item in cfg["data"][f"{args.split}_areas"]:
        area = item["area"]
        sp = stacked_dir / f"{area}{suffix}.tif"
        if not sp.exists():
            print(f"  {area}: {sp.name} fehlt, übersprungen"); continue
        with rasterio.open(sp) as src:
            img = src.read().astype(np.float32); transform = src.transform
        if channel_indices is not None:
            img = img[channel_indices]
        with torch.no_grad():
            out = predict_on_tile(model, torch.from_numpy(img).unsqueeze(0).to(device),
                                  patch_size=cfg["data"]["patch_size"],
                                  local_batch_size=cfg["training"]["batch_size"],
                                  stride=cfg["data"]["patch_size"] // 2)
        preds = extract_polygons(out[0, 0].cpu().numpy(), out[0, 1].cpu().numpy(),
                                 out[0, 2].cpu().numpy(), transform=transform, **pp)
        gts = list(gpd.read_file(base / item["gt_file"]).geometry)
        tiles.append((area, preds, gts))
        print(f"  {area}: {len(preds)} Vorhersagen, {len(gts)} GT-Kronen")

    if not tiles:
        print("Keine Kacheln."); return

    # ── 1. beste IoU pro GT-Krone → zero / partial / hit ────────────────────
    all_best = []
    for _, preds, gts in tiles:
        all_best.extend(best_iou_per_gt(preds, gts))
    all_best = np.array(all_best)
    n = len(all_best)
    t = args.partial_thresh
    zero = int((all_best == 0).sum())
    partial = int(((all_best > 0) & (all_best < t)).sum())
    hit = int((all_best >= t).sum())

    print(f"\n── GT-Kronen ({n} gesamt), beste IoU zu irgendeiner Vorhersage ──")
    print(f"  zero    (keine Überlappung, best IoU=0)   : {zero:4d}  ({100*zero/n:4.1f}%)  → Nicht-Erkennung")
    print(f"  partial (0 < IoU < {t})                     : {partial:4d}  ({100*partial/n:4.1f}%)  → Form/Größe")
    print(f"  hit     (IoU >= {t})                        : {hit:4d}  ({100*hit/n:4.1f}%)")
    med_overlap = float(np.median(all_best[all_best > 0])) if (all_best > 0).any() else 0.0
    print(f"  Median best-IoU der überlappenden GT       : {med_overlap:.3f}")

    # ── 2. P/R/F1 über IoU-Schwellen ────────────────────────────────────────
    print(f"\n── Precision/Recall/F1 über IoU-Schwellen ──")
    for thr in args.iou_thresholds:
        tp = fp = fn = 0
        for _, preds, gts in tiles:
            a, b, c = match_polygons(preds, gts, thr)
            tp += a; fp += b; fn += c
        p, r, f1 = compute_metrics(tp, fp, fn)
        print(f"  IoU>={thr:.1f}   P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    print("\nDeutung: viele 'zero' → Erkennungsproblem (Loss/Sampling); viel 'partial' bzw. "
          "stark steigender Recall bei lockerer IoU → Form/Größe (PP / dist-Loss / Metrik).")


if __name__ == "__main__":
    main()
