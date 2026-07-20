#!/usr/bin/env python3
"""
Pre-process training data for 6-channel fine-tuning.

Creates in TrainingAreas/:
  stacked_6ch/{area}.tif        — 6-band float32 [0,1]: R G B I NDVI nDOM  (7.5cm spring)
  stacked_6ch/{area}_summer.tif — same format, DOP20 resampled to 7.5cm    (20cm summer)
  gt_bands/{area}_gt.tif        — 3-band float32 [0,1]: mask | outline | dist_transform

Run once before training. Safe to re-run (skips existing files unless --overwrite).

Usage:
    cd ~/leafline
    # Spring 7.5cm (required)
    .venv/bin/python 3_Model/src/prepare_data.py --config 3_Model/configs/finetune_v1.yaml
    # Add summer 20cm variants (optional, for domain bridging)
    .venv/bin/python 3_Model/src/prepare_data.py --config 3_Model/configs/finetune_v1.yaml --summer
"""

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.features import rasterize as rio_rasterize
from rasterio.enums import Resampling as RS
from scipy.ndimage import distance_transform_edt, binary_erosion
from shapely.geometry import mapping
import yaml


def load_and_stack(dop_path: Path, ndom_path: Path):
    """
    Load RGBI TIF + nDOM TIF, return (6, H, W) float32 in [0, 1].

    Channel order: R G B I NDVI nDOM
    - RGBI  → divide by 255
    - NDVI  = (NIR - R) / (NIR + R + ε), rescaled from [-1,1] to [0,1]
    - nDOM  → per-tile min-max to [0,1]
    """
    with rasterio.open(dop_path) as src:
        rgbi = src.read([1, 2, 3, 4]).astype(np.float32) / 255.0
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform

    with rasterio.open(ndom_path) as src:
        H, W = rgbi.shape[1], rgbi.shape[2]
        ndom = src.read(
            1,
            out_shape=(H, W),
            resampling=RS.bilinear,
        ).astype(np.float32)

    r, nir = rgbi[0], rgbi[3]
    ndvi_raw = (nir - r) / (nir + r + 1e-8)
    ndvi_norm = ((ndvi_raw + 1.0) / 2.0).astype(np.float32)

    ndom_min, ndom_max = float(ndom.min()), float(ndom.max())
    if ndom_max > ndom_min:
        ndom_norm = ((ndom - ndom_min) / (ndom_max - ndom_min)).astype(np.float32)
    else:
        ndom_norm = np.zeros_like(ndom)

    stacked = np.stack([rgbi[0], rgbi[1], rgbi[2], rgbi[3], ndvi_norm, ndom_norm])
    return stacked, profile, crs, transform


def load_and_stack_summer(dop20_path: Path, ndom75_path: Path, dop75_path: Path):
    """
    Load DOP20 (20cm summer RGBI) + nDOM7-5 (7.5cm height), return (6, H, W).

    DOP7-5 is used as the reference grid (CRS + transform + size) because the
    nDOM7-5 TIFs have no embedded CRS. DOP20 is reprojected to match DOP7-5.
    nDOM7-5 is resampled to DOP7-5 size (simple reshape, no reprojection needed).
    """
    with rasterio.open(dop75_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_h, ref_w = ref.height, ref.width
        ref_profile = ref.profile.copy()

    with rasterio.open(dop20_path) as src:
        rgbi = np.zeros((4, ref_h, ref_w), dtype=np.float32)
        for band_idx in range(4):
            rasterio.warp.reproject(
                source=rasterio.band(src, band_idx + 1),
                destination=rgbi[band_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=RS.bilinear,
            )

    with rasterio.open(ndom75_path) as src:
        ndom = src.read(1, out_shape=(ref_h, ref_w), resampling=RS.bilinear).astype(np.float32)

    rgbi = (rgbi / 255.0).clip(0.0, 1.0)

    r, nir = rgbi[0], rgbi[3]
    ndvi_raw = (nir - r) / (nir + r + 1e-8)
    ndvi_norm = ((ndvi_raw + 1.0) / 2.0).astype(np.float32)

    ndom_min, ndom_max = float(ndom.min()), float(ndom.max())
    if ndom_max > ndom_min:
        ndom_norm = ((ndom - ndom_min) / (ndom_max - ndom_min)).astype(np.float32)
    else:
        ndom_norm = np.zeros_like(ndom)

    stacked = np.stack([rgbi[0], rgbi[1], rgbi[2], rgbi[3], ndvi_norm, ndom_norm])

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff", crs=ref_crs, transform=ref_transform,
        width=ref_w, height=ref_h, count=6, dtype="float32",
        compress="lzw", photometric="MINISBLACK",
    )

    return stacked, out_profile, ref_crs, ref_transform


def rasterize_gt(gdf, shape, transform, crs):
    """
    Convert polygon GeoDataFrame to (mask, outline, dist_transform) arrays.

    - mask:    1 inside crown, 0 outside
    - outline: crown boundary pixels (mask minus 2-px erosion)
    - dist:    distance to nearest non-tree pixel, normalized to [0, 1]
    """
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if len(gdf) == 0:
        warnings.warn("Empty GT — all arrays will be zero")
        z = np.zeros(shape, dtype=np.float32)
        return z, z, z

    geoms = [(mapping(geom), 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    mask = rio_rasterize(
        geoms,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    eroded = binary_erosion(mask, iterations=2)
    outline = (mask.astype(bool) & ~eroded).astype(np.float32)

    dist = distance_transform_edt(mask).astype(np.float32)
    dmax = dist.max()
    dist_norm = dist / dmax if dmax > 0 else dist

    return mask.astype(np.float32), outline, dist_norm


def process_area(area, gt_shp, dop_dir, ndom_dir, out_stacked, out_gt, overwrite=False):
    out_s = out_stacked / f"{area}.tif"
    out_g = out_gt / f"{area}_gt.tif"

    if out_s.exists() and out_g.exists() and not overwrite:
        print(f"  {area}: already exists, skipping")
        return

    dop_path = dop_dir / f"{area}.tif"
    ndom_path = ndom_dir / f"{area}_nDOM.tif"

    if not dop_path.exists():
        print(f"  {area}: DOP not found at {dop_path}, skipping")
        return
    if not ndom_path.exists():
        print(f"  {area}: nDOM not found at {ndom_path}, skipping")
        return

    print(f"  {area}: stacking rasters ...", end="", flush=True)
    stacked, profile, crs, transform = load_and_stack(dop_path, ndom_path)
    H, W = stacked.shape[1], stacked.shape[2]
    print(f" {H}×{W}")

    out_stacked.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=6, dtype="float32", compress="lzw", photometric="MINISBLACK")
    with rasterio.open(out_s, "w", **p) as dst:
        dst.write(stacked)

    if gt_shp is not None and gt_shp.exists():
        print(f"  {area}: rasterizing GT ...", end="", flush=True)
        gdf = gpd.read_file(gt_shp)
        mask_f, outline, dist = rasterize_gt(gdf, (H, W), transform, crs)

        out_gt.mkdir(parents=True, exist_ok=True)
        p2 = profile.copy()
        p2.update(count=3, dtype="float32", compress="lzw", photometric="MINISBLACK")
        with rasterio.open(out_g, "w", **p2) as dst:
            dst.write(mask_f, 1)
            dst.write(outline, 2)
            dst.write(dist, 3)
        print(f" {len(gdf)} crowns → {out_g.name}")
    else:
        print(f"  {area}: no GT shapefile — skipping GT creation")


def process_area_summer(area, dop20_dir, dop75_dir, ndom75_dir, out_stacked, out_gt, overwrite=False):
    """
    Process summer 20cm DOP variant for a training area.
    Output: stacked_6ch/{area}_summer.tif + gt_bands/{area}_summer_gt.tif
    GT is copied from the existing {area}_gt.tif (tree positions don't change seasonally).
    """
    out_s = out_stacked / f"{area}_summer.tif"
    out_g = out_gt / f"{area}_summer_gt.tif"

    if out_s.exists() and out_g.exists() and not overwrite:
        print(f"  {area}_summer: already exists, skipping")
        return

    dop20_path  = dop20_dir  / f"{area}.tif"
    dop75_path  = dop75_dir  / f"{area}.tif"
    ndom75_path = ndom75_dir / f"{area}_nDOM.tif"
    src_gt = out_gt / f"{area}_gt.tif"

    if not dop20_path.exists():
        print(f"  {area}_summer: DOP20 not found at {dop20_path}, skipping")
        return
    if not dop75_path.exists():
        print(f"  {area}_summer: DOP7-5 not found at {dop75_path}, skipping")
        return
    if not ndom75_path.exists():
        print(f"  {area}_summer: nDOM7-5 not found at {ndom75_path}, skipping")
        return
    if not src_gt.exists():
        print(f"  {area}_summer: GT not found at {src_gt} — run without --summer first")
        return

    print(f"  {area}_summer: resampling DOP20 to 7.5cm grid ...", end="", flush=True)
    stacked, out_profile, _, _ = load_and_stack_summer(dop20_path, ndom75_path, dop75_path)
    H, W = stacked.shape[1], stacked.shape[2]
    print(f" {H}×{W}")

    out_stacked.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_s, "w", **out_profile) as dst:
        dst.write(stacked)

    out_gt.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_gt, out_g)
    print(f"  {area}_summer: GT copied → {out_g.name}")


def main():
    parser = argparse.ArgumentParser(description="Pre-process 6-channel training data")
    parser.add_argument("--config", required=True, help="Path to finetune YAML config")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing files")
    parser.add_argument("--summer", action="store_true",
                        help="Also process DOP20 summer variants for train/valid/test areas")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["paths"]["base"])
    dop_dir   = base / "DOP7-5"
    ndom_dir  = base / "nDOM7-5"
    dop20_dir = base / "DOP20"
    out_stacked = base / "stacked_6ch"
    out_gt      = base / "gt_bands"

    splits = {
        "train": cfg["data"]["train_areas"],
        "valid": cfg["data"]["valid_areas"],
        "test":  cfg["data"]["test_areas"],
    }

    for split, area_list in splits.items():
        print(f"\n── {split} ──")
        for item in area_list:
            area = item["area"]
            if area.endswith("_summer"):
                continue  # handled below
            gt_shp = base / item["gt_file"] if item.get("gt_file") else None
            process_area(area, gt_shp, dop_dir, ndom_dir, out_stacked, out_gt, args.overwrite)

    if args.summer:
        print("\n── summer variants (DOP20 → 7.5cm resampled) ──")
        # All splits: train needs it for domain-bridging augmentation, valid/test need it
        # to evaluate whether the model still works on lower-resolution (20cm) summer imagery.
        for split, area_list in splits.items():
            print(f"  ── {split} ──")
            for item in area_list:
                area = item["area"]
                if area.endswith("_summer"):
                    continue
                process_area_summer(area, dop20_dir, dop_dir, ndom_dir, out_stacked, out_gt, args.overwrite)

    print("\nDone.")
    print(f"  Stacked TIFs : {out_stacked}")
    print(f"  GT TIFs      : {out_gt}")


if __name__ == "__main__":
    main()
