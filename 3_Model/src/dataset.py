"""
6-channel patch dataset for fine-tuning deeptrees on Kiel 7.5 cm data.

Memory-efficient: loads ONE area at a time, yields all patches from it,
then frees memory before loading the next. Peak RAM usage = 1 area ≈ 1 GB.

Expects pre-processed files from prepare_data.py:
  stacked_6ch/{area}.tif   (6 bands: R G B I NDVI nDOM, float32 [0,1])
  gt_bands/{area}_gt.tif   (3 bands: mask outline dist, float32 [0,1])

Channel indices (0-based):
  0 R  1 G  2 B  3 I(NIR)  4 NDVI  5 nDOM

Spring-aware augmentation:
  - NDVI (ch 4) is randomly scaled down (×0.2–1.0) to simulate bare/early foliage
  - NIR  (ch 3) is randomly scaled (×0.7–1.1) — less seasonal swing than NDVI
  - RGB  (ch 0-2) get brightness + contrast jitter to bridge summer→spring appearance
  - nDOM (ch 5) is left unchanged — height is season-independent
  - Spectral Gaussian noise on channels 0-4 (not nDOM)

Small-crown oversampling (oversample_small=True):
  - Small/short GT crowns are underrepresented under pure random patch sampling
    (see debug_predictions.ipynb, Abschnitt 11). A configurable fraction of patches
    per area are instead centered (with jitter) on crowns below small_area_thresh_m2.

Channel selection (channel_indices):
  - Use channel_indices_for(model.in_channels) to drop the nDOM channel (5-channel
    variant) for the with/without-nDOM ablation.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import label as cc_label
from torch.utils.data import IterableDataset


def channel_indices_for(in_channels: int) -> list[int] | None:
    """
    Map model.in_channels (config) to the raster channel subset to feed the model.

    6 = full stack (R G B I NDVI nDOM). 5 = drop nDOM (channel 5) — used for the
    height-ablation variant that measures the nDOM channel's contribution.
    """
    if in_channels == 6:
        return None
    if in_channels == 5:
        return [0, 1, 2, 3, 4]
    raise ValueError(f"Unsupported in_channels: {in_channels} (expected 5 or 6)")


# Default augmentation (all components on) reproduces the original spring-aware
# behaviour. Each component is individually toggleable via the config's `data.augment`
# section so single factors can be isolated — e.g. Schedule Schritt 1 runs 100% spring
# data and therefore switches the summer→spring spectral simulation (nir/ndvi/noise) OFF,
# leaving only geometric flips + brightness/contrast.
DEFAULT_AUGMENT = {
    "hflip": True,
    "vflip": True,
    "brightness": True,     # RGB per-channel brightness jitter (ch 0-2)
    "contrast": True,       # RGB per-channel contrast shift    (ch 0-2)
    "nir_scale": True,      # NIR mild scaling                  (ch 3)
    "ndvi_scale": True,     # NDVI down-scaling, spring sim      (ch 4)
    "spectral_noise": True, # Gaussian noise                    (ch 0-4)
}


def _augment(r_patch: torch.Tensor, cfg: dict) -> torch.Tensor:
    """In-place spectral augmentation; each component gated by `cfg` (see DEFAULT_AUGMENT).

    Robust to 5- or 6-channel patches: NIR/NDVI/noise only touch channels that exist,
    and nDOM (ch 5) is never modified (height is season-independent)."""
    n_ch = r_patch.shape[0]

    # --- RGB brightness jitter (channels 0-2) ---
    if cfg.get("brightness", True):
        brightness = torch.empty(3, 1, 1).uniform_(0.75, 1.25)
        r_patch[:3] = (r_patch[:3] * brightness).clamp(0.0, 1.0)

    # --- RGB contrast shift (channels 0-2) ---
    if cfg.get("contrast", True):
        contrast_shift = torch.empty(3, 1, 1).uniform_(-0.1, 0.1)
        r_patch[:3] = (r_patch[:3] + contrast_shift).clamp(0.0, 1.0)

    # --- NIR mild scaling (channel 3) ---
    if cfg.get("nir_scale", True) and n_ch > 3:
        nir_scale = torch.empty(1).uniform_(0.7, 1.1).item()
        r_patch[3] = r_patch[3].mul(nir_scale).clamp(0.0, 1.0)

    # --- NDVI spring simulation (channel 4): scale down to mimic bare foliage ---
    if cfg.get("ndvi_scale", True) and n_ch > 4:
        ndvi_scale = torch.empty(1).uniform_(0.2, 1.0).item()
        r_patch[4] = r_patch[4].mul(ndvi_scale).clamp(0.0, 1.0)

    # --- Spectral Gaussian noise on channels 0-4 (not nDOM) ---
    if cfg.get("spectral_noise", True):
        k = min(5, n_ch)
        noise = torch.randn(k, r_patch.shape[1], r_patch.shape[2]) * 0.02
        r_patch[:k] = (r_patch[:k] + noise).clamp(0.0, 1.0)

    return r_patch


class KielPatchDataset(IterableDataset):
    """
    Streams (6-ch image patch, 3-ch GT patch) pairs.

    Each epoch: shuffles the area order, loads one area at a time,
    samples patches_per_area patches from it, then frees that area's memory.
    This keeps peak RAM at ~1 area (~1 GB for HoernSued-sized tiles).
    """

    def __init__(
        self,
        areas: list[str],
        stacked_dir: Path,
        gt_dir: Path,
        patch_size: int = 256,
        augment: bool = True,
        channel_indices: list[int] | None = None,
        oversample_small: bool = False,
        small_area_thresh_m2: float = 16.0,
        oversample_frac: float = 0.5,
        augment_config: dict | None = None,
    ):
        """
        channel_indices: subset of the 6 raster channels to yield (see channel_indices_for).
        oversample_small: if True, a fraction of patches per area are centered on GT crowns
            smaller than small_area_thresh_m2 instead of drawn uniformly at random — small/short
            crowns are systematically underrepresented in the raw training footprint otherwise.
        augment_config: per-component augmentation toggles (see DEFAULT_AUGMENT). Merged over
            the defaults, so passing None (or omitting keys) keeps the full spring-aware behaviour.
        """
        self.patch_size = patch_size
        self.augment = augment
        self.augment_cfg = {**DEFAULT_AUGMENT, **(augment_config or {})}
        self.channel_indices = channel_indices
        self.oversample_small = oversample_small
        self.small_area_thresh_m2 = small_area_thresh_m2
        self.oversample_frac = oversample_frac
        self.file_pairs: list[tuple[Path, Path, int, int, list[tuple[int, int]]]] = []

        total_patches = 0
        for area in areas:
            r_path = Path(stacked_dir) / f"{area}.tif"
            g_path = Path(gt_dir) / f"{area}_gt.tif"

            if not r_path.exists():
                raise FileNotFoundError(f"Stacked TIF missing: {r_path}\nRun prepare_data.py first.")
            if not g_path.exists():
                raise FileNotFoundError(f"GT TIF missing: {g_path}\nRun prepare_data.py first.")

            with rasterio.open(r_path) as src:
                H, W = src.height, src.width
                px_area_m2 = abs(src.transform.a * src.transform.e)

            small_centroids: list[tuple[int, int]] = []
            if self.oversample_small:
                with rasterio.open(g_path) as src:
                    gt_mask = src.read(1)
                labeled, n_components = cc_label(gt_mask > 0.5)
                for comp_id in range(1, n_components + 1):
                    ys, xs = np.where(labeled == comp_id)
                    if ys.size and ys.size * px_area_m2 < self.small_area_thresh_m2:
                        small_centroids.append((int(ys.mean()), int(xs.mean())))

            n = max(1, (H // patch_size) * (W // patch_size))
            self.file_pairs.append((r_path, g_path, H, W, small_centroids))
            total_patches += n

        self._len = total_patches

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        ps = self.patch_size

        # Shuffle area order each epoch
        order = np.random.permutation(len(self.file_pairs))
        patches_yielded = 0

        for idx in order:
            if patches_yielded >= self._len:
                break

            r_path, g_path, H, W, small_centroids = self.file_pairs[idx]

            # Load single area into memory
            with rasterio.open(r_path) as src:
                raster = src.read().astype(np.float32)   # (6, H, W)
            if self.channel_indices is not None:
                raster = raster[self.channel_indices]
            with rasterio.open(g_path) as src:
                target = src.read().astype(np.float32)   # (3, H, W)

            patches_this_area = max(1, (H // ps) * (W // ps))

            for _ in range(patches_this_area):
                if patches_yielded >= self._len or H < ps or W < ps:
                    break

                if self.oversample_small and small_centroids and np.random.rand() < self.oversample_frac:
                    cy, cx = small_centroids[np.random.randint(len(small_centroids))]
                    jitter = ps // 4
                    row = int(np.clip(cy - ps // 2 + np.random.randint(-jitter, jitter + 1), 0, H - ps))
                    col = int(np.clip(cx - ps // 2 + np.random.randint(-jitter, jitter + 1), 0, W - ps))
                else:
                    row = np.random.randint(0, H - ps)
                    col = np.random.randint(0, W - ps)

                r_patch = torch.from_numpy(raster[:, row:row + ps, col:col + ps].copy())
                t_patch = torch.from_numpy(target[:, row:row + ps, col:col + ps].copy())

                if self.augment:
                    # Geometric
                    if self.augment_cfg.get("hflip", True) and torch.rand(1).item() > 0.5:
                        r_patch = torch.flip(r_patch, dims=[2])
                        t_patch = torch.flip(t_patch, dims=[2])
                    if self.augment_cfg.get("vflip", True) and torch.rand(1).item() > 0.5:
                        r_patch = torch.flip(r_patch, dims=[1])
                        t_patch = torch.flip(t_patch, dims=[1])
                    # Spectral (per-component toggles in augment_cfg)
                    r_patch = _augment(r_patch, self.augment_cfg)

                patches_yielded += 1
                yield torch.nan_to_num(r_patch), torch.nan_to_num(t_patch)

            # Explicitly free before loading next area
            del raster, target
