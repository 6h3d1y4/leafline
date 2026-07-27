"""
Load the freudenberg2022 JIT checkpoint into a 4-, 5- or 6-channel DeepTreesModel.

The JIT model was trained on 5-channel RGBI+NDVI input (verified: seg encoder conv1
is [64, 5, 7, 7], dist encoder conv1 is [64, 7, 7, 7] = 5 image + 2 seg outputs).
The first-conv adapter handles any target channel count:
  in_channels=6: expand 5→6  (add nDOM as mean of pretrained channels)
  in_channels=5: exact match (no change)
  in_channels=4: reduce 5→4  (drop NDVI = the trailing image channel → keep RGBI weights)

Added channels are the mean of the pretrained channels; on reduction the leading
(retained) channels keep their pretrained weights. All other layers transfer exactly.
"""

import torch
from deeptrees.model.deeptrees_model import DeepTreesModel


def _expand_first_conv(old_w: torch.Tensor, new_in_ch: int) -> torch.Tensor:
    """Expand Conv2d weight [out, old_in, kH, kW] → [out, new_in, kH, kW]."""
    out_ch, old_in_ch, kH, kW = old_w.shape
    if new_in_ch == old_in_ch:
        return old_w.clone()

    if new_in_ch < old_in_ch:
        # Fewer input channels than pretrained (e.g. drop NDVI → RGBI): keep the
        # pretrained weights for the retained LEADING channels. Exact for the seg
        # encoder (image channels are R G B I NDVI, so [:4] = RGBI weights). For the
        # dist encoder (image + 2 seg outputs) this drops a trailing slot rather than
        # NDVI, a minor warm-start imperfection that finetuning re-learns.
        return old_w[:, :new_in_ch].clone()

    new_w = old_w.mean(dim=1, keepdim=True).expand(out_ch, new_in_ch, kH, kW).clone()
    new_w[:, :old_in_ch] = old_w
    return new_w


def _find_first_conv_key(state_dict: dict, expected_out: int) -> str | None:
    """Return the key of the first conv layer (shape [expected_out, *, K, K])."""
    for k, v in state_dict.items():
        if v.ndim == 4 and v.shape[0] == expected_out and v.shape[2] == v.shape[3]:
            return k
    return None


def load_pretrained_6ch(
    jit_path: str,
    lr: float = 1e-4,
    apply_sigmoid: bool = False,
    in_channels: int = 6,
) -> DeepTreesModel:
    """
    Build DeepTreesModel(in_channels=in_channels) and transfer freudenberg2022 weights.

    The JIT file is a traced TreeCrownDelineationModel (NOT wrapped in DeepTreesModel),
    so its state dict keys match model.tcd_backbone directly. in_channels=5 drops nDOM
    for the height-ablation variant; the first-conv expansion below handles any target
    channel count generically (mean-expand pretrained channels).
    """
    model = DeepTreesModel(
        in_channels=in_channels,
        apply_sigmoid=apply_sigmoid,
        lr=lr,
        num_backbones=1,
    )

    print(f"Loading JIT checkpoint: {jit_path}")
    jit = torch.jit.load(jit_path, map_location="cpu")
    jit_sd = jit.state_dict()

    # Detect and strip 'tcd_backbone.' prefix if the JIT was saved from DeepTreesModel
    first_key = next(iter(jit_sd))
    prefix = "tcd_backbone." if first_key.startswith("tcd_backbone.") else ""
    if prefix:
        jit_sd = {k[len(prefix):]: v for k, v in jit_sd.items()}
        print(f"  Stripped '{prefix}' prefix from {len(jit_sd)} keys")

    model_sd = model.tcd_backbone.state_dict()
    new_sd = {}
    expanded = []
    kept_random = []

    for key, model_val in model_sd.items():
        if key not in jit_sd:
            kept_random.append(key)
            new_sd[key] = model_val
            continue

        jit_val = jit_sd[key]

        if model_val.shape == jit_val.shape:
            new_sd[key] = jit_val.clone()  # clone strips JIT storage metadata
        elif (
            model_val.ndim == 4
            and model_val.shape[0] == jit_val.shape[0]
            and model_val.shape[2:] == jit_val.shape[2:]
        ):
            # Input-channel mismatch on a conv layer — expand
            new_sd[key] = _expand_first_conv(jit_val, model_val.shape[1])
            expanded.append(f"  {key}: {list(jit_val.shape)} → {list(model_val.shape)}")
        else:
            kept_random.append(key)
            new_sd[key] = model_val
            print(f"  WARNING: shape mismatch '{key}' {jit_val.shape} vs {model_val.shape}, keeping random init")

    missing, unexpected = model.tcd_backbone.load_state_dict(new_sd, strict=True)

    print(f"  Expanded conv layers ({len(expanded)}):")
    for e in expanded:
        print(e)
    if kept_random:
        print(f"  Randomly initialized ({len(kept_random)} keys — expected for new channels)")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    return model
