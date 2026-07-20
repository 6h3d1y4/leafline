#!/usr/bin/env python3
"""Minimal GPU allocation test."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
device = torch.device("cuda")

# GPU details
props = torch.cuda.get_device_properties(0)
print(f"GPU      : {props.name}")
print(f"VRAM     : {props.total_memory / 1e9:.1f} GB")
print(f"ROCm     : {torch.version.hip}")

# Test A: CPU → GPU (Kopie)
t = torch.ones(4, 6, 256, 256).to(device)
print(f"[A] CPU→GPU : OK  {tuple(t.shape)}")
del t

# Test B0: leere Allokation (kein Fill-Kernel)
t = torch.empty(4, 6, 256, 256, device=device)
print(f"[B0] empty  : OK  {tuple(t.shape)}")
del t

# Test B: direkt auf GPU (braucht Fill-Kernel)
t = torch.ones(4, 6, 256, 256, device=device)
print(f"[B] direkt  : OK  {tuple(t.shape)}")
del t

# Test C: Convolution (CPU→GPU Tensoren)
import torchvision
m = torchvision.models.resnet18(weights=None).to(device)
x = torch.randn(1, 3, 64, 64).to(device)
y = m(x)
print(f"[C] Conv fwd: OK  output={tuple(y.shape)}")

print("\nAlle Tests bestanden.")
