#!/usr/bin/env python
"""Build the 512x384 metric-space target video from gt.raw.

target_512.raw = round(A(gt)) as uint8 RGB frames (N,384,512,3), where A is the
exact metric downsample: F.interpolate(float, (384,512), bilinear, align_corners=False).
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
H, W = 874, 1164
h, w = 384, 512

def main():
    src = HERE / 'gt.raw'
    n = src.stat().st_size // (H * W * 3)
    mm = np.memmap(src, dtype=np.uint8, mode='r', shape=(n, H, W, 3))
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    maxfrac = 0.0
    for i in range(0, n, 50):
        x = torch.from_numpy(np.ascontiguousarray(mm[i:i+50])).permute(0, 3, 1, 2).float()
        y = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        r = y.round().clamp(0, 255)
        out[i:i+50] = r.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    out.tofile(HERE / 'target_512.raw')
    print(f"wrote target_512.raw: {n} frames, {(HERE/'target_512.raw').stat().st_size:,} bytes")

if __name__ == '__main__':
    main()
