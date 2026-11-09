#!/usr/bin/env python3
"""Convert the synthetic XYZ demo into paired LAS and SSS inputs for AIG-OSR.

The image is a constructed intensity pattern and carries no acoustic calibration.
It demonstrates the method input contract, not scientific performance.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import cv2
import laspy
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "sample_data")
    args = ap.parse_args()
    data = args.data
    sparse = np.loadtxt(data / "reef_sparse_10pct.xyz", dtype=np.float64)
    gt = np.loadtxt(data / "reef_gt.xyz", dtype=np.float64)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.creation_date = date(2020, 1, 1)  # Stable synthetic-file hash across reruns.
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = sparse[:, 0], sparse[:, 1], sparse[:, 2]
    las_path = data / "reef_sparse_10pct.las"
    las.write(las_path)

    xy = np.unique(gt[:, 0])
    n = len(xy)
    if len(gt) != n * n:
        raise ValueError("The synthetic full reference must be a square grid")
    z = gt[:, 2].reshape(n, n)
    z_big = cv2.resize(z.astype(np.float32), (96, 96), interpolation=cv2.INTER_CUBIC)
    xpix = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
    yy, xx = np.meshgrid(xpix, xpix, indexing="ij")
    highlight = np.exp(-((xx / 0.35) ** 2 + (yy / 0.44) ** 2))
    shadow = np.exp(-(((xx - 0.44) / 0.22) ** 2 + (yy / 0.40) ** 2))
    intensity = 0.32 + 0.20 * (z_big - z_big.min()) / (np.ptp(z_big) + 1e-8) + 0.38 * highlight - 0.28 * shadow
    sonar = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
    sonar_path = data / "reef_sss_synthetic.png"
    if not cv2.imwrite(str(sonar_path), sonar):
        raise OSError(f"Could not write {sonar_path}")
    print(f"method inputs: {las_path}, {sonar_path}")


if __name__ == "__main__":
    main()
