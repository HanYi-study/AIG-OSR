#!/usr/bin/env python3
"""Evaluate elevation errors after XY alignment.

The manuscript reports elevation errors, so prediction and reference points are
matched in the horizontal plane first.  A 3-D nearest-neighbour distance is
intentionally not used here because it can hide vertical errors by matching a
prediction to a nearby point at a different elevation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_points(path: str) -> np.ndarray:
    """Read LAS (optional laspy dependency), XYZ, or CSV with x,y,z columns."""
    p = Path(path)
    if p.suffix.lower() == ".las":
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Reading .las requires laspy; use .xyz/.csv for the demo.") from exc
        las = laspy.read(path)
        return np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    with p.open("r", newline="") as f:
        first = f.readline()
    delimiter = "," if "," in first else None
    skip = 1 if any(ch.isalpha() for ch in first) else 0
    points = np.genfromtxt(path, delimiter=delimiter, skip_header=skip, comments="#")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{path} must contain at least three columns: x y z")
    points = points[:, :3]
    if not np.isfinite(points).all():
        points = points[np.isfinite(points).all(axis=1)]
    return points


def nearest_xy(pred: np.ndarray, ref: np.ndarray):
    """Return nearest prediction for each reference point in XY only."""
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(pred[:, :2]).query(ref[:, :2], k=1)
        return distances, indices
    except ImportError:  # pragma: no cover - fallback for minimal installs
        delta = ref[:, None, :2] - pred[None, :, :2]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        indices = np.argmin(distances, axis=1)
        return distances[np.arange(len(ref)), indices], indices


def evaluate(pred: np.ndarray, ref: np.ndarray, xy_tolerance: float, inlier_threshold: float):
    xy_distance, pred_index = nearest_xy(pred, ref)
    valid = xy_distance <= xy_tolerance
    dz = pred[pred_index, 2] - ref[:, 2]
    dz_valid = dz[valid]
    if not len(dz_valid):
        raise ValueError("No reference points were matched; increase --xy-tolerance or check coordinates.")
    return {
        "n_reference": int(len(ref)),
        "n_matched": int(len(dz_valid)),
        "coverage_percent": float(100.0 * len(dz_valid) / len(ref)),
        "rmse_m": float(np.sqrt(np.mean(dz_valid ** 2))),
        "mae_m": float(np.mean(np.abs(dz_valid))),
        "bias_m": float(np.mean(dz_valid)),
        "p95_abs_error_m": float(np.percentile(np.abs(dz_valid), 95)),
        "inlier_threshold_m": float(inlier_threshold),
        "inlier_ratio_percent": float(100.0 * np.mean(np.abs(dz_valid) <= inlier_threshold)),
        "xy_tolerance_m": float(xy_tolerance),
        "protocol": "nearest XY match, then elevation-only errors",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="Predicted XYZ/LAS/CSV")
    ap.add_argument("--gt", required=True, help="Reference XYZ/LAS/CSV")
    ap.add_argument("--xy-tolerance", type=float, default=1e-6)
    ap.add_argument("--inlier-threshold", type=float, default=0.10)
    ap.add_argument("--out", help="Optional JSON report path")
    args = ap.parse_args()
    report = evaluate(read_points(args.pred), read_points(args.gt), args.xy_tolerance, args.inlier_threshold)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
