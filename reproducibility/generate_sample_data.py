#!/usr/bin/env python3
"""Generate a small, non-sensitive dataset for the GitHub demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path(__file__).resolve().parent.parent / "sample_data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n = 32
    xy = np.linspace(0.0, 3.1, n)
    yy, xx = np.meshgrid(xy, xy, indexing="ij")
    seabed = 0.10 * np.sin(xx * 1.6) + 0.08 * np.cos(yy * 1.1)
    reef = 0.55 * np.exp(-(((xx - 1.65) / 0.55) ** 2 + ((yy - 1.60) / 0.70) ** 2))
    z = seabed + reef
    gt = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))

    # Select exactly 10% (rounded) from swath-like candidates; keep the four
    # corners so the sparse input and synthetic image share the same footprint.
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    candidates = (((2 * rows + cols) % 10 == 0) | (cols % 11 == 0)).ravel()
    corners = np.array([0, n - 1, n * (n - 1), n * n - 1])
    pool = np.setdiff1d(np.flatnonzero(candidates), corners)
    chosen = np.r_[corners, rng.choice(pool, size=round(0.10 * n * n) - len(corners), replace=False)]
    keep = np.zeros(n * n, dtype=bool)
    keep[chosen] = True
    keep = keep.reshape(n, n)
    sparse = np.column_stack((xx[keep], yy[keep], z[keep]))
    sparse[:, 2] += rng.normal(0.0, 0.005, len(sparse))

    np.savetxt(out / "reef_gt.xyz", gt, fmt="%.6f %.6f %.6f", header="x y z")
    np.savetxt(out / "reef_sparse_10pct.xyz", sparse, fmt="%.6f %.6f %.6f", header="x y z")
    metadata = {
        "name": "synthetic_reef_demo",
        "purpose": "I/O and metric-protocol demonstration only; not a paper result",
        "coordinate_system": "local Cartesian metres",
        "ground_truth_file": "reef_gt.xyz",
        "sparse_input_file": "reef_sparse_10pct.xyz",
        "n_ground_truth": int(len(gt)),
        "n_sparse": int(len(sparse)),
        "retained_fraction": float(len(sparse) / len(gt)),
        "seed": args.seed,
        "columns": ["x", "y", "z"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(gt)} GT points and {len(sparse)} sparse points to {out}")


if __name__ == "__main__":
    main()
