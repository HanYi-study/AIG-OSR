"""Convert one AIG-OSR metric XYZ DEM to a display-only height-field PLY.

The geometry path is copied from the locally audited figure generator alpha.py.
Mesh smoothing is for visualization and must not be used for DEM accuracy metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, median_filter

def masked_bilateral(z: np.ndarray, sigma_xy: float, sigma_z: float, strength: float) -> np.ndarray:
    if strength <= 0:
        return z.copy()
    valid = np.isfinite(z)
    original = np.where(valid, z, 0.0)
    num = np.zeros_like(original, dtype=np.float64)
    den = np.zeros_like(original, dtype=np.float64)
    radius = max(1, int(np.ceil(2 * sigma_xy)))
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nearby = np.roll(original, (-dr, -dc), axis=(0, 1))
            neighbour_valid = np.roll(valid, (-dr, -dc), axis=(0, 1))
            if dr > 0:
                neighbour_valid[-dr:, :] = False
            elif dr < 0:
                neighbour_valid[:-dr, :] = False
            if dc > 0:
                neighbour_valid[:, -dc:] = False
            elif dc < 0:
                neighbour_valid[:, :-dc] = False
            spatial = np.exp(-0.5 * (dr * dr + dc * dc) / (sigma_xy * sigma_xy))
            range_weight = np.exp(-0.5 * ((nearby - original) / sigma_z) ** 2)
            weight = spatial * range_weight * neighbour_valid
            num += weight * nearby
            den += weight
    filtered = np.divide(num, den, out=original.copy(), where=den > 0)
    result = (1.0 - strength) * original + strength * filtered
    result[~valid] = np.nan
    return result

def remove_isolated_spikes(z: np.ndarray, kernel: int, threshold_m: float,
                           max_change_m: float) -> np.ndarray:
    """Replace conspicuous local outliers without adding cells to the support mask."""
    if kernel <= 1:
        return z.copy()
    valid = np.isfinite(z)
    if not valid.any():
        return z.copy()
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = z[tuple(nearest)]
    local_median = median_filter(filled, size=kernel, mode="nearest")
    result = z.copy()
    spikes = valid & (np.abs(z - local_median) > threshold_m)
    result[spikes] = z[spikes] + np.clip(local_median[spikes] - z[spikes],
                                        -max_change_m, max_change_m)
    return result

def dem_mesh(path: Path, stride: int, sigma_xy: float, sigma_z: float, strength: float,
             median_kernel: int = 1, spike_threshold_m: float = 0.15,
             max_spike_change_m: float = 0.25,
             max_total_displacement_m: Optional[float] = None):
    xyz = np.loadtxt(path, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or not np.isfinite(xyz[:, :3]).all():
        raise ValueError(f"Invalid finite XYZ input: {path}")
    xy_unique = np.unique(xyz[:, :2], axis=0)
    if len(xy_unique) != len(xyz):
        raise ValueError("Metric DEM has repeated XY; it is not a height field")
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    dx = np.diff(np.unique(xyz[:, 0]))
    dy = np.diff(np.unique(xyz[:, 1]))
    step = float(np.median(np.r_[dx[dx > 0], dy[dy > 0]]))
    ix = np.rint((xyz[:, 0] - x0) / step).astype(np.int64)
    iy = np.rint((xyz[:, 1] - y0) / step).astype(np.int64)
    reconstructed = np.column_stack((x0 + ix * step, y0 + iy * step))
    if np.max(np.abs(reconstructed - xyz[:, :2])) > step * 0.1:
        raise ValueError("XYZ positions are not on a regular metric grid")
    z = np.full((int(iy.max()) + 1, int(ix.max()) + 1), np.nan, dtype=np.float64)
    z[iy, ix] = xyz[:, 2]
    sampled = z[::stride, ::stride]
    cleaned = remove_isolated_spikes(sampled, median_kernel, spike_threshold_m,
                                     max_spike_change_m)
    smooth = masked_bilateral(cleaned, sigma_xy, sigma_z, strength)
    if max_total_displacement_m is not None:
        smooth = sampled + np.clip(smooth - sampled, -max_total_displacement_m,
                                   max_total_displacement_m)
    valid = np.isfinite(sampled)
    h, w = sampled.shape
    grid_ids = np.arange(h * w, dtype=np.int64).reshape(h, w)
    quad = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    tl = grid_ids[:-1, :-1][quad]
    tr = grid_ids[:-1, 1:][quad]
    bl = grid_ids[1:, :-1][quad]
    br = grid_ids[1:, 1:][quad]
    faces_old = np.empty((2 * len(tl), 3), dtype=np.int64)
    faces_old[0::2] = np.column_stack((tl, tr, br))
    faces_old[1::2] = np.column_stack((tl, br, bl))
    mapping = np.full(h * w, -1, dtype=np.int64)
    mapping[valid.ravel()] = np.arange(np.count_nonzero(valid))
    faces = mapping[faces_old]
    rows, cols = np.nonzero(valid)
    vertices = np.column_stack((x0 + cols * stride * step,
                                y0 + rows * stride * step, smooth[valid]))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    displacement = np.abs(smooth[valid] - sampled[valid])
    stats = {"input_points": int(len(xyz)), "grid_step_m": step,
             "display_stride": stride, "max_vertical_displacement_m": float(displacement.max()),
             "mean_vertical_displacement_m": float(displacement.mean()),
             "spike_replaced_cells": int(np.count_nonzero(np.isfinite(sampled) & (cleaned != sampled))),
             "note": "Visualization mesh; smoothing changes some elevations. Evaluate the unsmoothed XYZ DEM."}
    return mesh, stats

def colour_mesh(mesh, zmin: float, zmax: float) -> None:
    z = np.asarray(mesh.vertices)[:, 2]
    colour = plt.get_cmap("turbo")(np.clip((z - zmin) / max(zmax - zmin, 1e-8), 0, 1))[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colour)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xyz", required=True, type=Path, help="AIG-OSR DEM_Final_SelectiveDenoise.xyz")
    ap.add_argument("--out", required=True, type=Path, help="Output triangular PLY")
    ap.add_argument("--kind", choices=("reef", "bridge"), required=True)
    ap.add_argument("--grid_stride", type=int, default=2)
    ap.add_argument("--color_min", type=float)
    ap.add_argument("--color_max", type=float)
    args = ap.parse_args()
    if args.grid_stride < 1:
        ap.error("--grid_stride must be >= 1")
    if (args.color_min is None) != (args.color_max is None):
        ap.error("Set both --color_min and --color_max, or neither")
    if args.color_min is not None and args.color_min >= args.color_max:
        ap.error("--color_min must be less than --color_max")
    if args.kind == "bridge":
        mesh, stats = dem_mesh(args.xyz, args.grid_stride, 1.8, 0.55, 0.8,
                               median_kernel=5, spike_threshold_m=0.15,
                               max_spike_change_m=0.25,
                               max_total_displacement_m=0.30)
    else:
        mesh, stats = dem_mesh(args.xyz, args.grid_stride, 1.2, 0.15, 0.35)
    if args.color_min is None:
        elevation = np.asarray(mesh.vertices)[:, 2]
        color_min, color_max = float(elevation.min()), float(elevation.max())
    else:
        color_min, color_max = args.color_min, args.color_max
    colour_mesh(mesh, color_min, color_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.out), mesh, write_ascii=False):
        raise OSError(f"Could not write {args.out}")
    report = {"source": str(args.xyz), "mesh": str(args.out), "kind": args.kind,
              "color_range_m": [color_min, color_max], "vertices": len(mesh.vertices),
              "triangles": len(mesh.triangles), **stats}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
