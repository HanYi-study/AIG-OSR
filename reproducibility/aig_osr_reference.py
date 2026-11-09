import os
import time
import argparse
import inspect
import hashlib
import json
import numpy as np
import cv2
import laspy

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ==============================================================================
# Utils
# ==============================================================================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + eps)

def save_xyz(xyz, out_path, fmt="%.6f %.6f %.4f"):
    np.savetxt(out_path, xyz, fmt=fmt)
    print(f"Saved: {out_path}")

def save_grid_as_xyz_world(Z, x0, y0, step, out_path):
    H, W = Z.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    X = x0 + xx * step
    Y = y0 + yy * step
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.astype(np.float64).ravel()])
    valid = ~np.isnan(xyz[:, 2])
    save_xyz(xyz[valid], out_path)

def box_filter_1d(x, k):
    k = int(k)
    if k <= 1: return x.astype(np.float32)
    ker = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode="same").astype(np.float32)

# ==============================================================================
# LAS & Grid Processing
# ==============================================================================
def load_las_points(las_path, z_outlier_sigma=10.0, apply_sor=True):
    las = laspy.read(las_path)
    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    z = pts[:, 2]
    m, s = float(np.mean(z)), float(np.std(z) + 1e-8)
    pts = pts[np.abs(z - m) < z_outlier_sigma * s]

    if apply_sor and len(pts) > 20:
        print("[Preprocessing] Applying SOR to remove spatial floating noise...")
        tree = cKDTree(pts)
        distances, _ = tree.query(pts, k=21)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        dist_mean, dist_std = np.mean(mean_distances), np.std(mean_distances)
        keep_idx = mean_distances < (dist_mean + 2.0 * dist_std)
        pts = pts[keep_idx]
        print(f"[Preprocessing] SOR removed {len(mean_distances) - np.sum(keep_idx)} noise points.")

    return pts

def bbox_xy(points_xyz, pad=0.0):
    x, y = points_xyz[:, 0].astype(np.float64), points_xyz[:, 1].astype(np.float64)
    return float(x.min())-pad, float(x.max())+pad, float(y.min())-pad, float(y.max())+pad

def rasterize_points_to_world_grid(points_xyz, x0, y0, step, H, W, method="max"):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    ix = np.floor((x - x0) / step + 1e-9).astype(np.int32)
    iy = np.floor((y - y0) / step + 1e-9).astype(np.int32)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z = ix[ok], iy[ok], z[ok]

    key = (iy.astype(np.int64) << 32) | ix.astype(np.int64)
    order = np.argsort(key)
    key_s = key[order]
    ix_s, iy_s, z_s = ix[order], iy[order], z[order]

    _, start = np.unique(key_s, return_index=True)
    end = np.r_[start[1:], len(key_s)]

    ix_u, iy_u, z_u, counts = [], [], [], []
    Z = np.full((H, W), np.nan, dtype=np.float32)

    for s, e in zip(start, end):
        ixx, iyy = int(ix_s[s]), int(iy_s[s])
        zz = z_s[s:e]
        if method == "max": val = float(np.max(zz))
        elif method == "mean": val = float(np.mean(zz))
        else: val = float(np.median(zz))
        
        ix_u.append(ixx); iy_u.append(iyy); z_u.append(val); counts.append(int(e - s))
        Z[iyy, ixx] = val

    return (np.array(ix_u, np.int32), np.array(iy_u, np.int32), np.array(z_u, np.float32), np.array(counts, np.int32), Z)

def build_robust_smooth_base(Z_sparse, obj_mask):
    H, W = Z_sparse.shape
    valid_mask = ~np.isnan(Z_sparse)
    if np.sum(valid_mask) < 5: return np.zeros_like(Z_sparse)

    iy_v, ix_v = np.where(valid_mask)
    points = np.column_stack((ix_v, iy_v))
    values = Z_sparse[valid_mask]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    
    Z_linear = griddata(points, values, (xx, yy), method='linear')
    Z_nearest = griddata(points, values, (xx, yy), method='nearest')
    Z_filled = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)
    
    mn = np.min(values)
    z_seabed_only = Z_sparse.copy()
    z_seabed_only[obj_mask > 0.5] = np.nan
    valid_seabed = ~np.isnan(z_seabed_only)
    
    if np.sum(valid_seabed) > 5:
        iy_s, ix_s = np.where(valid_seabed)
        seabed_pts = np.column_stack((ix_s, iy_s))
        seabed_vals = z_seabed_only[valid_seabed]
        Z_sb_lin = griddata(seabed_pts, seabed_vals, (xx, yy), method='linear')
        Z_sb_near = griddata(seabed_pts, seabed_vals, (xx, yy), method='nearest')
        Z_seabed = np.where(np.isnan(Z_sb_lin), Z_sb_near, Z_sb_lin)
        Z_seabed = cv2.GaussianBlur(Z_seabed.astype(np.float32), (15, 15), 0)
    else:
        Z_seabed = np.full((H, W), mn, dtype=np.float32)

    Z_obj = cv2.GaussianBlur(Z_filled.astype(np.float32), (3, 3), 0)
    base = Z_seabed * (1.0 - obj_mask) + Z_obj * obj_mask
    return cv2.GaussianBlur(base.astype(np.float32), (3, 3), 0)

# ==============================================================================
# Sonar Preprocessing & Masking
# ==============================================================================
def sonar_radiometric_normalize(I01_raw):
    I = np.clip(I01_raw.astype(np.float32), 1e-6, 1.0)
    I = np.log(I) - np.median(np.log(I))
    return normalize01(I - np.median(I, axis=0, keepdims=True)).astype(np.float32)

def extract_shadow_mask_quantile(I01, q=0.18, smooth_k=25, min_area=200):
    h, w = I01.shape
    Ix = cv2.Scharr(I01, cv2.CV_32F, 1, 0); Iy = cv2.Scharr(I01, cv2.CV_32F, 0, 1)
    G01 = normalize01(np.sqrt(Ix**2 + Iy**2))
    thr = np.clip(box_filter_1d(np.quantile(I01, q, axis=0).astype(np.float32), smooth_k), 0.0, 1.0)
    shadow = np.zeros((h, w), np.uint8)
    for x in range(w): shadow[:, x] = (I01[:, x] < thr[x]).astype(np.uint8) * 255
    shadow = (shadow & ((G01 < 0.35).astype(np.uint8) * 255)).astype(np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((5, 9), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats((shadow > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(shadow)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: cleaned[labels == i] = 255
    return (cleaned.astype(np.float32) / 255.0), G01.astype(np.float32)

def extract_obj_mask_pure(I01, G01):
    shadow = ((I01 < 0.25) & (G01 < 0.35)).astype(np.uint8) * 255
    bright = (I01 > (np.median(I01)+0.05)).astype(np.uint8) * 255
    cand = cv2.bitwise_or(shadow, bright)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 300: out[labels == i] = 255
    out[shadow > 0] = 255
    return (out.astype(np.float32) / 255.0)

def build_illum_shadow_partition(I01, shadow_mask, obj_mask, sonar_dir="-x"):
    bright_thr = float(np.quantile(I01[obj_mask > 0.5], 0.75)) if np.any(obj_mask > 0.5) else float(np.quantile(I01, 0.75))
    illum = np.clip((I01 - bright_thr) / (1.0 - bright_thr + 1e-8), 0.0, 1.0)
    obj_d = cv2.dilate((obj_mask > 0.5).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)).astype(np.float32) / 255.0
    illum_w = np.clip(illum * obj_d * (1.0 - shadow_mask), 0.0, 1.0)
    shadow_w = np.clip(shadow_mask * obj_d, 0.0, 1.0)
    s = illum_w + shadow_w + 1e-8
    return (illum_w / np.maximum(1.0, s)).astype(np.float32), (shadow_w / np.maximum(1.0, s)).astype(np.float32)

def bilinear_sample(img, x, y):
    H, W = img.shape
    gx = (x / (W - 1 + 1e-12)) * 2 - 1; gy = (y / (H - 1 + 1e-12)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(img.view(1, 1, H, W), grid, mode="bilinear", align_corners=True).view(-1)

# ==============================================================================
# Conditioned periodic neural field
# ==============================================================================
class PeriodicLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = bool(is_first)
        with torch.no_grad():
            b = 1/self.linear.in_features if self.is_first else np.sqrt(6/self.linear.in_features)/self.omega_0
            self.linear.weight.uniform_(-b, b)
    def forward(self, x): return torch.sin(self.omega_0 * self.linear(x))

class ConditionedPeriodicField(nn.Module):
    def __init__(self, hidden=64, omega_0=30.0, cond_dim=6, n_periodic_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([PeriodicLayer(2 if i==0 else hidden, hidden, is_first=(i==0), omega_0=omega_0) for i in range(n_periodic_layers)])
        self.conditioning_projection = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, n_periodic_layers * hidden * 2))
        self.final = nn.Linear(hidden, 1)
    def forward(self, xy, cond):
        scale_shift = self.conditioning_projection(cond).view(-1, 3, 64 * 2)
        h = xy
        for li, layer in enumerate(self.layers):
            h = (1.0 + 0.1 * torch.tanh(scale_shift[:, li, :64])) * layer(h) + 0.1 * torch.tanh(scale_shift[:, li, 64:])
        return self.final(h)

# ==============================================================================
# Edge-aware graph Laplacian and support mask
# ==============================================================================
def build_edge_aware_laplacian_8n(h, w, edge_mask=None):
    """
    Weighted eight-neighbour graph Laplacian with weak cross-mask links.
    """
    n = h * w; rows = []; cols = []; data = []
    def idx(r, c): return r * w + c
    ortho, diagw = 1.0, 0.5
    
    # 将 mask 转为布尔型以加速判断
    obj_bool = (edge_mask > 0.5) if edge_mask is not None else None

    for r in range(h):
        for c in range(w):
            i = idx(r, c); center = 0.0
            is_obj_i = obj_bool[r, c] if obj_bool is not None else False
            
            for dr, dc, ww in [(-1, 0, ortho), (1, 0, ortho), (0, -1, ortho), (0, 1, ortho),
                               (-1, -1, diagw), (-1, 1, diagw), (1, -1, diagw), (1, 1, diagw)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    is_obj_j = obj_bool[rr, cc] if obj_bool is not None else False
                    
                    # Reduce correction propagation across the candidate boundary.
                    if obj_bool is not None and is_obj_i != is_obj_j:
                        weight = 0.001  # 弱连接，允许形成垂直落差
                    else:
                        weight = ww
                        
                    j = idx(rr, cc); rows.append(i); cols.append(j); data.append(weight); center -= weight
            rows.append(i); cols.append(i); data.append(center)
            
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

def hard_dirichlet_projection(z0, ix_c, iy_c, z_c, obj_mask=None, smooth_lambda=0.01, solver="cg", cg_tol=1e-8, cg_maxiter=20000):
    # The global scale cancels from the exact constrained solution.
    h, w = z0.shape; 
    L = build_edge_aware_laplacian_8n(h, w, obj_mask).astype(np.float64) # 启用边缘感知
    A = (smooth_lambda * (L.T @ L)).tocsr(); z0v = z0.reshape(-1).astype(np.float64)
    b = A @ z0v; p_c = (iy_c.astype(np.int64) * w + ix_c.astype(np.int64))
    mask_c = np.zeros(h*w, dtype=bool); mask_c[p_c] = True
    idx_f, idx_c = np.where(~mask_c)[0], np.where(mask_c)[0]
    A_ff, A_fc = A[idx_f, :][:, idx_f].tocsr(), A[idx_f, :][:, idx_c].tocsr()
    b_f = b[idx_f] - (A_fc @ z_c.astype(np.float64))
    if solver == "cg":
        Mpre = spla.LinearOperator(A_ff.shape, matvec=lambda x: (1.0 / (A_ff.diagonal() + 1e-8)) * x, dtype=np.float64)
        tolerance_kw = {"rtol": cg_tol} if "rtol" in inspect.signature(spla.cg).parameters else {"tol": cg_tol}
        x_f, info = spla.cg(A_ff, b_f, x0=z0v[idx_f], maxiter=cg_maxiter, M=Mpre, **tolerance_kw)
        if info != 0 or not np.all(np.isfinite(x_f)):
            raise RuntimeError(f"Graph projection did not converge (CG info={info}); no DEM was exported.")
    else: x_f = spla.spsolve(A_ff, b_f)
    res = z0v.copy(); res[idx_f] = x_f; res[idx_c] = z_c.astype(np.float64)
    return res.reshape(h, w).astype(np.float32)

def verify_hard_constraints(z, ix, iy, z_gt):
    p = (iy.astype(np.int64) * z.shape[1] + ix.astype(np.int64))
    err = z.reshape(-1)[p] - z_gt.astype(np.float32)
    return float(np.max(np.abs(err))), float(np.mean(np.abs(err)))

def adaptive_confidence_smoothing(z_net, mask_gt):
    dist_transform = cv2.distanceTransform(1 - mask_gt.astype(np.uint8), cv2.DIST_L2, 3)
    smooth_weight = np.clip(dist_transform / 5.0, 0.0, 1.0)
    # 缩小高斯核到(3,3)，极力保护网络的锐利边缘预测
    z_smooth = cv2.GaussianBlur(z_net, (3, 3), 0)
    return z_net * (1.0 - smooth_weight) + z_smooth * smooth_weight

def apply_topology_hole_digging(z_grid, mask_gt, max_dist_px):
    dist_transform = cv2.distanceTransform(1 - mask_gt.astype(np.uint8), cv2.DIST_L2, 3)
    valid_footprint = dist_transform <= max_dist_px
    z_holed = z_grid.copy()
    z_holed[~valid_footprint] = np.nan
    return z_holed

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", required=True); ap.add_argument("--sonar", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--grid_step", type=float, default=0.02); ap.add_argument("--bbox_pad", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch_pts", type=int, default=4096)
    ap.add_argument("--w_mbes", type=float, default=20.0); ap.add_argument("--w_smooth", type=float, default=0.003); ap.add_argument("--w_base0", type=float, default=0.03)
    ap.add_argument("--w_illum_phys", type=float, default=3.0); ap.add_argument("--illum_gamma", type=float, default=4.0)
    ap.add_argument("--sonar_dir", choices=["+x", "-x"], default="-x")
    ap.add_argument("--shadow_q", type=float, default=0.18); ap.add_argument("--shadow_thr_smooth_k", type=int, default=25); ap.add_argument("--shadow_min_area", type=int, default=200)
    ap.add_argument("--hole_dist", type=float, default=0.30)
    
    ap.add_argument("--agg", choices=["median", "mean", "max"], default="max"); ap.add_argument("--z_outlier_sigma", type=float, default=10.0)
    # Global squared-Laplacian scale; it cancels from the exact solution.
    ap.add_argument("--hard_smooth_lambda", type=float, default=0.01); ap.add_argument("--hard_solver", choices=["cg", "spsolve"], default="cg")
    ap.add_argument("--hard_cg_tol", type=float, default=1e-8); ap.add_argument("--hard_cg_maxiter", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(); ensure_dir(args.out)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    manifest = {
        "parameters": vars(args),
        "input_sha256": {"las": file_sha256(args.las), "sonar": file_sha256(args.sonar)},
        "code_sha256": file_sha256(__file__),
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__,
                     "laspy": laspy.__version__, "torch": torch.__version__, "scipy": __import__("scipy").__version__},
    }
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    sonar_u8 = cv2.imread(args.sonar, cv2.IMREAD_UNCHANGED)
    sonar_gray = cv2.cvtColor(sonar_u8, cv2.COLOR_BGRA2GRAY) if sonar_u8.ndim == 3 else sonar_u8
    Ih, Iw = sonar_gray.shape

    I01 = sonar_radiometric_normalize(sonar_gray.astype(np.float32) / 255.0)
    shadow_mask, G01 = extract_shadow_mask_quantile(I01, q=args.shadow_q, smooth_k=args.shadow_thr_smooth_k, min_area=args.shadow_min_area)
    
    obj_mask = extract_obj_mask_pure(I01, G01)
    illum_w, shadow_w = build_illum_shadow_partition(I01, shadow_mask, obj_mask, sonar_dir=args.sonar_dir)

    pts = load_las_points(args.las, z_outlier_sigma=args.z_outlier_sigma, apply_sor=True)
    save_xyz(pts, os.path.join(args.out, "Input_Denoised_MBES.xyz"))

    x0, x1, y0, y1 = bbox_xy(pts, pad=args.bbox_pad); step = float(args.grid_step)
    W, H = int(np.ceil((x1 - x0) / step)) + 1, int(np.ceil((y1 - y0) / step)) + 1

    ix_u, iy_u, z_u, counts, Z_sparse = rasterize_points_to_world_grid(pts, x0, y0, step, H, W, method=args.agg)
    
    obj_mask_grid = cv2.resize(obj_mask, (W, H))
    base = build_robust_smooth_base(Z_sparse, obj_mask_grid)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_t = torch.from_numpy(base).to(device)
    I_t = torch.from_numpy(I01).to(device); illum_w_t = torch.from_numpy(illum_w).to(device)
    shadow_w_t = torch.from_numpy(shadow_w).to(device); obj_w_t = torch.from_numpy(obj_mask.astype(np.float32)).to(device)

    Ix = cv2.Scharr(I01, cv2.CV_32F, 1, 0); Iy = cv2.Scharr(I01, cv2.CV_32F, 0, 1); G = np.sqrt(Ix**2 + Iy**2)
    Ix_t = torch.from_numpy((Ix - Ix.mean()) / (Ix.std() + 1e-8)).to(device)
    Iy_t = torch.from_numpy((Iy - Iy.mean()) / (Iy.std() + 1e-8)).to(device)
    G_t = torch.from_numpy((G - G.mean()) / (G.std() + 1e-8)).to(device)

    def world_to_img_px(Xw, Yw): return (Xw - x0) / (x1 - x0 + 1e-12) * (Iw - 1), (Yw - y0) / (y1 - y0 + 1e-12) * (Ih - 1)

    model = ConditionedPeriodicField(hidden=64, omega_0=30.0, cond_dim=6, n_periodic_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ix_u_t = torch.from_numpy(ix_u.astype(np.int64)).to(device); iy_u_t = torch.from_numpy(iy_u.astype(np.int64)).to(device)
    dz_gt = (torch.from_numpy(z_u.astype(np.float32)).to(device) - base_t[iy_u_t, ix_u_t]).view(-1, 1)
    dz_mean, dz_std = dz_gt.mean(), dz_gt.std() + 1e-8; dz_gt_n = (dz_gt - dz_mean) / dz_std

    def cond_from_world_numpy(Xw_np, Yw_np):
        px_np, py_np = world_to_img_px(Xw_np, Yw_np)
        px = torch.from_numpy(px_np.astype(np.float32)).to(device); py = torch.from_numpy(py_np.astype(np.float32)).to(device)
        return torch.stack([bilinear_sample(I_t, px, py), bilinear_sample(Ix_t, px, py), bilinear_sample(Iy_t, px, py),
                            bilinear_sample(G_t, px, py), bilinear_sample(illum_w_t, px, py), bilinear_sample(obj_w_t, px, py)], dim=1), \
               bilinear_sample(illum_w_t, px, py), bilinear_sample(obj_w_t, px, py)

    def illum_physics_loss(z_grid, I01_img, illum_w_img, sonar_dir="-x", gamma=2.5):
        target = torch.clamp(I01_img, 0.0, 1.0) ** gamma
        dzdx = F.pad(z_grid[:, 1:] - z_grid[:, :-1], (0, 1), mode="replicate")
        facing = torch.sigmoid(15.0 * dzdx) if sonar_dir == "-x" else torch.sigmoid(15.0 * (-dzdx))
        return torch.mean(illum_w_img * (facing - target) ** 2)

    t0, n_cons = time.time(), dz_gt_n.shape[0]
    print(">>> Training AIG-OSR elevation field...")

    for step_i in range(args.steps):
        idxb = torch.randint(0, n_cons, (min(args.batch_pts, n_cons),), device=device)
        ixb, iyb, dzb = ix_u_t[idxb], iy_u_t[idxb], dz_gt_n[idxb]

        xy_b = torch.stack([(ixb.float() / (W - 1 + 1e-12)) * 2 - 1, (iyb.float() / (H - 1 + 1e-12)) * 2 - 1], dim=1)
        c_b, _, _ = cond_from_world_numpy(x0 + ixb.detach().cpu().numpy() * step, y0 + iyb.detach().cpu().numpy() * step)
        
        pred_dz = model(xy_b, c_b)
        loss_mbes = torch.mean((pred_dz - dzb) ** 2)

        ixr, iyr = torch.randint(0, W, (4096,), device=device), torch.randint(0, H, (4096,), device=device)
        xyr = torch.stack([(ixr.float() / (W - 1 + 1e-12)) * 2 - 1, (iyr.float() / (H - 1 + 1e-12)) * 2 - 1], dim=1).requires_grad_(True)
        c_r, _, obj_r = cond_from_world_numpy(x0 + ixr.detach().cpu().numpy() * step, y0 + iyr.detach().cpu().numpy() * step)

        dzr_n = model(xyr, c_r)
        g1 = torch.autograd.grad(dzr_n, xyr, torch.ones_like(dzr_n), create_graph=True, retain_graph=True)[0]
        
        grad_mag = torch.sqrt(g1[:, 0]**2 + g1[:, 1]**2 + 1e-8)
        loss_smooth = torch.mean(grad_mag * (1.0 - obj_r) + (g1[:, 0]**2 + g1[:, 1]**2)*obj_r*0.1 )
        
        dynamic_weight = (1.0 - obj_r.view(-1, 1)) * 1.0 + obj_r.view(-1, 1) * 0.05
        loss_base0 = torch.mean(dynamic_weight * (dzr_n ** 2))

        if step_i % 10 == 0:
            with torch.no_grad():
                yy, xx = torch.meshgrid(torch.arange(Ih, device=device), torch.arange(Iw, device=device), indexing="ij")
                Xwi, Ywi = x0 + (xx.float() / (Iw - 1 + 1e-12)) * (x1 - x0), y0 + (yy.float() / (Ih - 1 + 1e-12)) * (y1 - y0)
                ixg, iyg = (Xwi - x0) / (step + 1e-12), (Ywi - y0) / (step + 1e-12)
                xyn = torch.stack([(ixg / (W - 1 + 1e-12)) * 2 - 1, (iyg / (H - 1 + 1e-12)) * 2 - 1], dim=-1).view(-1, 2)
                c_all = torch.stack([I_t.flatten(), Ix_t.flatten(), Iy_t.flatten(), G_t.flatten(), illum_w_t.flatten(), obj_w_t.flatten()], dim=1)
                dz_i_n = model(xyn, c_all).view(Ih, Iw)
                grid = torch.stack([(ixg.flatten() / (W-1+1e-12)) * 2 - 1, (iyg.flatten() / (H-1+1e-12)) * 2 - 1], dim=-1).view(1, -1, 1, 2)
                base_i = F.grid_sample(base_t.view(1, 1, H, W), grid, mode="bilinear", align_corners=True).view(Ih, Iw)
                z_i = base_i + (dz_i_n * dz_std + dz_mean)
            loss_illum_phys = illum_physics_loss(z_i, I_t, illum_w_t, sonar_dir=args.sonar_dir, gamma=args.illum_gamma)
        else: loss_illum_phys = torch.tensor(0.0, device=device)

        loss = args.w_mbes * loss_mbes + args.w_smooth * loss_smooth + args.w_base0 * loss_base0 + args.w_illum_phys * loss_illum_phys
        opt.zero_grad(); loss.backward(); opt.step()

        if step_i % 200 == 0:
            print(f"[Train {step_i}/{args.steps}] loss={loss.item():.4f} mbes={loss_mbes.item():.4f} smooth={loss_smooth.item():.4f}")

    # ==============================================================================
    # Metric DEM output after hard control projection
    # ==============================================================================
    print("\n>>> Output Phase 1: Generating Metric Outputs for GT and Prediction...")
    model.eval()

    with torch.no_grad():
        iyg, ixg = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        xy = torch.stack([(ixg.float() / (W - 1 + 1e-12)) * 2 - 1, (iyg.float() / (H - 1 + 1e-12)) * 2 - 1], dim=-1).view(-1, 2)
        Xw, Yw = x0 + ixg.flatten().cpu().numpy() * step, y0 + iyg.flatten().cpu().numpy() * step
        c, _, _ = cond_from_world_numpy(Xw, Yw)
        z_net = (base_t + (model(xy, c).view(H, W) * dz_std + dz_mean)).detach().cpu().numpy().astype(np.float32)

    mask_gt = np.zeros((H, W), dtype=bool)
    mask_gt[iy_u, ix_u] = True
    z_net_clean = adaptive_confidence_smoothing(z_net, mask_gt)
    
    t_proj = time.time()
    # Project onto MBES controls with image-derived boundary weights.
    z_final = hard_dirichlet_projection(z_net_clean, ix_u, iy_u, z_u, obj_mask=obj_mask_grid, smooth_lambda=args.hard_smooth_lambda, solver=args.hard_solver, cg_tol=args.hard_cg_tol, cg_maxiter=args.hard_cg_maxiter)
    
    hole_px = max(2, int(args.hole_dist / step))
    z_final = apply_topology_hole_digging(z_final, mask_gt, hole_px)
    
    max_err, mean_err = verify_hard_constraints(z_final, ix_u, iy_u, z_u)
    print(f"[Metric Pred] HardProj time={time.time()-t_proj:.2f}s, Constraint max|err|={max_err:.6f}, mean|err|={mean_err:.6f}")
    save_grid_as_xyz_world(z_final, x0, y0, step, os.path.join(args.out, "DEM_Final_SelectiveDenoise.xyz"))

    save_grid_as_xyz_world(Z_sparse, x0, y0, step, os.path.join(args.out, "MBES_Control_Raster.xyz"))
    print("AIG-OSR metric DEM complete.")
