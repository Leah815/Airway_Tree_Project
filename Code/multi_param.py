import os, math, time, warnings, json, csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import hashlib
from typing import Tuple, Optional, Sequence, Union, List, Dict, Any

import numpy as np
try:
    import nibabel as nib
except Exception:
    nib = None
try:
    from scipy import ndimage
except Exception:
    ndimage = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"   

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
# -------------------------- Tools --------------------------

def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-12)

def rodrigues_rotate(v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    a = _unit(axis)
    ct, st = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
    return v * ct + torch.cross(a, v, dim=-1) * st + a * torch.sum(a * v, dim=-1, keepdim=True) * (1 - ct)

def orthonormal_basis(n: torch.Tensor):
    n = _unit(n)
    tmp = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
    if torch.abs(n[0]) >= 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    u = _unit(torch.cross(n, tmp, dim=-1))
    w = _unit(torch.cross(n, u, dim=-1))
    return u, w, n

def load_labels_nii(path: str):
    nii = nib.load(path)
    labels = np.asarray(nii.get_fdata(dtype=np.float32)).astype(np.int16)
    A = nii.affine
    spacing = np.linalg.norm(A[:3, :3], axis=0)  # (sx, sy, sz) mm
    shape = labels.shape  # (D, H, W) -> IJK
    return labels, A, spacing, shape

def world_from_ijk(A, ijk):
    return nib.affines.apply_affine(A, ijk)

def torch_inv_affine(A_np: np.ndarray, device, dtype):
    A = torch.tensor(A_np, dtype=dtype, device=device)
    Ainv = torch.linalg.inv(A)
    return A, Ainv

def grid_coords_from_ijk(ijk: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    D, H, W = shape
    i, j, k = ijk.unbind(-1)
    x = 2.0 * (k / max(W - 1, 1)) - 1.0
    y = 2.0 * (j / max(H - 1, 1)) - 1.0
    z = 2.0 * (i / max(D - 1, 1)) - 1.0
    return torch.stack([x, y, z], dim=-1)

def map_to_range(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return torch.as_tensor(mid, device=raw.device, dtype=raw.dtype) + \
           torch.as_tensor(half, device=raw.device, dtype=raw.dtype) * torch.tanh(raw)

def inv_map_from_value(val: float, lo: float, hi: float) -> float:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    x = (val - mid) / max(half, 1e-12)
    x = np.clip(x, -0.9999, 0.9999)
    return float(np.arctanh(x))

def map_softplus(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(raw) + eps

def inv_softplus_from_value(val: float, eps: float = 1e-6) -> float:
    y = max(float(val) - eps, 1e-8)
    if y > 20.0:
        return float(y)
    return float(np.log(np.expm1(y)))

def _stable_name_offset(name: str, mod: int = 1000) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(mod)

def _spawn_ctx():
    return mp.get_context("spawn")

_DEFAULT_PARAM_RANGES = {
    'theta_deg': (25.0, 65.0),
    'r0': (5.70, 10.90),
    'gamma': (2.50, 3.50),
    'ld_ratio': (1.00, 5.00),
    'asymmetry': (0.05, 0.15),
    'root_length_mm': (6.0, 18.0),
}

def _parse_range_arg(s: str) -> Tuple[float, float]:
    xs = [float(x.strip()) for x in str(s).split(',') if x.strip()]
    if len(xs) != 2:
        raise ValueError(f"Range must be 'lo,hi', got {s!r}")
    lo, hi = float(xs[0]), float(xs[1])
    if hi < lo:
        lo, hi = hi, lo
    return (lo, hi)

def _merge_param_ranges(param_ranges: Optional[dict]) -> dict:
    merged = dict(_DEFAULT_PARAM_RANGES)
    if param_ranges:
        for k, v in param_ranges.items():
            if v is None:
                continue
            lo, hi = float(v[0]), float(v[1])
            if hi < lo:
                lo, hi = hi, lo
            merged[k] = (lo, hi)
    return merged


def _full_binary_subtree_size(total_depth: int, level_idx: int) -> int:
    rem = int(total_depth) - int(level_idx) + 1
    if rem <= 0:
        return 0
    return (1 << rem) - 1


def bifurcation_level_stats(
    segments: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    total_depth: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute weak local bifurcation geometry statistics per parent level.

    This uses only the recursive full-binary ordering of the generated tree and compares
    level-wise aggregated branching geometry, avoiding exact branch-wise GT matching.
    """
    stats: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    n = len(segments)
    for idx, (p, q, _r, lvl) in enumerate(segments):
        lv = int(lvl.item()) if torch.is_tensor(lvl) else int(lvl)
        if lv >= int(total_depth):
            continue
        left_idx = idx + 1
        left_subtree = _full_binary_subtree_size(total_depth, lv + 1)
        right_idx = idx + 1 + left_subtree
        if left_idx >= n or right_idx >= n:
            continue
        p1, q1, _r1, _lvl1 = segments[left_idx]
        p2, q2, _r2, _lvl2 = segments[right_idx]
        up = _unit(q - p)
        u1 = _unit(q1 - p1)
        u2 = _unit(q2 - p2)
        cos12 = torch.clamp((u1 * u2).sum(), -1.0 + 1e-6, 1.0 - 1e-6)
        phi1 = torch.rad2deg(torch.arccos(torch.clamp((up * u1).sum(), -1.0 + 1e-6, 1.0 - 1e-6)))
        phi2 = torch.rad2deg(torch.arccos(torch.clamp((up * u2).sum(), -1.0 + 1e-6, 1.0 - 1e-6)))
        theta_cc = torch.rad2deg(torch.arccos(cos12))
        balance = torch.abs(phi1 - phi2)
        ent = stats.setdefault(lv, {'theta_cc': [], 'balance': []})
        ent['theta_cc'].append(theta_cc)
        ent['balance'].append(balance)

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lv, vals in stats.items():
        theta_stack = torch.stack(vals['theta_cc']) if vals['theta_cc'] else None
        bal_stack = torch.stack(vals['balance']) if vals['balance'] else None
        if theta_stack is None or bal_stack is None:
            continue
        out[lv] = {
            'theta_cc_mean': theta_stack.mean(),
            'balance_mean': bal_stack.mean(),
        }
    return out


def bifurcation_loss_from_stats(
    pred_stats: Dict[int, Dict[str, torch.Tensor]],
    target_stats: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    if len(pred_stats) == 0 or len(target_stats) == 0:
        return torch.zeros((), device=device)
    levels = sorted(set(pred_stats.keys()) & set(target_stats.keys()))
    if not levels:
        return torch.zeros((), device=device)
    terms = []
    for lv in levels:
        p = pred_stats[lv]
        t = target_stats[lv]
        term_theta = F.smooth_l1_loss(p['theta_cc_mean'], t['theta_cc_mean'])
        term_balance = F.smooth_l1_loss(p['balance_mean'], t['balance_mean'])
        terms.append(term_theta + 0.25 * term_balance)
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).mean()


def _sync_frozen_params_to_target(init_cfg: dict, target_cfg: dict, trainable_keys: Optional[List[str]], depth: int) -> dict:
    """For parameters not being optimized, force their init/fixed values to the target values."""
    init_cfg = dict(init_cfg)
    trainable = set(trainable_keys or [])
    vec_len = int(depth) - 1
    target_map = {
        'r0': target_cfg['r0'],
        'gamma': list(target_cfg['gamma']),
        'ld_ratio': list(target_cfg['ld_ratio']),
        'theta_deg': list(target_cfg['theta_deg']),
        'asymmetry': list(target_cfg['asym']),
        'roll0_normal': list(target_cfg['roll0_normal']),
        'root_dir': list(target_cfg['root_dir']),
        'root_length_mm': target_cfg['root_length_mm'],
        'orth_angle_deg': target_cfg.get('orth_angle_deg', 90.0),
        'free_target_angle_deg': list(target_cfg.get('free_target_angle_deg', [90.0] * vec_len)),
    }
    for k, v in target_map.items():
        if k not in trainable:
            init_cfg[k] = list(v) if isinstance(v, list) else v
    if 'asymmetry' not in trainable:
        init_cfg['asym'] = list(target_cfg['asym'])
    return init_cfg
def _adapt_segments_from_one_lobe(info, device, dtype):
    segs = []
    for (p, q, r, lvl_id, _lobe_id) in info['segments']:
        pt = torch.tensor(p, device=device, dtype=dtype)
        qt = torch.tensor(q, device=device, dtype=dtype)
        rt = torch.as_tensor(float(r), device=device, dtype=dtype)
        segs.append((pt, qt, rt, int(lvl_id)))
    return segs


def sample_centerline_points_edge_focus(
    segments,
    step_mm: float = 1.5,
    edge_ratio: float = 0.20,
    edge_factor: float = 5.0,
    min_points_per_seg: int = 8,
) -> torch.Tensor:
    if len(segments) == 0:
        return torch.zeros((0, 3), device='cpu')

    pts_all = []
    for (p, q, r, lvl) in segments:
        dev = p.device
        dt  = p.dtype

        L = (q - p).norm()
        n_base = torch.clamp(torch.ceil(L / step_mm).to(torch.int64) + 1, min=min_points_per_seg).item()

        mid_lo = edge_ratio
        mid_hi = 1.0 - edge_ratio
        if mid_hi <= mid_lo:
            t = torch.linspace(0.0, 1.0, int(n_base), device=dev, dtype=dt)
        else:
            n_mid = max(2, int(n_base * (1.0 - 2*edge_ratio)))
            t_mid = torch.linspace(mid_lo, mid_hi, n_mid, device=dev, dtype=dt)
            n_edge = max(2, int(n_base * edge_ratio * edge_factor))
            t_head = torch.linspace(0.0, edge_ratio, n_edge, device=dev, dtype=dt)
            t_tail = torch.linspace(1.0 - edge_ratio, 1.0, n_edge, device=dev, dtype=dt)
            t = torch.cat([t_head, t_mid, t_tail], dim=0)
            t, _ = torch.sort(torch.unique(t))

        t = t.unsqueeze(-1)
        pts_seg = p.unsqueeze(0) * (1.0 - t) + q.unsqueeze(0) * t
        pts_all.append(pts_seg)

    return torch.cat(pts_all, dim=0)


# -------------------------- Render --------------------------

def bbox_from_mask(mask: np.ndarray, pad: int = 4) -> Tuple[slice, slice, slice]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    imin, jmin, kmin = np.maximum(idx.min(0) - pad, 0)
    imax, jmax, kmax = np.minimum(idx.max(0) + pad, np.array(mask.shape) - 1)
    return slice(imin, imax + 1), slice(jmin, jmax + 1), slice(kmin, kmax + 1)

def make_world_grid(A: np.ndarray,
                    roi_ijk: Tuple[slice, slice, slice],
                    downsample: int = 1) -> Tuple[torch.Tensor, np.ndarray]:
    I, J, K = roi_ijk
    ii = np.arange(I.start, I.stop, downsample)
    jj = np.arange(J.start, J.stop, downsample)
    kk = np.arange(K.start, K.stop, downsample)
    IJK = np.stack(np.meshgrid(ii, jj, kk, indexing='ij'), axis=-1).reshape(-1, 3)
    XYZ = world_from_ijk(A, IJK)
    return torch.from_numpy(XYZ).float(), IJK  # (N,3), (N,3) int


def make_world_grid_from_segments(
    segments: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    step_mm: float = 2.0,
    margin_mm: float = 10.0,
    max_points: int = 250_000,
) -> torch.Tensor:
    if segments is None or len(segments) == 0:
        raise ValueError("segments is empty; cannot build world grid.")

    pts = []
    for (p, q, _r, _lvl) in segments:
        pts.append(p.detach().cpu().numpy())
        pts.append(q.detach().cpu().numpy())
    pts = np.stack(pts, axis=0)

    pmin = pts.min(axis=0) - float(margin_mm)
    pmax = pts.max(axis=0) + float(margin_mm)

    step = float(step_mm)
    def _n_points(step_val: float) -> int:
        nx = int(np.ceil((pmax[0] - pmin[0]) / step_val)) + 1
        ny = int(np.ceil((pmax[1] - pmin[1]) / step_val)) + 1
        nz = int(np.ceil((pmax[2] - pmin[2]) / step_val)) + 1
        return nx * ny * nz

    n = _n_points(step)
    while n > max_points:
        step *= 1.25
        n = _n_points(step)

    xs = np.arange(pmin[0], pmax[0] + 1e-6, step, dtype=np.float32)
    ys = np.arange(pmin[1], pmax[1] + 1e-6, step, dtype=np.float32)
    zs = np.arange(pmin[2], pmax[2] + 1e-6, step, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # [N,3]
    return torch.from_numpy(grid).float()

# -------------------------- Bifurcation --------------------------

def _angle_between_normals_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm() + 1e-12); b = b / (b.norm() + 1e-12)
    cosv = torch.clamp((a * b).sum(), -1.0, 1.0)
    return torch.rad2deg(torch.arccos(torch.abs(cosv)))

def _pick_phi_for_free_plane(u: torch.Tensor, w: torch.Tensor, parent_plane_normal: Optional[torch.Tensor],
                             target_deg_lo: float, target_deg_hi: float) -> torch.Tensor:
    if parent_plane_normal is None:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    nphi = 36
    phis = torch.arange(0.0, 2 * math.pi, step=(2 * math.pi) / nphi, device=u.device, dtype=u.dtype)
    best_phi = phis[0]; best_score = 1e9
    for phi in phis:
        a = _unit(torch.cos(phi) * u + torch.sin(phi) * w)
        ang = float(_angle_between_normals_deg(a, parent_plane_normal).item())
        if target_deg_lo <= ang <= target_deg_hi:
            score = abs(ang - 90.0)
        else:
            score = min(abs(ang - target_deg_lo), abs(ang - target_deg_hi))
        if score < best_score:
            best_score, best_phi = score, phi
    return best_phi

def _phi_for_exact_angle(u: torch.Tensor, w: torch.Tensor,
                         parent_plane_normal: torch.Tensor,
                         target_deg: float) -> torch.Tensor:
    n = parent_plane_normal
    alpha = torch.dot(u, n)
    beta = torch.dot(w, n)
    R = torch.sqrt(alpha * alpha + beta * beta) + 1e-12
    c = torch.cos(torch.deg2rad(torch.tensor(float(target_deg), device=u.device, dtype=u.dtype)))
    c = torch.clamp(c, 0.0, 1.0)
    delta = torch.atan2(beta, alpha)
    if c > R:
        return delta
    acos_arg = torch.clamp(c / R, 0.0, 1.0)
    t = torch.arccos(acos_arg)
    cand = [delta + t, delta - t, delta + (math.pi - t), delta - (math.pi - t)]
    best_phi, best_err = cand[0], 1e9
    for phi in cand:
        a = _unit(torch.cos(phi) * u + torch.sin(phi) * w)
        err = torch.abs(torch.abs(torch.dot(a, n)) - c)
        if err < best_err:
            best_err, best_phi = err, phi
    return best_phi

def _normalize_level_ranges(
    plane_angle_deg_range: Union[
        Tuple[float, float],
        Sequence[Union[float, int, Tuple[float, float], Sequence[float]]]
    ]
):
    def _norm_one(rr):
        if isinstance(rr, (int, float)):
            lo = hi = float(rr)
        else:
            rr = list(rr); lo = float(rr[0]); hi = float(rr[1])
        if lo > hi: lo, hi = hi, lo
        return (lo, hi)
    if isinstance(plane_angle_deg_range, (list, tuple)):
        if len(plane_angle_deg_range) == 0:
            return [(70.0, 100.0)]
        if len(plane_angle_deg_range) == 2 and all(isinstance(x, (int, float)) for x in plane_angle_deg_range):
            return [_norm_one(plane_angle_deg_range)]
        return [_norm_one(x) for x in plane_angle_deg_range]
    else:
        return [_norm_one(float(plane_angle_deg_range))]

def generate_segments_plane_policy(
    origin: torch.Tensor, direction: torch.Tensor,
    r0: torch.Tensor, depth: int,
    theta_deg_sched: torch.Tensor, gamma_val: torch.Tensor,
    ld_ratio_sched: torch.Tensor, asym_sched: torch.Tensor,
    roll0_deg: Optional[torch.Tensor] = None,
    roll0_normal: Optional[torch.Tensor] = None, 
    n_free_plane: int = 3,
    plane_angle_deg_range: Union[Tuple[float, float], Sequence[Union[float, int, Tuple[float, float], Sequence[float]]]] = (70.0, 100.0),
    orth_angle_deg: torch.Tensor = torch.tensor(90.0),
    free_phi_deg_sched: Optional[torch.Tensor] = None,
    free_target_angle_deg_sched: Optional[torch.Tensor] = None,
    root_length_mm: Optional[torch.Tensor] = None
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    device = origin.device; dt = origin.dtype
    level_ranges = _normalize_level_ranges(plane_angle_deg_range)

    def range_for_level(lvl_idx: int):
        idx = min(max(lvl_idx, 1) - 1, len(level_ranges) - 1)
        return level_ranges[idx]

    def at_level(arr, lvl):
        if isinstance(arr, torch.Tensor):
            if arr.numel() == 1: return arr
            idx = min(lvl - 1, arr.numel() - 1)
            return arr[idx]
        elif isinstance(arr, (list, tuple)):
            idx = min(lvl - 1, len(arr) - 1)
            return torch.tensor(arr[idx], dtype=dt, device=device)
        else:
            return torch.tensor(float(arr), dtype=dt, device=device)

    def _extend_sched(x: Optional[torch.Tensor], depth: int):
        if x is None: return None
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=dt, device=device)
        if x.ndim == 0: x = x.repeat(depth)
        if x.numel() < depth:
            x = torch.cat([x, x[-1].repeat(depth - x.numel())], dim=0)
        return x

    free_phi_deg_sched  = _extend_sched(free_phi_deg_sched,  depth)
    free_tgt_deg_sched  = _extend_sched(free_target_angle_deg_sched, depth)
    orth_angle_rad = torch.deg2rad(orth_angle_deg)

    segs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def add_branch(p, d, r, level, lvl_idx, parent_plane_normal: Optional[torch.Tensor], parent_roll_phi: Optional[torch.Tensor]):
        nonlocal segs
        if level <= 0: return

        if lvl_idx == 1 and (root_length_mm is not None):
            L = root_length_mm
        else:
            L = at_level(ld_ratio_sched, lvl_idx) * (2.0 * r)
        q = p + d * L
        segs.append((p, q, r, torch.tensor(lvl_idx, device=device, dtype=torch.int32)))
        if level == 1: return

        gamma_l = at_level(gamma_val, lvl_idx)
        asym_l  = torch.clamp(at_level(asym_sched,  lvl_idx), 0.0, 0.5)
        w_share = 0.5 + asym_l
        r_big   = (w_share * (r ** gamma_l)) ** (1.0 / gamma_l)
        r_small = ((1.0 - w_share) * (r ** gamma_l)) ** (1.0 / gamma_l)
        u, wv, _ = orthonormal_basis(d)
        theta = torch.deg2rad(at_level(theta_deg_sched, lvl_idx))

        if lvl_idx <= n_free_plane:
            if parent_plane_normal is None:
                if roll0_normal is not None:
                    a0 = _unit(roll0_normal.to(device=d.device, dtype=d.dtype))
                    a0 = _unit(a0 - (a0 * d).sum() * d)
                    a = a0
                    alpha = torch.dot(a, u)
                    beta  = torch.dot(a, wv)
                    phi = torch.atan2(beta, alpha)
                else:
                    if roll0_deg is None:
                        raise ValueError("Either roll0_normal or roll0_deg must be provided for level 1.")
                    phi = torch.deg2rad(roll0_deg)
                    a = _unit(torch.cos(phi) * u + torch.sin(phi) * wv)
            else:
                if free_phi_deg_sched is not None:
                    phi = torch.deg2rad(free_phi_deg_sched[lvl_idx - 1])
                    a = _unit(torch.cos(phi) * u + torch.sin(phi) * wv)
                elif free_target_angle_deg_sched is not None:
                    target_deg = float(free_target_angle_deg_sched[lvl_idx - 1].item())
                    phi = _phi_for_exact_angle(u, wv, parent_plane_normal, target_deg)
                    a = _unit(torch.cos(phi) * u + torch.sin(phi) * wv)
                else:
                    lo, hi = range_for_level(lvl_idx)
                    phi = _pick_phi_for_free_plane(u, wv, parent_plane_normal, lo, hi)
                    a = _unit(torch.cos(phi) * u + torch.sin(phi) * wv)

            next_parent_plane_normal = a.clone()
            next_roll = phi
        else:
            base = (parent_roll_phi if parent_roll_phi is not None
                    else (torch.deg2rad(roll0_deg) if roll0_deg is not None else 0.0)) + orth_angle_rad
            a = _unit(torch.cos(base) * u + torch.sin(base) * wv)
            next_parent_plane_normal = a.clone()
            next_roll = base

        d_big = _unit(rodrigues_rotate(d, a, +theta))
        d_sml = _unit(rodrigues_rotate(d, a, -theta))
        add_branch(q, d_big, r_big, level - 1, lvl_idx + 1, next_parent_plane_normal, next_roll)
        add_branch(q, d_sml, r_small, level - 1, lvl_idx + 1, next_parent_plane_normal, next_roll)

    add_branch(origin, _unit(direction), r0, depth, 1, None, None)
    return segs

# -------------------------- Soft SDF --------------------------

def signed_distance_to_segment(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab, ap = b - a, points - a
    denom = (ab * ab).sum(dim=-1).clamp_min(1e-12)
    t = torch.clamp((ap * ab).sum(dim=-1) / denom, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return (points - closest).norm(dim=-1)  # >=0

def cylinder_sdf(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return signed_distance_to_segment(points, a, b) - r

def smin_logsumexp(sd_list: List[torch.Tensor], tau: float) -> torch.Tensor:
    if len(sd_list) == 1:
        return sd_list[0]
    xs = torch.stack(sd_list, dim=0)
    xs = torch.clamp(xs, -1e6, 1e6)
    m = torch.min(xs, dim=0, keepdim=True)[0]
    z = torch.exp(torch.clamp(-(xs - m) / tau, max=50.0))
    s = torch.sum(z, dim=0)
    return -(m.squeeze(0) + tau * torch.log(s + 1e-8))

def render_soft_mask(points_world: torch.Tensor,
                     segments: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                     smin_tau_mm: float = 1.0,
                     occ_eps_mm: float = 0.8,
                     batch_seg: int = 256) -> torch.Tensor:
    device = points_world.device
    N = points_world.shape[0]
    pred_occ = torch.zeros((N,), device=device)

    sdist = torch.full((N,), float('inf'), device=device)
    sds_parts: List[torch.Tensor] = []
    for i in range(0, len(segments), batch_seg):
        chunk = segments[i:i + batch_seg]
        sd_list = []
        for (pa, pb, rr, _) in chunk:
            pa = pa.reshape(1, 3).to(device); pb = pb.reshape(1, 3).to(device)
            rr = torch.as_tensor(rr, device=device).reshape(())
            sd = cylinder_sdf(points_world, pa, pb, rr)
            sd_list.append(sd)
        if len(sd_list) > 0:
            sds_parts.append(smin_logsumexp(sd_list, smin_tau_mm))

    if len(sds_parts) == 0:
        return pred_occ

    sdist = smin_logsumexp(sds_parts, smin_tau_mm)
    sdist = torch.clamp(sdist, -1e6, 1e6)
    occ = torch.sigmoid(torch.clamp(-sdist / max(occ_eps_mm, 1e-6), -50.0, 50.0))
    return occ
def sample_centerline_points(segments, step_mm: float = 2.0) -> torch.Tensor:
    pts = []
    for (p, q, r, lvl) in segments:
        L = (q - p).norm()
        n = torch.clamp(torch.ceil(L / step_mm).to(torch.int64) + 1, min=2)
        t = torch.linspace(0.0, 1.0, int(n), device=p.device, dtype=p.dtype).unsqueeze(-1)
        pts_seg = p.unsqueeze(0) * (1.0 - t) + q.unsqueeze(0) * t
        pts.append(pts_seg)
    if len(pts) == 0:
        return torch.zeros((0, 3), device=segments[0][0].device)
    return torch.cat(pts, dim=0)


def chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((), device=a.device if a.numel() else b.device)
    D = torch.cdist(a, b)
    d_ab = D.min(dim=1).values.pow(2).mean()
    d_ba = D.min(dim=0).values.pow(2).mean()
    return (d_ab + d_ba) * 0.5
# -------------------------- Loss --------------------------

def soft_dice(p: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (p * t).sum()
    den = p.sum() + t.sum()
    return 1.0 - (2.0 * inter + eps) / (den + eps)

def tversky(p: torch.Tensor, t: torch.Tensor, alpha=0.3, beta=0.7, eps=1e-6) -> torch.Tensor:
    TP = (p * t).sum()
    FP = (p * (1.0 - t)).sum()
    FN = ((1.0 - p) * t).sum()
    T = TP / (TP + alpha * FP + beta * FN + eps)
    return 1.0 - T



class ParamPack(nn.Module):
    def __init__(self, init_cfg, device, depth, n_head, learn_orth_angle: bool = False, learn_gamma: bool=True, learn_r0: bool=True, learn_root: bool=True, learn_root_dir: bool=True, fixed_plane_angle_deg: float = 90.0, param_ranges: Optional[dict] = None):
        super().__init__()
        self.depth = int(depth)
        if self.depth < 2:
            raise ValueError('depth must be >= 2 for vector-valued bifurcation parameters.')
        self.vec_len = self.depth - 1
        self.n_head = min(int(n_head), self.vec_len)
        self.learn_orth_angle = bool(learn_orth_angle)
        self.learn_gamma = bool(learn_gamma)
        self.learn_r0 = bool(learn_r0)
        self.learn_root = bool(learn_root)
        self.learn_root_dir = bool(learn_root_dir)
        self.fixed_plane_angle_deg = float(fixed_plane_angle_deg)
        self.param_ranges = _merge_param_ranges(param_ranges)
        theta_lo, theta_hi = self.param_ranges['theta_deg']
        r0_lo, r0_hi = self.param_ranges['r0']
        gamma_lo, gamma_hi = self.param_ranges['gamma']
        ld_lo, ld_hi = self.param_ranges['ld_ratio']
        asym_lo, asym_hi = self.param_ranges['asymmetry']
        root_lo, root_hi = self.param_ranges['root_length_mm']
        def to_raw_list(key, n):
            vals = torch.tensor(init_cfg[key], dtype=torch.float32, device=device)
            if vals.ndim == 0:
                vals = vals.repeat(n)
            vals = vals.reshape(-1)
            if vals.numel() < n:
                vals = torch.cat([vals, vals[-1].repeat(n - vals.numel())], dim=0)
            elif vals.numel() > n:
                vals = vals[:n]
            return nn.Parameter(vals.clone())

        theta_init = torch.tensor(init_cfg.get('theta_deg', [36.0] * self.n_head), dtype=torch.float32, device=device)
        if theta_init.ndim == 0:
            theta_init = theta_init.repeat(self.n_head)
        theta_init = theta_init.reshape(-1)
        if theta_init.numel() < self.n_head:
            theta_init = torch.cat([theta_init, theta_init[-1].repeat(self.n_head - theta_init.numel())], dim=0)
        elif theta_init.numel() > self.n_head:
            theta_init = theta_init[:self.n_head]
        theta_raw_list = [inv_map_from_value(float(v), theta_lo, theta_hi) for v in theta_init.tolist()]
        self.theta_raw = nn.Parameter(torch.tensor(theta_raw_list, dtype=torch.float32, device=device))
        self.roll0_deg = nn.Parameter(
            torch.tensor(init_cfg.get('roll0_deg', 0.0), dtype=torch.float32, device=device)
        )

        roll0_n_init = torch.tensor(
            init_cfg.get('roll0_normal', [0.0, 1.0, 0.0]),
            dtype=torch.float32, device=device
        )
        self.roll0_normal = nn.Parameter(roll0_n_init.clone())
        if self.learn_root_dir:
            root_dir_init = torch.tensor(init_cfg.get('root_dir', [0.0,0.0,-1.0]), dtype=torch.float32, device=device)
            if root_dir_init.numel() != 3:
                raise ValueError("init_cfg['root_dir'] must be length-3")
            self.root_dir_raw = nn.Parameter(root_dir_init.clone())
        else:
            root_dir_val = torch.tensor(init_cfg.get('root_dir', [0.0,0.0,-1.0]), dtype=torch.float32, device=device)
            self.register_buffer('root_dir_const', root_dir_val)
        
        if self.learn_r0:
            r0_init_raw = inv_map_from_value(init_cfg.get('r0', 7.5), r0_lo, r0_hi)
            self.r0_raw = nn.Parameter(torch.tensor(r0_init_raw, dtype=torch.float32, device=device))
        else:
            r0_val = init_cfg.get('r0', 7.5)
            self.register_buffer(
                "r0_const",
                torch.tensor(float(r0_val), dtype=torch.float32, device=device)
            )

        if self.learn_gamma:
            gamma_init = torch.tensor(
                init_cfg.get('gamma', 3.0),
                dtype=torch.float32, device=device
            )
            if gamma_init.ndim == 0:
                gamma_init = gamma_init.repeat(self.n_head)
            gamma_init = gamma_init.reshape(-1)
            if gamma_init.numel() < self.n_head:
                gamma_init = torch.cat([gamma_init, gamma_init[-1].repeat(self.n_head - gamma_init.numel())], dim=0)
            elif gamma_init.numel() > self.n_head:
                gamma_init = gamma_init[:self.n_head]
            gamma_raw_list = []
            for g in gamma_init.tolist():
                gamma_raw_list.append(inv_softplus_from_value(float(g)))
            self.gamma_raw = nn.Parameter(
                torch.tensor(gamma_raw_list, dtype=torch.float32, device=device)
            )
        else:
            gamma_val = init_cfg.get('gamma', 3.0)
            if isinstance(gamma_val, (int, float)):
                gamma_list = [float(gamma_val)] * self.n_head
            elif isinstance(gamma_val, (list, tuple)):
                gamma_list = list(gamma_val)
                if len(gamma_list) < self.n_head:
                    gamma_list = gamma_list + [gamma_list[-1]] * (self.n_head - len(gamma_list))
                gamma_list = gamma_list[:self.n_head]
            else:
                raise ValueError("init_cfg['gamma'] must be a number or a list")        
            gamma_tensor = torch.tensor(gamma_list, dtype=torch.float32, device=device)
            self.register_buffer("gamma_const", gamma_tensor)

        if self.learn_root:
            root_init_raw = inv_softplus_from_value(init_cfg.get('root_length_mm', 9.5))
            self.root_raw = nn.Parameter(torch.tensor(root_init_raw, dtype=torch.float32, device=device))
        else:
            root_val = init_cfg.get('root_length_mm', 9.5)
            self.register_buffer(
                "root_const",
                torch.tensor(float(root_val), dtype=torch.float32, device=device)
            )
        ld_init = torch.tensor(
            init_cfg.get('ld_ratio', 3.0),
            dtype=torch.float32, device=device,
        )
        if ld_init.ndim == 0:
            ld_init = ld_init.repeat(self.n_head)
        ld_init = ld_init.reshape(-1)
        if ld_init.numel() < self.n_head:
            ld_init = torch.cat([ld_init, ld_init[-1].repeat(self.n_head - ld_init.numel())], dim=0)
        elif ld_init.numel() > self.n_head:
            ld_init = ld_init[:self.n_head]

        ld_raw_list = []
        for v in ld_init.tolist():
            ld_raw_list.append(inv_map_from_value(float(v), ld_lo, ld_hi))

        self.ld_raw = nn.Parameter(
            torch.tensor(ld_raw_list, dtype=torch.float32, device=device)
        )

        asym_init = torch.tensor(init_cfg.get('asym', [0.08] * self.n_head), dtype=torch.float32, device=device)
        if asym_init.ndim == 0:
            asym_init = asym_init.repeat(self.n_head)
        asym_init = asym_init.reshape(-1)
        if asym_init.numel() < self.n_head:
            asym_init = torch.cat([asym_init, asym_init[-1].repeat(self.n_head - asym_init.numel())], dim=0)
        elif asym_init.numel() > self.n_head:
            asym_init = asym_init[:self.n_head]
        asym_raw_list = [inv_map_from_value(float(v), asym_lo, asym_hi) for v in asym_init.tolist()]
        self.asym = nn.Parameter(torch.tensor(asym_raw_list, dtype=torch.float32, device=device))
        fixed_ang = torch.full((self.n_head,), self.fixed_plane_angle_deg, dtype=torch.float32, device=device)
        self.register_buffer('free_target_angle_deg_const', fixed_ang)
        init_orth = init_cfg.get('orth_angle_deg', 90.0)
        if self.learn_orth_angle:
            self.orth_angle_raw = nn.Parameter(torch.tensor(init_orth, dtype=torch.float32, device=device))
        else:
            self.register_buffer('orth_angle_const', torch.tensor(float(init_orth), dtype=torch.float32, device=device))

    def forward(self):
        def extend(p, total):
            if p.numel() < total:
                pad = p[-1].repeat(total - p.numel())
                p = torch.cat([p, pad], dim=0)
            return p

        theta_raw = extend(self.theta_raw, self.vec_len)
        theta_lo, theta_hi = self.param_ranges['theta_deg']
        r0_lo, r0_hi = self.param_ranges['r0']
        ld_lo, ld_hi = self.param_ranges['ld_ratio']
        asym_lo, asym_hi = self.param_ranges['asymmetry']
        theta = map_to_range(theta_raw, theta_lo, theta_hi)
        asym  = extend(self.asym,      self.vec_len)
        ld_r  = extend(self.ld_raw,    self.vec_len)

        if self.learn_r0:
            r0 = map_to_range(self.r0_raw, r0_lo, r0_hi)
        else:
            r0 = self.r0_const
 
        if self.learn_gamma:
            gamma = map_softplus(self.gamma_raw)
        else:
            gamma = self.gamma_const

        gamma = extend(gamma, self.vec_len)

        head_levels = min(self.n_head, self.vec_len)
        if self.vec_len > head_levels:
            tail_val = gamma[head_levels - 1]
            gamma[head_levels:] = tail_val

        ld = map_to_range(ld_r, ld_lo, ld_hi)

        if self.learn_root:
            rootL = map_softplus(self.root_raw)
        else:
            rootL = self.root_const
        
        asym = map_to_range(asym, asym_lo, asym_hi)
        free_tgt = extend(self.free_target_angle_deg_const, self.vec_len)
        if self.learn_orth_angle:
            orth_angle_deg = 70.0 + 40.0 * torch.sigmoid((self.orth_angle_raw - 70.0) / 10.0)
        else:
            orth_angle_deg = self.orth_angle_const
        roll0_normal = _unit(self.roll0_normal)
        if self.learn_root_dir:
            root_dir = _unit(self.root_dir_raw)
        else:
            root_dir = _unit(self.root_dir_const)

        return {
            'r0': r0,
            'depth': self.depth,
            'theta_deg': theta,
            'gamma': gamma,
            'ld_ratio': ld,
            'asymmetry': asym,
            'roll0_deg': self.roll0_deg,
            'roll0_normal': roll0_normal,
            'root_dir': root_dir,
            'root_length_mm': rootL,
            'orth_angle_deg': orth_angle_deg,
            'free_target_angle_deg': free_tgt,
        }

# -------------------------- Visualization --------------------------

def lines_from_segments(segs, color, name):
    xs, ys, zs = [], [], []
    for (p, q, r, lvl) in segs:
        p = p.detach().cpu().numpy(); q = q.detach().cpu().numpy()
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                        line=dict(width=6, color=color), name=name, opacity=1.0)

def mesh_from_label_world(labels, A, label_id):
    vol = (labels == label_id).astype(np.uint8)
    if vol.sum() == 0:
        return None
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5)
    verts_world = world_from_ijk(A, verts[:, [0, 1, 2]])
    mesh = go.Mesh3d(x=verts_world[:, 0], y=verts_world[:, 1], z=verts_world[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     opacity=0.25, color='lightgray', name=f'Label {label_id}')
    return mesh

def print_params(pars: dict, title: str = ""):
    def _to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    if title: print(title)
    for k, v in pars.items():
        if torch.is_tensor(v):
            arr = _to_np(v)
            if arr.ndim == 0:
                print(f"  {k}: {float(arr):.6f}")
            else:
                np.set_printoptions(precision=6, suppress=True)
                print(f"  {k}: {arr}")
        else:
            print(f"  {k}: {v}")

# -------------------------- Main Flow --------------------------
def demo_fit_softsdf(
    device: Optional[torch.device] = None,
    depth: int = 6,
    n_head: int = 5,
    n_free_plane: int = 3,
    plane_angle_deg_range=((80, 100), (72, 98), (80, 100)),
    iters_adam_stage1: int = 200,
    iters_adam_stage2: int = 200,
    learn_orth_angle: bool = False,
    learn_gamma: bool = True,
    learn_r0: bool = True,
    learn_root: bool = True,
    learn_root_dir: bool = False,
    fixed_plane_angle_deg: float = 90.0,
    exp_name: str = "fit_target_geom",
    init_cfg: Optional[dict] = None,
    
    target_cfg_override: Optional[dict] = None,
    trainable_keys: Optional[List[str]] = None,
    return_result: bool = True,
    airway_mask_path: Optional[str] = None,
    save_target_html: bool = True,
    param_ranges: Optional[dict] = None,
    if_bif: bool = True,
    snapshot_interval: int = 60,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 80)
    print(f"Experiment: {exp_name}")
    print("=" * 80)

    A = np.eye(4, dtype=np.float32)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    shape = (1, 1, 1)

    origin_fixed = torch.tensor([-143, -185, 201], device=device, dtype=torch.float32)
    direction_fixed = _unit(torch.tensor([1.0, 0.5, 0.5], device=device, dtype=torch.float32))

    _default_target_cfg = dict(
        r0=4.8, depth=depth, n_head=n_head,
        theta_deg=[42, 38, 36, 34],
        gamma=[3.2, 3.2, 3.0, 3.0],
        ld_ratio=[3.0, 3.2, 2.8, 2.9],
        asym=[0.05, 0.08, 0.08, 0.10],
        roll0_normal=[0.0, 1.0, 0.0],
        root_dir=[0.0, 0.0, -1.0],
        root_length_mm=9.5,
        orth_angle_deg=90.0,
    )
    target_cfg = target_cfg_override.copy() if target_cfg_override is not None else _default_target_cfg
    target_free_ang = torch.full((n_head,), 90.0, device=device)

    seg_t = generate_segments_plane_policy(
        origin_fixed, _unit(torch.tensor(target_cfg.get('root_dir', [0.0,0.0,-1.0]), device=device, dtype=torch.float32)),
        r0=torch.tensor(target_cfg['r0'], device=device),
        depth=target_cfg['depth'],
        theta_deg_sched=torch.tensor(target_cfg['theta_deg'], device=device),
        gamma_val=torch.tensor(target_cfg['gamma'], device=device),
        ld_ratio_sched=torch.tensor(target_cfg['ld_ratio'], device=device),
        asym_sched=torch.tensor(target_cfg['asym'], device=device),
        roll0_deg=None,
        roll0_normal=torch.tensor(target_cfg['roll0_normal'], device=device),
        n_free_plane=n_free_plane,
        plane_angle_deg_range=plane_angle_deg_range,
        orth_angle_deg=torch.tensor(target_cfg['orth_angle_deg'], device=device),
        free_phi_deg_sched=None,
        free_target_angle_deg_sched=target_free_ang,
        root_length_mm=torch.tensor(target_cfg['root_length_mm'], device=device),
    )

    target_bif_stats = bifurcation_level_stats(seg_t, depth) if if_bif else {}

    if save_target_html:
        traces = [lines_from_segments(seg_t, color="red", name="Target Airway")]
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{exp_name} | Target airway tree",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(x=0.02, y=0.98)
        )
        out_html = f"{exp_name}_target_tree.html"
        pio.write_html(fig, out_html, auto_open=False)
        print(f"[Saved] Target tree figure -> {out_html}")

    grid_world = make_world_grid_from_segments(seg_t, step_mm=2.0, margin_mm=15.0).to(device)

    O_t = render_soft_mask(
        grid_world, seg_t,
        smin_tau_mm=1.0,
        occ_eps_mm=0.8,
    )

    if airway_mask_path is not None:
        print("[INFO] airway_mask_path is provided but ignored in 4.2/4.3 pure-simulation mode.")

    dt_torch = None
    if init_cfg is None:
        init_cfg = dict(
            r0=4.8,
            theta_deg=[36, 36, 34, 34],
            gamma=(target_cfg["gamma"] if not learn_gamma else [3.0, 3.0, 3.0, 3.0]),
            ld_ratio=[2.8, 2.8, 2.8, 2.8],
            asym=[0.02, 0.05, 0.05, 0.05],
            roll0_normal=target_cfg['roll0_normal'],
            root_dir=[0.0,0.0,-1.0],
            root_length_mm=9.5,
            orth_angle_deg=90.0,
            free_target_angle_deg=[95.0, 92.0, 90.0, 90.0],
        )
    init_cfg = _sync_frozen_params_to_target(init_cfg, target_cfg, trainable_keys, depth)

    effective_trainable_keys = list(trainable_keys) if trainable_keys is not None else [
        'r0', 'gamma', 'ld_ratio', 'theta_deg', 'asymmetry', 'root_length_mm'
    ]
    learn_gamma_flag = bool(learn_gamma) and ('gamma' in effective_trainable_keys)
    learn_r0_flag = bool(learn_r0) and ('r0' in effective_trainable_keys)
    learn_root_flag = bool(learn_root) and ('root_length_mm' in effective_trainable_keys)
    learn_root_dir_flag = bool(learn_root_dir) and ('root_dir' in effective_trainable_keys)

    pack = ParamPack(
        init_cfg, device,
        depth=depth, n_head=n_head,
        learn_orth_angle=learn_orth_angle,
        learn_gamma=learn_gamma_flag,
        learn_r0=learn_r0_flag,
        learn_root=learn_root_flag,
        learn_root_dir=learn_root_dir_flag,
    ).to(device)
    if trainable_keys is not None:
        key2names = {
            'r0': ['r0_raw'],
            'gamma': ['gamma_raw'],
            'ld_ratio': ['ld_raw', 'ld_raw_list', 'ld_raws'],
            'theta_deg': ['theta_raw'],
            'asymmetry': ['asym'],
            'roll0_deg': ['roll0_deg'],
            'roll0_normal': ['roll0_normal'],
            'root_length_mm': ['root_raw'],
            'root_dir': ['root_dir_raw'],
        }
        allow = set()
        for k in trainable_keys:
            if k not in key2names:
                raise ValueError(f"Unknown trainable key: {k}. Allowed: {sorted(key2names.keys())}")
            allow.update(key2names[k])

        for name, p in pack.named_parameters():
            base = name.split('.')[0]
            p.requires_grad = (base in allow)

        tr = [n for n,p in pack.named_parameters() if p.requires_grad]
        print(f"[4.2.1] Trainable params: {tr}")

    def _active_level_caps(depth_: int) -> List[int]:
        if int(depth_) <= 1:
            return [1]
        return list(range(2, int(depth_) + 1))

    def _stage_level_weight(level_cap: Optional[int], max_depth: int) -> float:
        if level_cap is None:
            return 1.0
        rel = max(0, int(level_cap) - 2)
        return float(1.0 + 0.18 * rel)

    level_caps = _active_level_caps(depth)
    stage_cfgs: List[Dict[str, Any]] = []

    progressive_specs = [
        dict(prefix="Stage-A", max_iters=iters_adam_stage1,
             lr_fast=1e-2, lr_slow=5e-3, lr_theta=5e-2,
             smin_tau=1.2, occ_eps=0.75,
             min_iters=60, patience=24, rel_tol=4e-4, cl_rel_tol=4e-4),
        dict(prefix="Stage-B", max_iters=iters_adam_stage2,
             lr_fast=6e-3, lr_slow=3e-3, lr_theta=2.5e-2,
             smin_tau=0.9, occ_eps=0.5,
             min_iters=70, patience=28, rel_tol=3e-4, cl_rel_tol=3e-4),
    ]
    for spec in progressive_specs:
        n_caps = max(1, len(level_caps))
        for i, cap in enumerate(level_caps, start=1):
            stage_iters = max(35, int(round(spec['max_iters'] / n_caps)))
            stage_cfgs.append(dict(
                name=f"{spec['prefix']}-L{cap}",
                max_iters=stage_iters,
                lr_fast=spec['lr_fast'],
                lr_slow=spec['lr_slow'],
                lr_theta=spec['lr_theta'],
                smin_tau=spec['smin_tau'],
                occ_eps=spec['occ_eps'],
                min_iters=min(spec['min_iters'], max(25, int(round(stage_iters * 0.45)))),
                patience=spec['patience'],
                rel_tol=spec['rel_tol'],
                cl_rel_tol=spec['cl_rel_tol'],
                stage_level=cap,
                stage_weight=_stage_level_weight(cap, depth),
                loss_mode="incremental",
            ))

    stage_cfgs.extend([
        dict(name="Stage-C-global-1",
             max_iters=max(50, int(round(iters_adam_stage2 * 0.30))),
             lr_fast=4e-3, lr_slow=2e-3, lr_theta=1.5e-2,
             smin_tau=0.85, occ_eps=0.55,
             min_iters=50, patience=24, rel_tol=2e-4, cl_rel_tol=2e-4,
             stage_level=None, stage_weight=1.0, loss_mode="full"),
        dict(name="Stage-C-global-2",
             max_iters=max(40, int(round(iters_adam_stage2 * 0.20))),
             lr_fast=2e-3, lr_slow=1e-3, lr_theta=8e-3,
             smin_tau=0.80, occ_eps=0.50,
             min_iters=40, patience=20, rel_tol=1.5e-4, cl_rel_tol=1.5e-4,
             stage_level=None, stage_weight=1.0, loss_mode="full"),
    ])

    def build_optimizer(lr_fast, lr_slow, lr_theta):
        grp_fast = [pack.ld_raw, pack.asym, pack.roll0_normal]
        grp_slow = [pack.roll0_deg]
        if hasattr(pack, 'root_dir_raw'):
            grp_slow.insert(0, pack.root_dir_raw)
        grp_theta = [pack.theta_raw]

        if hasattr(pack, "gamma_raw"):
            grp_slow.insert(0, pack.gamma_raw)
        if hasattr(pack, "r0_raw"):
            grp_slow.insert(0, pack.r0_raw)
        if hasattr(pack, "root_raw"):
            grp_slow.insert(0, pack.root_raw)
        if learn_orth_angle and hasattr(pack, "orth_angle_raw"):
            grp_slow.append(pack.orth_angle_raw)

        opt = torch.optim.Adam(
            [
                {'params': grp_fast, 'lr': lr_fast},
                {'params': grp_slow, 'lr': lr_slow},
                {'params': grp_theta, 'lr': lr_theta},
            ],
            betas=(0.9, 0.999), weight_decay=0.0,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=999999, eta_min=lr_slow * 0.1
        )
        return opt, sch

    loss_history: List[Tuple[str, int, float]] = []
    snapshot_interval = max(1, int(snapshot_interval))
    param_trace: List[dict] = []

    def _record_param_snapshot(tag: str, global_step: int, stage_name: str, pars_dict: dict) -> None:
        snap = dict(
            tag=str(tag),
            global_step=int(global_step),
            stage_name=str(stage_name),
        )
        for k, v in _pars_to_row('snapshot', pars_dict).items():
            snap[k] = v
        param_trace.append(snap)

    def _segments_upto_level(seg_list, level_cap: Optional[int]):
        if level_cap is None:
            return seg_list
        cap = int(level_cap)
        return [(p, q, r, lvl) for (p, q, r, lvl) in seg_list if int(lvl.item()) <= cap]

    def _segments_at_level(seg_list, level_exact: Optional[int]):
        if level_exact is None:
            return seg_list
        lev = int(level_exact)
        return [(p, q, r, lvl) for (p, q, r, lvl) in seg_list if int(lvl.item()) == lev]

    def forward_loss(
        smin_tau,
        occ_eps,
        compute_per_level: bool = False,
        level_cap: Optional[int] = None,
        loss_mode: str = "full",
        stage_weight: float = 1.0,
    ):
        pars = pack()
        seg_p = generate_segments_plane_policy(
            origin_fixed, pars['root_dir'],
            r0=pars['r0'], depth=pars['depth'],
            theta_deg_sched=pars['theta_deg'],
            gamma_val=pars['gamma'],
            ld_ratio_sched=pars['ld_ratio'],
            asym_sched=pars['asymmetry'],
            roll0_deg=None,
            n_free_plane=n_free_plane,
            plane_angle_deg_range=plane_angle_deg_range,
            orth_angle_deg=pars['orth_angle_deg'],
            free_target_angle_deg_sched=pars['free_target_angle_deg'],
            free_phi_deg_sched=None,
            root_length_mm=pars['root_length_mm'],
            roll0_normal=pars['roll0_normal'],
        )

        if loss_mode == "full" or level_cap is None:
            seg_p_loss = seg_p
            seg_t_loss = seg_t
            seg_p_prefix = seg_p
            seg_t_prefix = seg_t
        elif loss_mode == "prefix":
            seg_p_loss = _segments_upto_level(seg_p, level_cap)
            seg_t_loss = _segments_upto_level(seg_t, level_cap)
            seg_p_prefix = seg_p_loss
            seg_t_prefix = seg_t_loss
        elif loss_mode == "incremental":
            seg_p_loss = _segments_at_level(seg_p, level_cap)
            seg_t_loss = _segments_at_level(seg_t, level_cap)
            if len(seg_p_loss) == 0 or len(seg_t_loss) == 0:
                seg_p_loss = _segments_upto_level(seg_p, level_cap)
                seg_t_loss = _segments_upto_level(seg_t, level_cap)
            seg_p_prefix = _segments_upto_level(seg_p, level_cap)
            seg_t_prefix = _segments_upto_level(seg_t, level_cap)
        else:
            raise ValueError(f"Unknown loss_mode={loss_mode!r}")

        O_p = render_soft_mask(grid_world, seg_p_loss, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
        O_t_loss = O_t if (loss_mode == "full" or level_cap is None) else render_soft_mask(
            grid_world, seg_t_loss, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps)
        )

        if level_cap is None:
            cl_step_mm = 1.5
            cl_edge_ratio = 0.20
            cl_edge_factor = 5.0
            cl_min_pts = 8
        else:
            rel = max(0, int(level_cap) - 2)
            cl_step_mm = max(0.7, 1.4 - 0.10 * rel)
            cl_edge_ratio = min(0.30, 0.20 + 0.015 * rel)
            cl_edge_factor = min(8.0, 5.0 + 0.5 * rel)
            cl_min_pts = 8 + rel

        pts_p = sample_centerline_points_edge_focus(seg_p_loss, step_mm=cl_step_mm, edge_ratio=cl_edge_ratio, edge_factor=cl_edge_factor, min_points_per_seg=cl_min_pts)
        pts_t = sample_centerline_points_edge_focus(seg_t_loss, step_mm=cl_step_mm, edge_ratio=cl_edge_ratio, edge_factor=cl_edge_factor, min_points_per_seg=cl_min_pts)
        loss_cl = chamfer_l2(pts_p, pts_t)

        def _proj_normal(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            d = _unit(d)
            n = _unit(n)
            n = n - (n * d).sum() * d
            return n / (n.norm() + 1e-12)

        n_pred = _proj_normal(pars['roll0_normal'], pars['root_dir'])
        n_gt = _proj_normal(
            torch.tensor(target_cfg['roll0_normal'], device=device, dtype=torch.float32),
            torch.tensor(target_cfg['root_dir'], device=device, dtype=torch.float32),
        )
        loss_plane = 1.0 - torch.abs((n_pred * n_gt).sum()).clamp(0.0, 1.0)

        loss_t = tversky(O_p, O_t_loss, alpha=0.3, beta=0.7)
        vol_p = O_p.sum()
        vol_t_loss = O_t_loss.sum()
        loss_vol = (vol_p - vol_t_loss).abs() / (vol_t_loss + 1e-6)

        if if_bif:
            pred_bif_stats = bifurcation_level_stats(seg_p_loss, depth)
            tgt_bif_stats = {lv: st for lv, st in target_bif_stats.items() if (level_cap is None or lv <= int(level_cap))}
            loss_bif = bifurcation_loss_from_stats(pred_bif_stats, tgt_bif_stats, device=device)
        else:
            loss_bif = torch.zeros_like(loss_t)

        aux_prefix_weight = 0.15 if (loss_mode == "incremental" and level_cap is not None) else 0.0
        aux_global_weight = 0.05 if (loss_mode == "incremental" and level_cap is not None) else 0.0

        if aux_prefix_weight > 0.0:
            O_p_prefix = render_soft_mask(grid_world, seg_p_prefix, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
            O_t_prefix = render_soft_mask(grid_world, seg_t_prefix, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
            pts_p_prefix = sample_centerline_points_edge_focus(seg_p_prefix, step_mm=cl_step_mm, edge_ratio=cl_edge_ratio, edge_factor=cl_edge_factor, min_points_per_seg=cl_min_pts)
            pts_t_prefix = sample_centerline_points_edge_focus(seg_t_prefix, step_mm=cl_step_mm, edge_ratio=cl_edge_ratio, edge_factor=cl_edge_factor, min_points_per_seg=cl_min_pts)
            loss_cl_prefix = chamfer_l2(pts_p_prefix, pts_t_prefix)
            loss_t_prefix = tversky(O_p_prefix, O_t_prefix, alpha=0.3, beta=0.7)
            vol_p_prefix = O_p_prefix.sum()
            vol_t_prefix = O_t_prefix.sum()
            loss_vol_prefix = (vol_p_prefix - vol_t_prefix).abs() / (vol_t_prefix + 1e-6)
        else:
            loss_cl_prefix = torch.zeros_like(loss_cl)
            loss_t_prefix = torch.zeros_like(loss_t)
            loss_vol_prefix = torch.zeros_like(loss_vol)

        if aux_global_weight > 0.0:
            O_p_full = render_soft_mask(grid_world, seg_p, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
            pts_p_full = sample_centerline_points_edge_focus(seg_p, step_mm=1.5)
            pts_t_full = sample_centerline_points_edge_focus(seg_t, step_mm=1.5)
            loss_cl_full = chamfer_l2(pts_p_full, pts_t_full)
            loss_t_full = tversky(O_p_full, O_t, alpha=0.3, beta=0.7)
            vol_p_full = O_p_full.sum()
            vol_t_full = O_t.sum()
            loss_vol_full = (vol_p_full - vol_t_full).abs() / (vol_t_full + 1e-6)
        else:
            loss_cl_full = torch.zeros_like(loss_cl)
            loss_t_full = torch.zeros_like(loss_t)
            loss_vol_full = torch.zeros_like(loss_vol)

        bif_weight = 2.0 if if_bif else 0.0
        loss_main_unweighted = 1.0 * loss_t + 5.0 * loss_cl + 10.0 * loss_vol + bif_weight * loss_bif
        loss_aux_prefix = 1.0 * loss_t_prefix + 5.0 * loss_cl_prefix + 10.0 * loss_vol_prefix
        loss_aux_global = 1.0 * loss_t_full + 2.5 * loss_cl_full + 5.0 * loss_vol_full
        loss = stage_weight * loss_main_unweighted + aux_prefix_weight * loss_aux_prefix + aux_global_weight * loss_aux_global + 2.0 * loss_plane
        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        O_p_eval = render_soft_mask(grid_world, seg_p, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
        dice_score = 1.0 - soft_dice(O_p_eval, O_t)
        parts: Dict[str, Any] = dict(
            total=loss,
            t=loss_t,
            cl=loss_cl,
            vol=loss_vol,
            plane=loss_plane,
            bif=loss_bif,
            if_bif=bool(if_bif),
            dice=dice_score,
            stage_weight=float(stage_weight),
            main_unweighted=loss_main_unweighted,
            aux_prefix_weight=float(aux_prefix_weight),
            aux_global_weight=float(aux_global_weight),
            t_prefix=loss_t_prefix,
            cl_prefix=loss_cl_prefix,
            vol_prefix=loss_vol_prefix,
            t_full=loss_t_full,
            cl_full=loss_cl_full,
            vol_full=loss_vol_full,
            mode=loss_mode,
            level_cap=level_cap,
        )

        if compute_per_level and len(seg_p) > 0:
            by_lvl: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
            max_lvl = 0
            for (p, q, r, lvl) in seg_p:
                lv = int(lvl.item())
                max_lvl = max(max_lvl, lv)
                by_lvl.setdefault(lv, []).append((p, q, r, lvl))

            cum_stats: Dict[int, Dict[str, float]] = {}
            for L in range(1, max_lvl + 1):
                segs_L: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                for lv in range(1, L + 1):
                    segs_L.extend(by_lvl.get(lv, []))
                if not segs_L:
                    continue
                tgt_L = _segments_upto_level(seg_t, L)
                O_p_L = render_soft_mask(grid_world, segs_L, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
                O_t_L = render_soft_mask(grid_world, tgt_L, smin_tau_mm=float(smin_tau), occ_eps_mm=float(occ_eps))
                pts_p_L = sample_centerline_points_edge_focus(segs_L, step_mm=1.5)
                pts_t_L = sample_centerline_points_edge_focus(tgt_L, step_mm=1.5)
                loss_cl_L = chamfer_l2(pts_p_L, pts_t_L)
                loss_t_L = tversky(O_p_L, O_t_L, alpha=0.3, beta=0.7)
                vol_p_L = O_p_L.sum()
                vol_t_L = O_t_L.sum()
                loss_vol_L = (vol_p_L - vol_t_L).abs() / (vol_t_L + 1e-6)
                loss_total_L = 1.0 * loss_t_L + 5.0 * loss_cl_L + 10.0 * loss_vol_L
                cum_stats[L] = dict(total=float(loss_total_L.detach().cpu().item()), t=float(loss_t_L.detach().cpu().item()), cl=float(loss_cl_L.detach().cpu().item()), vol=float(loss_vol_L.detach().cpu().item()))

            inc_stats: Dict[int, Dict[str, float]] = {}
            prev = dict(total=0.0, t=0.0, cl=0.0, vol=0.0)
            for L in range(1, max_lvl + 1):
                if L not in cum_stats:
                    continue
                cur = cum_stats[L]
                inc_stats[L] = dict(total=cur["total"] - prev["total"], t=cur["t"] - prev["t"], cl=cur["cl"] - prev["cl"], vol=cur["vol"] - prev["vol"])
                prev = cur

            parts["per_level_cum"] = cum_stats
            parts["per_level_inc"] = inc_stats

        return loss, pars, seg_p, parts

    def _clone_trainable_state() -> Dict[str, torch.Tensor]:
        return {name: p.detach().clone() for name, p in pack.named_parameters() if p.requires_grad}

    def _restore_trainable_state(state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in pack.named_parameters():
                if p.requires_grad and (name in state):
                    p.copy_(state[name])

    def run_stage(stage_name: str,
                  max_iters: int,
                  smin_tau: float,
                  occ_eps: float,
                  lr_fast: float,
                  lr_slow: float,
                  lr_theta: float = 0.05,
                  min_iters: int = 100,
                  patience: int = 100,
                  rel_tol: float = 5e-4,
                  cl_rel_tol: float = 5e-4,
                  stage_level: Optional[int] = None,
                  stage_weight: float = 1.0,
                  loss_mode: str = "full",
                  global_step_start: int = 0) -> int:
        opt, sch = build_optimizer(lr_fast, lr_slow, lr_theta)
        best_loss = float('inf')
        best_cl = float('inf')
        best_state = _clone_trainable_state()
        no_improve = 0
        t0 = time.time()
        global_step = global_step_start
        ema_total = None
        ema_cl = None
        ema_beta = 0.85

        print(f"\n==> {stage_name}: max_iters={max_iters}, smin_tau={smin_tau}, occ_eps={occ_eps}")
        print(f"    lr_fast={lr_fast}, lr_slow={lr_slow}, lr_theta={lr_theta}")
        print(f"    stage_level={stage_level}, stage_weight={stage_weight:.3f}, loss_mode={loss_mode}")
        print(f"    early stopping: min_iters={min_iters}, patience={patience}, rel_tol={rel_tol}, cl_rel_tol={cl_rel_tol}")

        for it in range(1, max_iters + 1):
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss, pars, seg_p, parts = forward_loss(
                smin_tau, occ_eps,
                level_cap=stage_level,
                loss_mode=loss_mode,
                stage_weight=stage_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pack.parameters(), max_norm=1.2)
            opt.step()
            sch.step()

            with torch.no_grad():
                for p_ in pack.parameters():
                    if torch.isnan(p_).any() or torch.isinf(p_).any():
                        p_.nan_to_num_(0.0)
                if hasattr(pack, "r0_raw"):
                    pack.r0_raw.data.clamp_(-8.0, 8.0)
                if hasattr(pack, "gamma_raw"):
                    pack.gamma_raw.data.clamp_(-8.0, 8.0)
                if hasattr(pack, "ld_raw"):
                    pack.ld_raw.data.clamp_(-8.0, 8.0)
                if hasattr(pack, "theta_raw"):
                    pack.theta_raw.data.clamp_(-8.0, 8.0)

            loss_val = float(loss.detach().cpu().item())
            cl_val = float(parts['cl'].detach().cpu().item())
            loss_history.append((stage_name, global_step, loss_val))

            if it % 30 == 0 or it == 1:
                elapsed = time.time() - t0
                print(f"[{stage_name} {it}/{max_iters}] loss={loss_val:.6f} cl={cl_val:.6f} time={elapsed:.1f}s")
                print_params(pars, title="  Current parameters:")
                print(
                    "  components: "
                    f"t={float(parts['t'].detach().cpu()):.4e} "
                    f"cl={float(parts['cl'].detach().cpu()):.4e} "
                    f"vol={float(parts['vol'].detach().cpu()):.4e} "
                    f"plane={float(parts['plane'].detach().cpu()):.4e} "
                    f"bif={float(parts['bif'].detach().cpu()):.4e}"
                )
                print(
                    "  aux: "
                    f"prefix_w={parts.get('aux_prefix_weight', 0.0):.3f} "
                    f"global_w={parts.get('aux_global_weight', 0.0):.3f} "
                    f"mode={parts.get('mode', 'full')}"
                )

            if global_step % snapshot_interval == 0:
                _record_param_snapshot(tag='iter', global_step=global_step, stage_name=stage_name, pars_dict=pars)

            if ema_total is None:
                ema_total = loss_val
                ema_cl = cl_val
            else:
                ema_total = ema_beta * ema_total + (1.0 - ema_beta) * loss_val
                ema_cl = ema_beta * ema_cl + (1.0 - ema_beta) * cl_val

            improved_total = ema_total < best_loss * (1.0 - rel_tol)
            improved_cl = ema_cl < best_cl * (1.0 - cl_rel_tol)
            if improved_total or improved_cl:
                if improved_total:
                    best_loss = ema_total
                else:
                    best_loss = min(best_loss, ema_total)
                if improved_cl:
                    best_cl = ema_cl
                else:
                    best_cl = min(best_cl, ema_cl)
                best_state = _clone_trainable_state()
                no_improve = 0
            else:
                no_improve += 1

            if it >= min_iters and no_improve >= patience:
                print(f"[{stage_name}] Early stopping at iter {it}, best_ema_total={best_loss:.6f}, best_ema_cl={best_cl:.6f}, no_improve={no_improve}")
                break

        _restore_trainable_state(best_state)
        with torch.no_grad():
            loss_best, _, _, parts_best = forward_loss(
                smin_tau, occ_eps,
                level_cap=stage_level,
                loss_mode=loss_mode,
                stage_weight=stage_weight,
            )
        print(f"[{stage_name}] restored best checkpoint: total={float(loss_best.detach().cpu()):.6f}, cl={float(parts_best['cl'].detach().cpu()):.6f}")
        return global_step

    with torch.no_grad():
        _, pars_init_eval, _seg_init_eval, parts_init_eval = forward_loss(
            smin_tau=0.8, occ_eps=0.7, compute_per_level=False
        )
    _record_param_snapshot(tag='init_eval', global_step=0, stage_name='init', pars_dict=pars_init_eval)

    train_start = time.time()
    global_step = 0
    for sc in stage_cfgs:
        global_step = run_stage(
            stage_name=sc["name"],
            max_iters=sc["max_iters"],
            smin_tau=sc["smin_tau"],
            occ_eps=sc["occ_eps"],
            lr_fast=sc["lr_fast"],
            lr_slow=sc["lr_slow"],
            lr_theta=sc["lr_theta"],
            min_iters=sc["min_iters"],
            patience=sc["patience"],
            rel_tol=sc["rel_tol"],
            cl_rel_tol=sc["cl_rel_tol"],
            stage_level=sc.get("stage_level", None),
            stage_weight=sc.get("stage_weight", 1.0),
            loss_mode=sc.get("loss_mode", "full"),
            global_step_start=global_step,
        )

    print("\n[Info] LBFGS refinement disabled in this staged-all-parameter version.")

    total_minutes = (time.time() - train_start) / 60.0
    print(f"\nTotal optimization time (progressive stages only): {total_minutes:.2f} minutes")

    with torch.no_grad():
        _, pars_final, seg_p_final, parts = forward_loss(smin_tau=0.8, occ_eps=0.7, compute_per_level=True)
    _record_param_snapshot(tag='final', global_step=global_step, stage_name='final', pars_dict=pars_final)

    print_params(pars_final, title=f"\nFinal optimized parameters ({exp_name}):")
    print("\nFinal loss decomposition:")
    for key in ["t", "cl", "vol", "ldr", "dir", "dice"]:
        val = parts.get(key, None)
        if val is not None:
            print(f"  {key}: {float(val.detach().cpu().item()):.6f}")

    if parts.get("ld_ratio_detail", None):
        print("\nPer-level mean L/(2r) (pred / target):")
        for lvl in sorted(parts["ld_ratio_detail"].keys()):
            pred_val, true_val = parts["ld_ratio_detail"][lvl]
            print(f"  Level {lvl}: pred={pred_val:.3f}, target={true_val:.3f}")

    per_inc = parts.get("per_level_inc", None)
    if per_inc is not None:
        print("\nPer-level incremental contributions (sum ≈ full loss):")
        sum_inc = 0.0
        for lvl in sorted(per_inc.keys()):
            d = per_inc[lvl]
            sum_inc += d["total"]
            print(
                f"  Level {lvl}: "
                f"delta_total={d['total']:.6f}, "
                f"delta_t={d['t']:.6f}, "
                f"delta_cl={d['cl']:.6f}, "
                f"delta_vol={d['vol']:.6f}"
            )
        print(f"  Sum of delta_total over levels = {sum_inc:.6f}")

    

    if return_result:
        out = dict(
            exp_name=exp_name,
            init_cfg=init_cfg,
            target_cfg=target_cfg,
            init_pars={k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in pars_init_eval.items()},
            init_losses={k: (float(v.detach().cpu().item()) if torch.is_tensor(v) else v) for k, v in parts_init_eval.items() if isinstance(v, (torch.Tensor, float, int))},
            init_dice=float(parts_init_eval['dice'].detach().cpu().item()) if torch.is_tensor(parts_init_eval['dice']) else float(parts_init_eval['dice']),
            final_pars={k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in pars_final.items()},
            final_losses={k: (float(v.detach().cpu().item()) if torch.is_tensor(v) else v) for k, v in parts.items() if isinstance(v, (torch.Tensor, float, int))},
            final_dice=float(parts['dice'].detach().cpu().item()) if torch.is_tensor(parts['dice']) else float(parts['dice']),
            param_trace=[_to_serializable_value(x) for x in param_trace],
        )
        return out
    return None

# -------------------------- Enter --------------------------
def print_params_from_cfg(cfg: dict, title="Initial parameters"):
    print(title)
    for k, v in cfg.items():
        print(f"  {k}: {v}")


def _rand_unit_vec(rng: np.random.Generator) -> List[float]:
    v = rng.normal(size=(3,))
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).tolist()


def _rand_unit_vec_near(rng: np.random.Generator, v_true: Sequence[float], max_angle_deg: float = 15.0) -> List[float]:
    """Sample a random unit vector within a cone of max_angle_deg around v_true."""
    vt = np.asarray(v_true, dtype=np.float64).reshape(3)
    vt = vt / (np.linalg.norm(vt) + 1e-12)
    # sample rotation axis uniformly from directions orthogonal to vt
    a = rng.normal(size=(3,))
    a = a - vt * np.dot(a, vt)
    na = np.linalg.norm(a)
    if na < 1e-8:
        # fallback: pick a deterministic orthogonal axis
        a = np.cross(vt, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(a) < 1e-8:
            a = np.cross(vt, np.array([0.0, 1.0, 0.0]))
        na = np.linalg.norm(a)
    a = a / (na + 1e-12)

    ang = float(rng.uniform(0.0, max_angle_deg)) * np.pi / 180.0
    # Rodrigues' rotation formula
    v = (vt * np.cos(ang) +
         np.cross(a, vt) * np.sin(ang) +
         a * np.dot(a, vt) * (1.0 - np.cos(ang)))
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float64).tolist()



def _to_np_any(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_treeA_target_cfg(depth: int, n_head: int) -> dict:
    if depth < 2:
        raise ValueError('depth must be >= 2.')
    vec_len = depth - 1
    n_head = min(int(n_head), vec_len)
    return dict(
        r0=7.6, depth=depth, n_head=n_head,
        theta_deg=[46.0, 41.0, 37.0, 35.0, 35.0, 35.0][:vec_len],
        gamma=[3.35, 3.25, 3.15, 3.10, 3.10, 3.10][:vec_len],
        ld_ratio=[2.80, 3.10, 2.90, 2.70, 2.70, 2.70][:vec_len],
        asym=[0.06, 0.08, 0.07, 0.06, 0.06, 0.06][:vec_len],
        roll0_normal=[-0.71, 0.70, 0.02],
        root_dir=[0.0,0.0,-1.0],
        root_length_mm=9.5,
        orth_angle_deg=90.0,
        free_target_angle_deg=[90.0] * vec_len,
    )


def get_gt_pars(target_cfg: dict, depth: int, n_head: int, param_ranges: Optional[dict] = None) -> dict:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        _gt_pack = ParamPack(
            init_cfg=target_cfg,
            device=dev,
            depth=depth,
            n_head=n_head,
            learn_orth_angle=False,
            learn_gamma=False,
            learn_r0=False,
            learn_root=False,
            learn_root_dir=False,
            param_ranges=param_ranges,
        ).to(dev)
        _gt = _gt_pack.forward()
    return dict(
        r0=float(_to_np_any(_gt['r0'])),
        root_length_mm=float(_to_np_any(_gt['root_length_mm'])),
        gamma=_to_np_any(_gt['gamma']).reshape(-1).astype(np.float64),
        ld_ratio=_to_np_any(_gt['ld_ratio']).reshape(-1).astype(np.float64),
        theta_deg=_to_np_any(_gt['theta_deg']).reshape(-1).astype(np.float64),
        asymmetry=_to_np_any(_gt['asymmetry']).reshape(-1).astype(np.float64),
        roll0_normal=np.asarray(target_cfg['roll0_normal'], dtype=np.float64).reshape(3),
        root_dir=_to_np_any(_gt['root_dir']).reshape(3).astype(np.float64),
    )


def _proj_normal_np(v: np.ndarray, d: np.ndarray):
    d = d / (np.linalg.norm(d) + 1e-12)
    vp = v - np.dot(v, d) * d
    nv = np.linalg.norm(vp)
    if nv < 1e-8:
        return None
    return vp / (nv + 1e-12)


def param_err_from_gt(pname: str, final_pars: dict, gt_pars: dict) -> float:
    if pname == 'r0':
        return float(abs(float(final_pars['r0']) - gt_pars['r0']))
    if pname == 'root_length_mm':
        return float(abs(float(final_pars['root_length_mm']) - gt_pars['root_length_mm']))
    if pname == 'gamma':
        g = np.asarray(final_pars['gamma'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(g - gt_pars['gamma'])))
    if pname == 'ld_ratio':
        ld = np.asarray(final_pars['ld_ratio'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(ld - gt_pars['ld_ratio'])))
    if pname == 'theta_deg':
        th = np.asarray(final_pars['theta_deg'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(th - gt_pars['theta_deg'])))
    if pname == 'asymmetry':
        a = np.asarray(final_pars['asymmetry'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(a - gt_pars['asymmetry'])))
    if pname == 'root_dir':
        v = np.asarray(final_pars['root_dir'], dtype=np.float64).reshape(3)
        vt = np.asarray(gt_pars['root_dir'], dtype=np.float64).reshape(3)
        v = v / (np.linalg.norm(v) + 1e-12)
        vt = vt / (np.linalg.norm(vt) + 1e-12)
        cosang = float(np.clip(np.dot(v, vt), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))
    if pname == 'roll0_normal':
        v = np.asarray(final_pars['roll0_normal'], dtype=np.float64).reshape(3)
        d = np.asarray(final_pars['root_dir'], dtype=np.float64).reshape(3)
        vt = np.asarray(gt_pars['roll0_normal'], dtype=np.float64).reshape(3)
        dt = np.asarray(gt_pars['root_dir'], dtype=np.float64).reshape(3)
        vp = _proj_normal_np(v, d)
        vtp = _proj_normal_np(vt, dt)
        if vp is None or vtp is None:
            return float(90.0)
        cosang = float(np.clip(np.abs(np.dot(vp, vtp)), 0.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))
    raise ValueError(pname)


def compute_all_param_errors(final_pars: dict, gt_pars: dict, params: List[str]) -> Dict[str, float]:
    return {p: param_err_from_gt(p, final_pars, gt_pars) for p in params}


def random_init_for_param(rng: np.random.Generator, target_cfg: dict, pname: str, depth: int, param_ranges: Optional[dict] = None) -> dict:
    if depth < 2:
        raise ValueError('depth must be >= 2.')
    vec_len = depth - 1
    pr = _merge_param_ranges(param_ranges)
    theta_lo, theta_hi = pr['theta_deg']
    r0_lo, r0_hi = pr['r0']
    gamma_lo, gamma_hi = pr['gamma']
    ld_lo, ld_hi = pr['ld_ratio']
    asym_lo, asym_hi = pr['asymmetry']
    root_lo, root_hi = pr['root_length_mm']
    init_cfg = dict(target_cfg)
    init_cfg['free_target_angle_deg'] = [90.0] * vec_len
    init_cfg['roll0_normal'] = list(target_cfg['roll0_normal'])
    if pname == 'r0':
        init_cfg['r0'] = float(rng.uniform(r0_lo, r0_hi))
    elif pname == 'gamma':
        init_cfg['gamma'] = [float(rng.uniform(gamma_lo, gamma_hi)) for _ in range(vec_len)]
    elif pname == 'ld_ratio':
        init_cfg['ld_ratio'] = [float(rng.uniform(ld_lo, ld_hi)) for _ in range(vec_len)]
    elif pname == 'theta_deg':
        init_cfg['theta_deg'] = [float(rng.uniform(theta_lo, theta_hi)) for _ in range(vec_len)]
    elif pname == 'asymmetry':
        base = np.asarray(target_cfg['asym'], dtype=np.float64).reshape(-1)
        noise = rng.normal(loc=0.0, scale=0.02, size=base.shape)
        init_cfg['asym'] = np.clip(base + noise, asym_lo, asym_hi).astype(np.float64).tolist()
    elif pname == 'root_dir':
        init_cfg['root_dir'] = _rand_unit_vec(rng)
    elif pname == 'root_length_mm':
        init_cfg['root_length_mm'] = float(rng.uniform(root_lo, root_hi))
    elif pname == 'all_params':
        init_cfg['r0'] = float(rng.uniform(r0_lo, r0_hi))
        init_cfg['gamma'] = [float(rng.uniform(gamma_lo, gamma_hi)) for _ in range(vec_len)]
        init_cfg['ld_ratio'] = [float(rng.uniform(ld_lo, ld_hi)) for _ in range(vec_len)]
        init_cfg['theta_deg'] = [float(rng.uniform(theta_lo, theta_hi)) for _ in range(vec_len)]
        base = np.asarray(target_cfg['asym'], dtype=np.float64).reshape(-1)
        noise = rng.normal(loc=0.0, scale=0.02, size=base.shape)
        init_cfg['asym'] = np.clip(base + noise, asym_lo, asym_hi).astype(np.float64).tolist()
        init_cfg['root_dir'] = _rand_unit_vec(rng)
        init_cfg['root_length_mm'] = float(rng.uniform(root_lo, root_hi))
    else:
        raise ValueError(pname)
    return init_cfg


def warm_start_init_cfg(pname: str, final_pars: dict, target_cfg: dict, n_head: int) -> dict:
    init_cfg = dict(target_cfg)
    init_cfg['free_target_angle_deg'] = [90.0] * n_head
    init_cfg['roll0_normal'] = list(target_cfg['roll0_normal'])
    if pname == 'r0':
        init_cfg['r0'] = float(final_pars['r0'])
    elif pname == 'gamma':
        init_cfg['gamma'] = [float(x) for x in np.asarray(final_pars['gamma'], dtype=np.float64).reshape(-1)]
    elif pname == 'ld_ratio':
        init_cfg['ld_ratio'] = [float(x) for x in np.asarray(final_pars['ld_ratio'], dtype=np.float64).reshape(-1)]
    elif pname == 'theta_deg':
        init_cfg['theta_deg'] = [float(x) for x in np.asarray(final_pars['theta_deg'], dtype=np.float64).reshape(-1)]
    elif pname == 'root_dir':
        init_cfg['root_dir'] = [float(x) for x in np.asarray(final_pars['root_dir'], dtype=np.float64).reshape(3)]
    elif pname == 'root_length_mm':
        init_cfg['root_length_mm'] = float(final_pars['root_length_mm'])
    elif pname == 'asymmetry':
        init_cfg['asym'] = [float(x) for x in np.asarray(final_pars['asymmetry'], dtype=np.float64).reshape(-1)]
    elif pname == 'all_params':
        init_cfg['r0'] = float(final_pars['r0'])
        init_cfg['gamma'] = [float(x) for x in np.asarray(final_pars['gamma'], dtype=np.float64).reshape(-1)]
        init_cfg['ld_ratio'] = [float(x) for x in np.asarray(final_pars['ld_ratio'], dtype=np.float64).reshape(-1)]
        init_cfg['theta_deg'] = [float(x) for x in np.asarray(final_pars['theta_deg'], dtype=np.float64).reshape(-1)]
        init_cfg['root_dir'] = [float(x) for x in np.asarray(final_pars['root_dir'], dtype=np.float64).reshape(3)]
        init_cfg['root_length_mm'] = float(final_pars['root_length_mm'])
        init_cfg['asym'] = [float(x) for x in np.asarray(final_pars['asymmetry'], dtype=np.float64).reshape(-1)]
    else:
        raise ValueError(pname)
    return init_cfg


def _plot_dice_histograms(rows: List[dict], figdir: str, title_prefix: str = '') -> None:
    df = pd.DataFrame(rows)
    os.makedirs(figdir, exist_ok=True)
    if 'param' not in df.columns or 'dice' not in df.columns:
        return
    for pname in sorted(df['param'].unique()):
        sub = df[df['param'] == pname].copy()
        vals = sub['dice'].astype(float).values
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        plt.figure()
        plt.hist(vals, bins=min(20, max(5, len(vals)//2)))
        plt.axvline(med, linestyle='--', linewidth=1.5)
        ymax = plt.gca().get_ylim()[1]
        plt.text(med, ymax * 0.9, f'median={med:.4f}', rotation=90, va='top', ha='right')
        ax = plt.gca()
        ax.ticklabel_format(style='plain', axis='x', useOffset=False)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))
        plt.xlabel('Dice')
        plt.ylabel('count')
        plt.title(f'{title_prefix}{pname} Dice distribution (n={len(vals)})')
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f'dice_hist_{pname}.png'), dpi=160)
        plt.close()


def _save_json(obj: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def run_421_single_param_recovery(
    n_restarts: int = 10,
    params: Optional[List[str]] = None,
    seed: int = 0,
    depth: int = 6,
    n_head: int = 5,
    iters_adam_stage1: int = 200,
    iters_adam_stage2: int = 200,
    out_csv: str = "results_single_param.csv",
    warm_start_rounds: int = 1,
    param_ranges: Optional[dict] = None,
):
    if params is None:
        params = ['r0', 'gamma', 'ld_ratio', 'theta_deg', 'asymmetry', 'root_length_mm']
    params = [p for p in params if p not in ('roll0_normal', 'root_dir')]

    if depth < 2:
        raise ValueError('depth must be >= 2.')
    n_head = min(int(n_head), depth - 1)

    if depth < 2:
        raise ValueError('depth must be >= 2.')
    n_head = min(int(n_head), depth - 1)

    if int(warm_start_rounds) < 0:
        raise ValueError('warm_start_rounds must be >= 0')

    rng = np.random.default_rng(seed)
    param_ranges = _merge_param_ranges(param_ranges)
    target_cfg = build_treeA_target_cfg(depth=depth, n_head=n_head)
    gt_pars = get_gt_pars(target_cfg, depth=depth, n_head=n_head, param_ranges=param_ranges)
    rows: List[dict] = []

    for pname in params:
        print("\n" + "#" * 80)
        print(f"Single-parameter recovery: {pname} (restarts={n_restarts}, warm_rounds={warm_start_rounds})")
        print("#" * 80)

        for rep in range(n_restarts):
            init_cfg = random_init_for_param(rng, target_cfg, pname, depth, param_ranges=param_ranges)

            out = demo_fit_softsdf(
                depth=depth, n_head=n_head,
                iters_adam_stage1=iters_adam_stage1,
                iters_adam_stage2=iters_adam_stage2,
                exp_name=f"421_{pname}_rep{rep:02d}_round0",
                init_cfg=init_cfg,
                target_cfg_override=target_cfg,
                trainable_keys=[pname],
                learn_root_dir=('root_dir' in trainable),
                return_result=True,
                save_target_html=(rep == 0),
                param_ranges=param_ranges,
                if_bif=if_bif,
            )
            err0 = param_err_from_gt(pname, out['final_pars'], gt_pars)
            for rr in range(1, int(warm_start_rounds) + 1):
                init_cfg = warm_start_init_cfg(pname, out['final_pars'], target_cfg, n_head)
                out = demo_fit_softsdf(
                    depth=depth, n_head=n_head,
                    iters_adam_stage1=iters_adam_stage1,
                    iters_adam_stage2=iters_adam_stage2,
                    exp_name=f"421_{pname}_rep{rep:02d}_round{rr}",
                    init_cfg=init_cfg,
                    target_cfg_override=target_cfg,
                    trainable_keys=[pname],
                    learn_root_dir=False,
                    return_result=True,
                    save_target_html=False,
                    param_ranges=param_ranges,
                    if_bif=if_bif,
                )

            err = param_err_from_gt(pname, out['final_pars'], gt_pars)
            row = dict(
                experiment='single_param',
                param=pname,
                rep=rep,
                err=err,
                err_round0=err0,
                dice=float(out.get('final_dice', np.nan)),
                warm_rounds=int(warm_start_rounds),
                total_runs=int(warm_start_rounds) + 1,
            )
            for k, v in out.get('final_losses', {}).items():
                if isinstance(v, (float, int)):
                    row[f"loss_{k}"] = v
            rows.append(row)
            print(f"rep={rep:02d} err0={err0:.6f} -> err={err:.6f}, dice={row['dice']:.4f}, total_runs={int(warm_start_rounds) + 1}")

    if len(rows) == 0:
        raise RuntimeError('No rows were generated; check params and n_restarts.')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(rows)
        figdir = os.path.splitext(out_csv)[0] + '_figs'
        os.makedirs(figdir, exist_ok=True)

        for pname in sorted(df['param'].unique()):
            sub = df[df['param'] == pname].copy()
            errs = sub['err'].astype(float).values
            med_err = float(np.median(errs)) if len(errs) else float('nan')

            plt.figure()
            plt.hist(errs, bins=min(20, max(5, len(errs)//2)))
            plt.axvline(med_err, linestyle='--', linewidth=1.5)
            ymax = plt.gca().get_ylim()[1]
            plt.text(med_err, ymax * 0.9, f'median={med_err:.4f}', rotation=90, va='top', ha='right')
            plt.xlabel('error (abs or angular degrees)')
            plt.ylabel('count')
            plt.title(f'Single-parameter recovery: {pname} (n={len(errs)})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'err_hist_{pname}.png'), dpi=160)
            plt.close()

        plt.figure(figsize=(max(6, 1.2 * len(df['param'].unique())), 4))
        order = sorted(df['param'].unique())
        data = [df[df['param'] == p]['err'].astype(float).values for p in order]
        plt.boxplot(data, labels=order, showfliers=True)
        plt.ylabel('error (abs or angular degrees)')
        plt.title('Single-parameter recovery errors (boxplot)')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, 'err_boxplot_all_params.png'), dpi=160)
        plt.close()

        _plot_dice_histograms(rows, figdir, title_prefix='4.2.1 ')
        print(f'Saved figures to: {figdir}')
    except Exception as e:
        print(f'Plotting skipped due to error: {e}')

    return rows


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def _to_serializable_value(v: Any):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_serializable_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_serializable_value(val) for k, val in v.items()}
    return v


def _np_array_cpu(v: Any, dtype=np.float64) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    elif isinstance(v, (list, tuple)):
        v = np.asarray([_to_serializable_value(x) for x in v], dtype=dtype)
    else:
        v = np.asarray(v, dtype=dtype)
    return np.asarray(v, dtype=dtype)


def _target_cfg_to_row(target_cfg: dict) -> dict:
    row = {}
    row['target_r0'] = float(target_cfg['r0'])
    row['target_root_length_mm'] = float(target_cfg['root_length_mm'])
    row['target_theta_deg_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['theta_deg']).reshape(-1)])
    row['target_gamma_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['gamma']).reshape(-1)])
    row['target_ld_ratio_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['ld_ratio']).reshape(-1)])
    row['target_asym_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['asym']).reshape(-1)])
    row['target_root_dir_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['root_dir']).reshape(3)])
    row['target_roll0_normal_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['roll0_normal']).reshape(3)])
    row['target_free_target_angle_deg_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(target_cfg['free_target_angle_deg']).reshape(-1)])
    row['target_orth_angle_deg'] = float(target_cfg.get('orth_angle_deg', 90.0))
    return row


def _pars_to_row(prefix: str, pars: dict) -> dict:
    row = {}
    if 'r0' in pars:
        row[f'{prefix}_r0'] = float(_np_array_cpu(pars['r0']).reshape(()))
    if 'root_length_mm' in pars:
        row[f'{prefix}_root_length_mm'] = float(_np_array_cpu(pars['root_length_mm']).reshape(()))
    if 'theta_deg' in pars:
        row[f'{prefix}_theta_deg_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['theta_deg']).reshape(-1)])
    if 'gamma' in pars:
        row[f'{prefix}_gamma_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['gamma']).reshape(-1)])
    if 'ld_ratio' in pars:
        row[f'{prefix}_ld_ratio_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['ld_ratio']).reshape(-1)])
    if 'asymmetry' in pars:
        row[f'{prefix}_asymmetry_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['asymmetry']).reshape(-1)])
    elif 'asym' in pars:
        row[f'{prefix}_asymmetry_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['asym']).reshape(-1)])
    if 'root_dir' in pars:
        row[f'{prefix}_root_dir_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['root_dir']).reshape(3)])
    if 'roll0_normal' in pars:
        row[f'{prefix}_roll0_normal_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['roll0_normal']).reshape(3)])
    if 'free_target_angle_deg' in pars:
        row[f'{prefix}_free_target_angle_deg_json'] = _json_dumps_compact([float(x) for x in _np_array_cpu(pars['free_target_angle_deg']).reshape(-1)])
    if 'orth_angle_deg' in pars:
        row[f'{prefix}_orth_angle_deg'] = float(_np_array_cpu(pars['orth_angle_deg']).reshape(()))
    return row


def _losses_to_row(prefix: str, losses: dict) -> dict:
    row = {}
    for k, v in losses.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            row[f'{prefix}_{k}'] = float(v)
    return row


def _mean_param_error_row(row: dict, trainable: List[str]) -> float:
    vals = []
    for p in trainable:
        key = f'err_{p}'
        try:
            v = float(row.get(key, np.nan))
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')


def _relative_param_error_from_gt(pname: str, final_pars: dict, gt_pars: dict) -> float:
    eps = 1e-8
    if pname == 'r0':
        gt = float(gt_pars['r0'])
        return float(abs(float(final_pars['r0']) - gt) / max(abs(gt), eps))
    if pname == 'root_length_mm':
        gt = float(gt_pars['root_length_mm'])
        return float(abs(float(final_pars['root_length_mm']) - gt) / max(abs(gt), eps))
    if pname == 'gamma':
        x = np.asarray(final_pars['gamma'], dtype=np.float64).reshape(-1)
        gt = np.asarray(gt_pars['gamma'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(x - gt) / np.maximum(np.abs(gt), eps)))
    if pname == 'ld_ratio':
        x = np.asarray(final_pars['ld_ratio'], dtype=np.float64).reshape(-1)
        gt = np.asarray(gt_pars['ld_ratio'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(x - gt) / np.maximum(np.abs(gt), eps)))
    if pname == 'theta_deg':
        x = np.asarray(final_pars['theta_deg'], dtype=np.float64).reshape(-1)
        gt = np.asarray(gt_pars['theta_deg'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(x - gt) / np.maximum(np.abs(gt), eps)))
    if pname == 'asymmetry':
        x = np.asarray(final_pars['asymmetry'], dtype=np.float64).reshape(-1)
        gt = np.asarray(gt_pars['asymmetry'], dtype=np.float64).reshape(-1)
        return float(np.mean(np.abs(x - gt) / np.maximum(np.abs(gt), eps)))
    if pname in ('root_dir', 'roll0_normal'):
        return float('nan')
    raise ValueError(pname)


def compute_all_param_rel_errors(final_pars: dict, gt_pars: dict, params: List[str]) -> Dict[str, float]:
    return {p: _relative_param_error_from_gt(p, final_pars, gt_pars) for p in params}


def _mean_param_rel_error_row(row: dict, trainable: List[str]) -> float:
    vals = []
    for p in trainable:
        key = f'rel_err_{p}'
        try:
            v = float(row.get(key, np.nan))
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')


def _sample_signed_uniform(rng: np.random.Generator, center: float, pct: float) -> float:
    width = abs(center) * pct
    if width < 1e-8:
        width = pct
    return float(rng.uniform(center - width, center + width))


def _sample_vec_pct(rng: np.random.Generator, base: Sequence[float], pct: float, clip_lo: Optional[float] = None, clip_hi: Optional[float] = None) -> List[float]:
    arr = np.asarray(base, dtype=np.float64).reshape(-1)
    out = np.empty_like(arr)
    for i, c in enumerate(arr):
        out[i] = _sample_signed_uniform(rng, float(c), pct)
    if clip_lo is not None or clip_hi is not None:
        lo = -np.inf if clip_lo is None else clip_lo
        hi = np.inf if clip_hi is None else clip_hi
        out = np.clip(out, lo, hi)
    return out.astype(np.float64).tolist()


def _sample_root_dir_pct(rng: np.random.Generator, base: Sequence[float], pct: float) -> List[float]:
    arr = np.asarray(base, dtype=np.float64).reshape(3)
    out = np.array([_sample_signed_uniform(rng, float(c), pct) for c in arr], dtype=np.float64)
    n = np.linalg.norm(out)
    if n < 1e-8:
        out = np.asarray(base, dtype=np.float64).reshape(3)
        n = np.linalg.norm(out)
    return (out / (n + 1e-12)).tolist()


def _orthonormal_roll0_from_root(rng: np.random.Generator, root_dir: Sequence[float], base_roll0: Optional[Sequence[float]] = None) -> List[float]:
    d = np.asarray(root_dir, dtype=np.float64).reshape(3)
    d = d / (np.linalg.norm(d) + 1e-12)
    if base_roll0 is None:
        a = rng.normal(size=(3,))
    else:
        a = np.asarray(base_roll0, dtype=np.float64).reshape(3) + rng.normal(0.0, 0.08, size=(3,))
    a = a - np.dot(a, d) * d
    na = np.linalg.norm(a)
    if na < 1e-8:
        a = np.cross(d, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(a) < 1e-8:
            a = np.cross(d, np.array([0.0, 1.0, 0.0]))
        na = np.linalg.norm(a)
    return (a / (na + 1e-12)).astype(np.float64).tolist()


def build_random_target_cfg(rng: np.random.Generator, depth: int, n_head: int, base_cfg: Optional[dict] = None, param_ranges: Optional[dict] = None) -> dict:
    if base_cfg is None:
        base_cfg = build_treeA_target_cfg(depth=depth, n_head=n_head)
    vec_len = depth - 1
    pr = _merge_param_ranges(param_ranges)
    theta_lo, theta_hi = pr['theta_deg']
    r0_lo, r0_hi = pr['r0']
    gamma_lo, gamma_hi = pr['gamma']
    ld_lo, ld_hi = pr['ld_ratio']
    asym_lo, asym_hi = pr['asymmetry']
    root_lo, root_hi = pr['root_length_mm']
    theta_base = np.asarray(base_cfg['theta_deg'], dtype=np.float64).reshape(-1)
    gamma_base = np.asarray(base_cfg['gamma'], dtype=np.float64).reshape(-1)
    ld_base = np.asarray(base_cfg['ld_ratio'], dtype=np.float64).reshape(-1)
    asym_base = np.asarray(base_cfg['asym'], dtype=np.float64).reshape(-1)
    theta = np.clip(theta_base + rng.normal(0.0, 5.0, size=theta_base.shape), theta_lo, theta_hi)
    gamma = np.clip(gamma_base + rng.normal(0.0, 0.12, size=gamma_base.shape), gamma_lo, gamma_hi)
    ld_ratio = np.clip(ld_base + rng.normal(0.0, 0.35, size=ld_base.shape), ld_lo, ld_hi)
    asym = np.clip(asym_base + rng.normal(0.0, 0.015, size=asym_base.shape), asym_lo, asym_hi)
    root_dir = _rand_unit_vec_near(rng, base_cfg['root_dir'], max_angle_deg=25.0)
    roll0_normal = _orthonormal_roll0_from_root(rng, root_dir, base_roll0=base_cfg['roll0_normal'])
    return dict(
        r0=float(np.clip(base_cfg['r0'] + rng.normal(0.0, 0.45), r0_lo, r0_hi)),
        depth=depth,
        n_head=min(int(n_head), vec_len),
        theta_deg=theta.astype(np.float64).tolist(),
        gamma=gamma.astype(np.float64).tolist(),
        ld_ratio=ld_ratio.astype(np.float64).tolist(),
        asym=asym.astype(np.float64).tolist(),
        roll0_normal=roll0_normal,
        root_dir=root_dir,
        root_length_mm=float(np.clip(base_cfg['root_length_mm'] + rng.normal(0.0, 1.0), root_lo, root_hi)),
        orth_angle_deg=float(base_cfg.get('orth_angle_deg', 90.0)),
        free_target_angle_deg=[90.0] * vec_len,
    )


def make_all_param_init_cfg(rng: np.random.Generator, target_cfg: dict, depth: int,
                            init_mode: str = 'random', init_perturb_pct: float = 0.20, param_ranges: Optional[dict] = None) -> dict:
    pr = _merge_param_ranges(param_ranges)
    theta_lo, theta_hi = pr['theta_deg']
    r0_lo, r0_hi = pr['r0']
    gamma_lo, gamma_hi = pr['gamma']
    ld_lo, ld_hi = pr['ld_ratio']
    asym_lo, asym_hi = pr['asymmetry']
    root_lo, root_hi = pr['root_length_mm']
    if init_mode == 'random':
        return random_init_for_param(rng, target_cfg, 'all_params', depth, param_ranges=pr)
    if init_mode != 'perturb_target':
        raise ValueError(f'Unknown init_mode={init_mode!r}; allowed: random, perturb_target')
    vec_len = depth - 1
    pr = _merge_param_ranges(param_ranges)
    theta_lo, theta_hi = pr['theta_deg']
    r0_lo, r0_hi = pr['r0']
    gamma_lo, gamma_hi = pr['gamma']
    ld_lo, ld_hi = pr['ld_ratio']
    asym_lo, asym_hi = pr['asymmetry']
    root_lo, root_hi = pr['root_length_mm']
    init_cfg = dict(target_cfg)
    init_cfg['free_target_angle_deg'] = [90.0] * vec_len
    init_cfg['r0'] = float(np.clip(_sample_signed_uniform(rng, float(target_cfg['r0']), init_perturb_pct), r0_lo, r0_hi))
    init_cfg['gamma'] = _sample_vec_pct(rng, target_cfg['gamma'], init_perturb_pct, gamma_lo, gamma_hi)
    init_cfg['ld_ratio'] = _sample_vec_pct(rng, target_cfg['ld_ratio'], init_perturb_pct, ld_lo, ld_hi)
    init_cfg['theta_deg'] = _sample_vec_pct(rng, target_cfg['theta_deg'], init_perturb_pct, theta_lo, theta_hi)
    init_cfg['asym'] = _sample_vec_pct(rng, target_cfg['asym'], init_perturb_pct, asym_lo, asym_hi)
    init_cfg['root_dir'] = _sample_root_dir_pct(rng, target_cfg['root_dir'], init_perturb_pct)
    init_cfg['root_length_mm'] = float(np.clip(_sample_signed_uniform(rng, float(target_cfg['root_length_mm']), init_perturb_pct), root_lo, root_hi))
    init_cfg['roll0_normal'] = list(target_cfg['roll0_normal'])
    return init_cfg


def _save_csv_rows(rows: List[dict], path: str) -> None:
    if len(rows) == 0:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _expand_best_trace_rows(best_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for best_row in best_rows:
        target_base = {k: v for k, v in best_row.items() if not str(k).startswith('param_trace_')}
        trace_json = best_row.get('param_trace_json', '[]')
        try:
            trace = json.loads(trace_json) if isinstance(trace_json, str) else list(trace_json)
        except Exception:
            trace = []

        target_snapshot = dict(target_base)
        target_snapshot['trace_tag'] = 'target'
        target_snapshot['trace_global_step'] = 0
        target_snapshot['trace_stage_name'] = 'target'
        target_snapshot.update(_target_cfg_to_row({
            'r0': best_row.get('target_r0', np.nan),
            'root_length_mm': best_row.get('target_root_length_mm', np.nan),
            'theta_deg': json.loads(best_row.get('target_theta_deg_json', '[]')),
            'gamma': json.loads(best_row.get('target_gamma_json', '[]')),
            'ld_ratio': json.loads(best_row.get('target_ld_ratio_json', '[]')),
            'asym': json.loads(best_row.get('target_asym_json', '[]')),
            'root_dir': json.loads(best_row.get('target_root_dir_json', '[0,0,0]')),
            'roll0_normal': json.loads(best_row.get('target_roll0_normal_json', '[0,0,0]')),
            'free_target_angle_deg': json.loads(best_row.get('target_free_target_angle_deg_json', '[]')),
            'orth_angle_deg': best_row.get('target_orth_angle_deg', 90.0),
        }))
        rows.append(target_snapshot)

        for snap in trace:
            row = dict(target_base)
            row['trace_tag'] = snap.get('tag', '')
            row['trace_global_step'] = int(snap.get('global_step', 0))
            row['trace_stage_name'] = snap.get('stage_name', '')
            for k, v in snap.items():
                if k in ('tag', 'global_step', 'stage_name'):
                    continue
                row[k] = v
            rows.append(row)
    return rows


def _plot_multitarget_allparam_figures(detail_rows: List[dict], best_rows: List[dict], out_prefix: str, trainable: List[str]) -> None:
    figdir = f'{out_prefix}_figs'
    os.makedirs(figdir, exist_ok=True)

    df_all = pd.DataFrame(detail_rows)
    df_best = pd.DataFrame(best_rows)

    def _series(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            return np.asarray([], dtype=np.float64)
        return pd.to_numeric(df[col], errors='coerce').dropna().values.astype(np.float64)

    def _parse_json_vec(v) -> np.ndarray:
        if isinstance(v, str):
            try:
                return np.asarray(json.loads(v), dtype=np.float64).reshape(-1)
            except Exception:
                return np.asarray([], dtype=np.float64)
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.asarray(v, dtype=np.float64).reshape(-1)
        return np.asarray([], dtype=np.float64)

    def _box(vals: np.ndarray, fname: str, title: str, ylabel: str, label: str):
        if vals.size == 0:
            return
        plt.figure(figsize=(5.8, 4.2))
        plt.boxplot([vals], labels=[label], showfliers=True)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, fname), dpi=170)
        plt.close()

    for suffix, df in [('all', df_all), ('best', df_best)]:
        _box(_series(df, 'final_dice'), f'combo_dice_{suffix}.png', f'Dice ({suffix})', 'Dice', 'dice')
        _box(_series(df, 'final_loss_cl'), f'combo_centerline_{suffix}.png', f'Centerline loss ({suffix})', 'Chamfer loss', 'centerline')
        _box(_series(df, 'final_loss_vol'), f'combo_volume_{suffix}.png', f'Volume loss ({suffix})', 'Volume loss', 'volume')
        _box(_series(df, 'avg_rel_err'), f'box_avg_rel_err_{suffix}.png', f'Average relative error ({suffix})', 'Average relative error', 'avg rel err')

        dice_vals = _series(df, 'final_dice')
        if dice_vals.size:
            succ = float(np.mean(dice_vals >= 0.95))
            plt.figure(figsize=(4.6, 4.2))
            plt.bar([0], [succ])
            plt.xticks([0], ['dice>=0.95'])
            plt.ylim(0.0, 1.0)
            plt.ylabel('Success rate')
            plt.title(f'Success rate ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'combo_success_dice95_{suffix}.png'), dpi=170)
            plt.close()

        rel_vals = _series(df, 'avg_rel_err')
        if rel_vals.size:
            succ = float(np.mean(rel_vals <= 0.10))
            plt.figure(figsize=(4.6, 4.2))
            plt.bar([0], [succ])
            plt.xticks([0], ['avg rel err<=0.10'])
            plt.ylim(0.0, 1.0)
            plt.ylabel('Success rate')
            plt.title(f'Success rate ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'combo_success_rel10_{suffix}.png'), dpi=170)
            plt.close()

        # summary bars per parameter
        abs_means, abs_stds, rel_means, rel_stds, labels = [], [], [], [], []
        for p in trainable:
            labels.append(p)
            av = _series(df, f'err_{p}')
            rv = _series(df, f'rel_err_{p}')
            abs_means.append(float(np.mean(av)) if av.size else np.nan)
            abs_stds.append(float(np.std(av, ddof=0)) if av.size else np.nan)
            rel_means.append(float(np.mean(rv)) if rv.size else np.nan)
            rel_stds.append(float(np.std(rv, ddof=0)) if rv.size else np.nan)

            _box(av, f'param_{p}_error_{suffix}.png', f'{p} absolute error ({suffix})', 'Absolute error', p)
            _box(rv, f'param_{p}_rel_error_{suffix}.png', f'{p} relative error ({suffix})', 'Relative error', p)

        x = np.arange(len(labels))
        if len(labels):
            plt.figure(figsize=(max(7, 1.1 * len(labels)), 4.6))
            plt.bar(x, abs_means, yerr=abs_stds, capsize=4)
            plt.xticks(x, labels, rotation=30, ha='right')
            plt.ylabel('Mean absolute error')
            plt.title(f'Per-parameter mean absolute error ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'combo_avg_err_{suffix}.png'), dpi=170)
            plt.close()

            plt.figure(figsize=(max(7, 1.1 * len(labels)), 4.6))
            plt.bar(x, rel_means, yerr=rel_stds, capsize=4)
            plt.xticks(x, labels, rotation=30, ha='right')
            plt.ylabel('Mean relative error')
            plt.title(f'Per-parameter mean relative error ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'combo_avg_rel_err_{suffix}.png'), dpi=170)
            plt.close()

    # Levelwise analysis (all and best), x-axis strictly integer levels
    level_specs = [
        ('theta_deg', 'final_theta_deg_json', 'target_theta_deg_json'),
        ('gamma', 'final_gamma_json', 'target_gamma_json'),
        ('ld_ratio', 'final_ld_ratio_json', 'target_ld_ratio_json'),
        ('asymmetry', 'final_asymmetry_json', 'target_asym_json'),
    ]
    trainable_set = set(trainable)
    for pname, pred_col, tgt_col in level_specs:
        if pname not in trainable_set:
            continue
        for suffix, df in [('all', df_all), ('best', df_best)]:
            curves = []
            for _, row in df.iterrows():
                pred = _parse_json_vec(row.get(pred_col, '[]'))
                tgt = _parse_json_vec(row.get(tgt_col, '[]'))
                m = min(pred.size, tgt.size)
                if m > 0:
                    curves.append(np.abs(pred[:m] - tgt[:m]))
            if not curves:
                continue
            L = min(len(c) for c in curves)
            arr = np.stack([c[:L] for c in curves], axis=0)
            levels = np.arange(2, 2 + L, dtype=int)
            mean = arr.mean(axis=0)
            err_min = arr.min(axis=0)
            err_max = arr.max(axis=0)
            plt.figure(figsize=(6.2, 4.4))
            plt.plot(levels, mean, marker='o')
            plt.fill_between(levels, err_min, err_max, alpha=0.2)
            plt.xticks(levels, levels)
            plt.xlabel('Level')
            plt.ylabel('Absolute error')
            plt.title(f'Levelwise {pname} error ({suffix})')
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f'levelwise_{pname}_{suffix}.png'), dpi=170)
            plt.close()

    print(f'Saved figures to: {figdir}')


def _run_all_param_restart_worker(task: dict) -> dict:
    torch.set_num_threads(int(task.get('torch_num_threads', 1)))
    seed = int(task['seed'])
    np.random.seed(seed)
    target_cfg = task['target_cfg']
    depth = int(task['depth'])
    n_head = int(task['n_head'])
    trainable = list(task['trainable'])
    warm_rounds = int(task['warm_rounds'])
    param_ranges = task.get('param_ranges')
    rng = np.random.default_rng(seed)
    init_cfg = make_all_param_init_cfg(rng, target_cfg, depth, init_mode=task['init_mode'], init_perturb_pct=float(task['init_perturb_pct']), param_ranges=param_ranges)
    out = demo_fit_softsdf(
        depth=depth, n_head=n_head,
        iters_adam_stage1=int(task['iters1']),
        iters_adam_stage2=int(task['iters2']),
        exp_name=str(task['exp_name']),
        init_cfg=init_cfg,
        target_cfg_override=target_cfg,
        trainable_keys=trainable,
        learn_root_dir=('root_dir' in trainable),
        return_result=True,
        save_target_html=bool(task.get('save_target_html', False)),
        param_ranges=param_ranges,
        if_bif=bool(task.get('if_bif', True)),
        snapshot_interval=int(task.get('snapshot_interval', 60)),
    )
    for rr in range(1, warm_rounds + 1):
        init_cfg = warm_start_init_cfg('all_params', out['final_pars'], target_cfg, n_head)
        out = demo_fit_softsdf(
            depth=depth, n_head=n_head,
            iters_adam_stage1=int(task['iters1']),
            iters_adam_stage2=int(task['iters2']),
            exp_name=f"{task['exp_name']}_warm{rr}",
            init_cfg=init_cfg,
            target_cfg_override=target_cfg,
            trainable_keys=trainable,
            learn_root_dir=('root_dir' in trainable),
            return_result=True,
            save_target_html=False,
            param_ranges=param_ranges,
            if_bif=bool(task.get('if_bif', True)),
        )
    gt_pars = task['gt_pars']
    err_dict = compute_all_param_errors(out['final_pars'], gt_pars, trainable)
    rel_err_dict = compute_all_param_rel_errors(out['final_pars'], gt_pars, trainable)
    row = dict(
        experiment='all_params_multitarget',
        param='all_params',
        target_id=int(task['target_id']),
        rep=int(task['rep']),
        seed=int(seed),
        depth=depth,
        n_head=n_head,
        init_mode=str(task['init_mode']),
        init_perturb_pct=float(task['init_perturb_pct']),
        warm_rounds=warm_rounds,
        total_runs=warm_rounds + 1,
        init_dice=float(out.get('init_dice', np.nan)),
        final_dice=float(out.get('final_dice', np.nan)),
    )
    row.update(_target_cfg_to_row(target_cfg))
    row.update(_pars_to_row('init', out.get('init_cfg', {})))
    row.update(_pars_to_row('init_eval', out.get('init_pars', {})))
    row.update(_pars_to_row('final', out.get('final_pars', {})))
    row.update(_losses_to_row('init_loss', out.get('init_losses', {})))
    row.update(_losses_to_row('final_loss', out.get('final_losses', {})))
    row['param_trace_json'] = _json_dumps_compact(out.get('param_trace', []))
    for p in trainable:
        row[f'err_{p}'] = float(err_dict[p])
        row[f'rel_err_{p}'] = float(rel_err_dict.get(p, np.nan))
    row['avg_err'] = _mean_param_error_row(row, trainable)
    row['avg_rel_err'] = _mean_param_rel_error_row(row, trainable)
    return row


def run_421_all_param_recovery(
    n_restarts: int = 10,
    seed: int = 0,
    depth: int = 6,
    n_head: int = 5,
    iters_adam_stage1: int = 200,
    iters_adam_stage2: int = 200,
    out_prefix: str = 'results_multi_param',
    warm_start_rounds: int = 1,
    num_targets: int = 1,
    init_mode: str = 'random',
    init_perturb_pct: float = 0.20,
    trainable_params: Optional[List[str]] = None,
    num_workers: int = 1,
    param_ranges: Optional[dict] = None,
    if_bif: bool = True,
    snapshot_interval: int = 60,
):
    if int(warm_start_rounds) < 0:
        raise ValueError('warm_start_rounds must be >= 0')
    if int(num_targets) < 1:
        raise ValueError('num_targets must be >= 1')
    if float(init_perturb_pct) < 0.0:
        raise ValueError('init_perturb_pct must be >= 0')

    param_ranges = _merge_param_ranges(param_ranges)
    num_workers = max(1, int(num_workers))
    rng = np.random.default_rng(seed)
    base_target = build_treeA_target_cfg(depth=depth, n_head=n_head)
    target_cfgs: List[dict] = []
    for tid in range(int(num_targets)):
        if int(num_targets) == 1 and tid == 0:
            target_cfgs.append(base_target)
        else:
            target_cfgs.append(build_random_target_cfg(rng, depth=depth, n_head=n_head, base_cfg=base_target, param_ranges=param_ranges))

    allowed_trainable = ['r0', 'gamma', 'ld_ratio', 'theta_deg', 'asymmetry', 'root_length_mm', 'root_dir', 'roll0_normal']
    trainable = list(trainable_params) if trainable_params is not None else ['r0', 'gamma', 'ld_ratio', 'theta_deg', 'asymmetry', 'root_length_mm']
    bad = [p for p in trainable if p not in allowed_trainable]
    if bad:
        raise ValueError(f'Unknown trainable_params={bad}; allowed={allowed_trainable}')

    gt_by_target = {tid: get_gt_pars(cfg, depth=depth, n_head=n_head, param_ranges=param_ranges) for tid, cfg in enumerate(target_cfgs)}

    task_list = []
    for tid, target_cfg in enumerate(target_cfgs):
        for rep in range(int(n_restarts)):
            rep_seed = int(seed) + 10007 * int(tid) + 100003 * int(rep) + 97 * int(depth)
            task_list.append(dict(
                target_id=int(tid), rep=int(rep), seed=int(rep_seed), target_cfg=target_cfg, gt_pars=gt_by_target[tid],
                depth=int(depth), n_head=int(n_head), trainable=trainable, warm_rounds=int(warm_start_rounds),
                iters1=int(iters_adam_stage1), iters2=int(iters_adam_stage2),
                init_mode=str(init_mode), init_perturb_pct=float(init_perturb_pct),
                exp_name=f'multi_params_t{tid:02d}_rep{rep:02d}_round0', save_target_html=(rep == 0),
                torch_num_threads=1, param_ranges=param_ranges, if_bif=bool(if_bif),
                snapshot_interval=int(snapshot_interval),
            ))

    print(f"Parallel all-parameter recovery: tasks={len(task_list)}, num_workers={num_workers}")
    if num_workers > 1 and len(task_list) > 1:
        max_workers = min(num_workers, len(task_list))
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=_spawn_ctx()) as ex:
            detail_rows = list(ex.map(_run_all_param_restart_worker, task_list))
    else:
        detail_rows = [_run_all_param_restart_worker(t) for t in task_list]

    detail_rows = sorted(detail_rows, key=lambda r: (int(r.get('target_id', 10**9)), int(r.get('rep', 10**9))))
    best_rows: List[dict] = []
    for tid in range(int(num_targets)):
        target_rows = [r for r in detail_rows if int(r.get('target_id', -1)) == tid]
        for row in target_rows:
            print(f"target={tid:02d} rep={int(row.get('rep',-1)):02d} final_dice={float(row.get('final_dice', np.nan)):.4f}, avg_rel_err={float(row.get('avg_rel_err', np.nan)):.4f}")
        if target_rows:
            # Best is selected by the observable optimization objective, not by GT parameter error.
            best_row = min(target_rows, key=lambda r: (float(r.get('final_loss_total', np.inf)), -float(r.get('final_dice', -np.inf))))
            best_row2 = dict(best_row)
            best_row2['is_best_target'] = 1
            best_rows.append(best_row2)

    summary_rows = []
    summary_best_rows = []
    for p in trainable:
        vals_all = np.asarray([r.get(f'err_{p}', np.nan) for r in detail_rows], dtype=np.float64)
        vals_all = vals_all[np.isfinite(vals_all)]
        rel_all = np.asarray([r.get(f'rel_err_{p}', np.nan) for r in detail_rows], dtype=np.float64)
        rel_all = rel_all[np.isfinite(rel_all)]
        summary_rows.append(dict(param=p, n=int(vals_all.size), mean_err=float(vals_all.mean()) if vals_all.size else np.nan,
                                 std=float(vals_all.std(ddof=0)) if vals_all.size else np.nan,
                                 mean_rel_err=float(rel_all.mean()) if rel_all.size else np.nan,
                                 std_rel_err=float(rel_all.std(ddof=0)) if rel_all.size else np.nan))
        vals_best = np.asarray([r.get(f'err_{p}', np.nan) for r in best_rows], dtype=np.float64)
        vals_best = vals_best[np.isfinite(vals_best)]
        rel_best = np.asarray([r.get(f'rel_err_{p}', np.nan) for r in best_rows], dtype=np.float64)
        rel_best = rel_best[np.isfinite(rel_best)]
        summary_best_rows.append(dict(param=p, n=int(vals_best.size), mean_err=float(vals_best.mean()) if vals_best.size else np.nan,
                                      std=float(vals_best.std(ddof=0)) if vals_best.size else np.nan,
                                      mean_rel_err=float(rel_best.mean()) if rel_best.size else np.nan,
                                      std_rel_err=float(rel_best.std(ddof=0)) if rel_best.size else np.nan))

    detail_csv = f'{out_prefix}_detail.csv'
    best_csv = f'{out_prefix}_best_per_target.csv'
    best_trace_csv = f'{out_prefix}_best_per_target_iter_trace.csv'
    summary_csv = f'{out_prefix}_summary_all_restarts.csv'
    summary_best_csv = f'{out_prefix}_summary_best_per_target.csv'
    _save_csv_rows(detail_rows, detail_csv)
    _save_csv_rows(best_rows, best_csv)
    _save_csv_rows(_expand_best_trace_rows(best_rows), best_trace_csv)
    _save_csv_rows(summary_rows, summary_csv)
    _save_csv_rows(summary_best_rows, summary_best_csv)
    _save_json({'targets': [_to_serializable_value(cfg) for cfg in target_cfgs], 'param_ranges': _to_serializable_value(param_ranges)}, f'{out_prefix}_target_cfgs.json')

    try:
        _plot_multitarget_allparam_figures(detail_rows, best_rows, out_prefix, trainable)
    except Exception as e:
        print(f'Plotting skipped due to error: {e}')

    print(f'Saved detail CSV: {detail_csv}')
    print(f'Saved best-per-target CSV: {best_csv}')
    print(f'Saved summary CSV: {summary_csv}')
    print(f'Saved best summary CSV: {summary_best_csv}')
    return detail_rows, best_rows, summary_rows, summary_best_rows


# =============================================================================
# Main (CLI)
# =============================================================================

def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description='parameter recovery with fixed roll0_normal and root_dir')
    ap.add_argument('--out', type=str, default='results_single_param.csv',
                    help='single-parameter output CSV path; figures will be saved to <out>_figs/')
    ap.add_argument('--all_out_prefix', type=str, default='results_multi_param',
                    help='prefix for the multi-parameter recovery outputs')
    ap.add_argument('--restarts', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--n_head', type=int, default=5)
    ap.add_argument('--iters1', type=int, default=200)
    ap.add_argument('--iters2', type=int, default=200)
    ap.add_argument('--warm_rounds', type=int, default=1,
                    help='number of extra warm-start rounds; 0 means only the initial run, 1 means initial run + one warm rerun')
    ap.add_argument('--num_targets', type=int, default=1,
                    help='number of synthetic targets for multi-parameter recovery')
    ap.add_argument('--init_mode', type=str, default='random', choices=['random', 'perturb_target'],
                    help='restart initialization mode for multi-parameter recovery')
    ap.add_argument('--init_perturb_pct', type=float, default=0.20,
                    help='used when init_mode=perturb_target; init range is target +- pct')
    ap.add_argument('--params', type=str,
                    default='r0,gamma,ld_ratio,theta_deg,asymmetry,root_length_mm',
                    help='comma-separated single-parameter list; roll0_normal is automatically excluded')
    ap.add_argument('--all_trainable_params', type=str,
                    default='r0,gamma,ld_ratio,theta_deg,asymmetry,root_length_mm',
                    help='comma-separated parameter set for the joint all-parameter run')
    ap.add_argument('--num_workers', type=int, default=1, help='parallel workers across (target,restart) tasks for all-parameter recovery')
    ap.add_argument('--theta_range', type=str, default='25.0,65.0', help="bounded-tanh range for theta_deg as 'lo,hi'")
    ap.add_argument('--r0_range', type=str, default='5.70,10.90', help="bounded-tanh range for r0 as 'lo,hi'")
    ap.add_argument('--gamma_range', type=str, default='2.50,3.50', help="softplus lower bound + sampling clip for gamma as 'lo,hi'")
    ap.add_argument('--ld_ratio_range', type=str, default='1.00,5.00', help="bounded-tanh range for ld_ratio as 'lo,hi'")
    ap.add_argument('--asymmetry_range', type=str, default='0.05,0.15', help="bounded-tanh range for asymmetry as 'lo,hi'")
    ap.add_argument('--root_length_range', type=str, default='6.0,18.0', help="softplus lower bound + sampling clip for root_length_mm as 'lo,hi'")
    ap.add_argument('--skip_single', action='store_true', help='skip the one-at-a-time recovery run')
    ap.add_argument('--skip_all', action='store_true', help='skip the all-parameter recovery run')
    ap.add_argument('--if_bif', type=str, default='True', choices=['True','False','true','false','1','0'],
                    help='whether to include loss_bif in the loss function and optimization')
    return ap.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = _parse_cli()
    params = [p.strip() for p in args.params.split(',') if p.strip() and p.strip() != 'roll0_normal']
    all_trainable_params = [p.strip() for p in args.all_trainable_params.split(',') if p.strip()]
    param_ranges = {
        'theta_deg': _parse_range_arg(args.theta_range),
        'r0': _parse_range_arg(args.r0_range),
        'gamma': _parse_range_arg(args.gamma_range),
        'ld_ratio': _parse_range_arg(args.ld_ratio_range),
        'asymmetry': _parse_range_arg(args.asymmetry_range),
        'root_length_mm': _parse_range_arg(args.root_length_range),
    }
    print(f'[Range config] {param_ranges}')
    if_bif = str(args.if_bif).lower() in ('true', '1')
    print(f'[Loss config] if_bif={if_bif}')

    if not args.skip_single:
        run_421_single_param_recovery(
            n_restarts=args.restarts,
            params=params,
            seed=args.seed,
            depth=args.depth,
            n_head=args.n_head,
            iters_adam_stage1=args.iters1,
            iters_adam_stage2=args.iters2,
            out_csv=args.out,
            warm_start_rounds=args.warm_rounds,
            param_ranges=param_ranges,
            if_bif=if_bif,
        )

    if not args.skip_all:
        run_421_all_param_recovery(
            n_restarts=args.restarts,
            seed=args.seed,
            depth=args.depth,
            n_head=args.n_head,
            iters_adam_stage1=args.iters1,
            iters_adam_stage2=args.iters2,
            out_prefix=args.all_out_prefix,
            warm_start_rounds=args.warm_rounds,
            num_targets=args.num_targets,
            init_mode=args.init_mode,
            init_perturb_pct=args.init_perturb_pct,
            if_bif=if_bif,
            trainable_params=all_trainable_params,
            num_workers=args.num_workers,
            param_ranges=param_ranges,
        )
