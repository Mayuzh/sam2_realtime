#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the deepest (most central) shoreline per 10-min clip using a curve boxplot workflow.

Given a root like seabright_new_rec with:
- baseline_warped.csv (offshore straight baseline in world coordinates)
- subfolders per clip, each containing shoreline CSV files with X_warped, Y_warped

For each clip folder:
 1) Sample the baseline every --spacing meters
 2) Build shore-normal transects of length --length meters (positive seaward)
 3) Intersect each shoreline polyline with all transects to get cross-shore distance d(s)
 4) Drop sparse curves (< --min-coverage coverage) and fill small gaps by interpolation
 5) Compute a curve boxplot: median, quartiles, whiskers (min/max ignoring NaNs)
 6) Compute Modified Band Depth (MBD) and select the deepest curve
 7) Export deepest as (s,d) and back-projected world coords; save PNG plot per clip

Usage:
  python compute_deepest_shoreline.py --root ./csv/seabright_new_rec \
      --baseline ./csv/seabright_new_rec/baseline_warped.csv \
      --out ./csv/seabright_new_rec/averaged --spacing 4.0 --length 200 \
      --min-coverage 0.8 --max-gap 12 --invert-normal

Notes:
- Assumes baseline and shorelines are in the same projected CRS (meters).
- Shoreline CSVs must contain columns X_warped, Y_warped and be ordered along the polyline.
- If a CSV contains multiple polylines, the longest by total length is used.
"""

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------
# Geometry helpers
# ---------------------

def _polyline_length(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))


def _resample_polyline(x: np.ndarray, y: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a polyline at approximately uniform spacing.
    Returns (xs, ys, s) where s is cumulative arc-length (starts at 0).
    """
    if len(x) < 2:
        raise ValueError("Polyline too short to resample")
    # Cumulative length
    dx = np.diff(x)
    dy = np.diff(y)
    seglen = np.hypot(dx, dy)
    clen = np.concatenate([[0.0], np.cumsum(seglen)])
    total = clen[-1]
    if total <= 0:
        raise ValueError("Degenerate baseline (zero length)")
    n_samples = max(2, int(math.floor(total / spacing)) + 1)
    s_target = np.linspace(0.0, total, n_samples)
    # Interpolate
    xs = np.interp(s_target, clen, x)
    ys = np.interp(s_target, clen, y)
    return xs, ys, s_target


def _tangent_normals(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit tangent and unit normal at each sample (forward differences + padding)."""
    tx = np.gradient(xs)
    ty = np.gradient(ys)
    tnorm = np.hypot(tx, ty)
    tnorm[tnorm == 0] = 1.0
    tx /= tnorm
    ty /= tnorm
    # Left-hand normal (rotate +90°): (-ty, tx)
    nx = -ty
    ny = tx
    return np.stack([tx, ty], axis=1), np.stack([nx, ny], axis=1)


def _nearest_index_xy(xs: np.ndarray, ys: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """For each (px,py) returns index of nearest baseline sample point."""
    # KD-tree would be faster, but this is fine for modest sizes
    bx = xs[None, :]
    by = ys[None, :]
    d2 = (px[:, None] - bx) ** 2 + (py[:, None] - by) ** 2
    return np.argmin(d2, axis=1)


def _segment_intersect(p: np.ndarray, r: np.ndarray, q: np.ndarray, s: np.ndarray) -> Optional[Tuple[float, float]]:
    """Intersect segment p->p+r with q->q+s.
    Returns (t, u) such that p + t r = q + u s, with t,u in [0,1]; else None.
    """
    rxs = r[0] * s[1] - r[1] * s[0]
    q_p = q - p
    q_pxr = q_p[0] * r[1] - q_p[1] * r[0]
    if abs(rxs) < 1e-12:
        # Parallel or colinear: treat as no single intersection
        return None
    t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
    u = q_pxr / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


def _polyline_intersections_with_transect(poly_x: np.ndarray, poly_y: np.ndarray,
                                           P: np.ndarray, nvec: np.ndarray, L: float) -> List[float]:
    """Return list of signed distances along nvec from P to all intersections with shoreline polyline.
    Only the forward half-segment is used (P to P + L * nvec).
    """
    r = nvec * L
    hits: List[float] = []
    for i in range(len(poly_x) - 1):
        A = np.array([poly_x[i], poly_y[i]], dtype=float)
        B = np.array([poly_x[i+1], poly_y[i+1]], dtype=float)
        q = A
        svec = B - A
        inter = _segment_intersect(P, r, q, svec)
        if inter is None:
            continue
        t, u = inter
        # Distance along the transect
        hits.append(float(t * L))
    return hits


# ---------------------
# Depth and boxplot
# ---------------------

def _mbd(curves: np.ndarray) -> np.ndarray:
    """Compute Modified Band Depth (MBD) efficiently.

    For each station m, let n_m be the number of finite values, and r_k(m) the average rank of curve k
    among those n_m values (1..n_m). The fraction of bands at station m containing x_k is:
        ((r_k - 1) * (n_m - r_k)) / C(n_m, 2)
    MBD is the mean of this fraction over stations where curve k is finite.

    curves: (N, M) array with NaNs allowed.
    Returns: depth (N,) in [0,1].
    """
    N, M = curves.shape
    depth = np.zeros(N, dtype=float)
    counts = np.zeros(N, dtype=float)  # number of stations contributing for each curve

    for m in range(M):
        col = curves[:, m]
        finite = np.isfinite(col)
        idx = np.nonzero(finite)[0]
        n = int(idx.size)
        if n < 2:
            continue
        vals = col[idx]
        # Stable sort to help tie handling determinism
        order = np.argsort(vals, kind="mergesort")
        v_sorted = vals[order]
        # Compute average ranks for ties
        ranks = np.empty(n, dtype=float)
        start = 0
        while start < n:
            end = start
            while end + 1 < n and v_sorted[end + 1] == v_sorted[start]:
                end += 1
            # average rank for this tie group (1-based ranks)
            avg_rank = 0.5 * ((start + 1) + (end + 1))
            ranks[start:end + 1] = avg_rank
            start = end + 1
        # Map ranks back to original indices
        r_full = np.empty(n, dtype=float)
        r_full[order] = ranks
        # Per-curve contribution at this station
        denom = n * (n - 1) / 2.0
        if denom <= 0:
            continue
        contrib = ((r_full - 1.0) * (n - r_full)) / denom
        # Accumulate
        depth[idx] += contrib
        counts[idx] += 1.0

    # Normalize by number of contributing stations
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(counts > 0, depth / counts, 0.0)
    return depth


def _pointwise_stats(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, q25, q75, min, max) along stations, ignoring NaNs."""
    med = np.nanmedian(curves, axis=0)
    q25 = np.nanpercentile(curves, 25, axis=0)
    q75 = np.nanpercentile(curves, 75, axis=0)
    vmin = np.nanmin(curves, axis=0)
    vmax = np.nanmax(curves, axis=0)
    return med, q25, q75, vmin, vmax


def _interpolate_small_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps whose length <= max_gap samples."""
    y = y.copy()
    isnan = ~np.isfinite(y)
    if not np.any(isnan):
        return y
    idx = np.arange(len(y))
    # Identify contiguous NaN runs
    in_gap = False
    start = 0
    for i, nan in enumerate(isnan.tolist() + [False]):  # sentinel
        if nan and not in_gap:
            in_gap = True
            start = i
        elif not nan and in_gap:
            end = i - 1
            gap_len = end - start + 1
            in_gap = False
            if gap_len <= max_gap and start > 0 and end < len(y) - 1 and np.isfinite(y[start-1]) and np.isfinite(y[end+1]):
                y[start:end+1] = np.interp(idx[start:end+1], [start-1, end+1], [y[start-1], y[end+1]])
    return y


# ---------------------
# IO helpers
# ---------------------

def _load_polyline_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if not {"x_warped", "y_warped"}.issubset(lower.keys()):
        raise ValueError(f"CSV missing X_warped/Y_warped columns: {path}")
    x = df[lower["x_warped"]].to_numpy(float)
    y = df[lower["y_warped"]].to_numpy(float)
    # If multiple features exist, attempt to keep the longest contiguous
    if "feature_id" in lower:
        # group by feature and pick the one with max length
        best_len = -1.0
        best_xy = (x, y)
        for fid, g in df.groupby(lower["feature_id"]):
            gx = g[lower["x_warped"]].to_numpy(float)
            gy = g[lower["y_warped"]].to_numpy(float)
            L = _polyline_length(gx, gy)
            if L > best_len:
                best_len = L
                best_xy = (gx, gy)
        return best_xy
    return x, y


# ---------------------
# Main processing
# ---------------------

def process_clip(clip_dir: Path, baseline_xy: Tuple[np.ndarray, np.ndarray], spacing: float, length: float,
                 min_coverage: float, max_gap_samples: int, invert_normal: bool,
                 out_dir: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Path]]:
    """Process a single clip directory; returns (sd_df, world_df, fig_path) or None if no valid curves."""
    bx, by = baseline_xy
    bxs, bys, s = _resample_polyline(bx, by, spacing)
    tang, norms = _tangent_normals(bxs, bys)
    if invert_normal:
        norms = -norms

    # Collect shoreline polylines in this clip
    csvs = sorted([p for p in clip_dir.glob("*.csv") if p.name.endswith("_warped.csv")])
    curves: List[np.ndarray] = []
    names: List[str] = []

    # Heuristic: determine seaward side by comparing median distances
    # Compute centroid of all shoreline points (first pass)
    all_pts = []
    for csv in csvs:
        try:
            x, y = _load_polyline_csv(csv)
            all_pts.append(np.stack([x, y], axis=1))
        except Exception:
            continue
    if not all_pts:
        print(f"[skip] No shoreline polylines in {clip_dir}")
        return None
    all_pts = np.concatenate(all_pts, axis=0)
    # nearest baseline index per shoreline point
    idx = _nearest_index_xy(bxs, bys, all_pts[:, 0], all_pts[:, 1])
    vecs = all_pts - np.stack([bxs[idx], bys[idx]], axis=1)
    sign_a = np.median(np.sum(vecs * norms[idx], axis=1))
    # choose outward sign to make distances positive toward shoreline cluster
    if sign_a < 0:
        norms = -norms

    # Build d(s) for each shoreline
    L = float(length)
    for csv in csvs:
        try:
            x, y = _load_polyline_csv(csv)
        except Exception as e:
            print(f"[warn] {e}")
            continue
        dvals = np.full_like(s, np.nan, dtype=float)
        for i in range(len(s)):
            P = np.array([bxs[i], bys[i]], dtype=float)
            nvec = norms[i]
            hits = _polyline_intersections_with_transect(x, y, P, nvec, L)
            if not hits:
                continue
            d = min(hits)  # closest positive along transect
            if d >= 0:
                dvals[i] = d
        # coverage
        cov = np.mean(np.isfinite(dvals)) if dvals.size > 0 else 0.0
        if cov < min_coverage:
            continue
        # fill small gaps
        dvals = _interpolate_small_gaps(dvals, max_gap_samples)
        curves.append(dvals)
        names.append(csv.name)

    if not curves:
        print(f"[skip] No curves passed coverage filter in {clip_dir}")
        return None

    D = np.vstack(curves)  # (N, M)
    depth = _mbd(D)
    k_star = int(np.nanargmax(depth))
    deepest = D[k_star]

    med, q25, q75, vmin, vmax = _pointwise_stats(D)

    # Back-project deepest into world coords: P_i + d_i * n_i
    Pw = np.stack([bxs, bys], axis=1) + (deepest[:, None] * norms)

    # Output dataframes
    sd_df = pd.DataFrame({
        "s": s,
        "d": deepest
    })
    world_df = pd.DataFrame({
        "s": s,
        "X_warped": Pw[:, 0],
        "Y_warped": Pw[:, 1]
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # all curves faint
    for i in range(D.shape[0]):
        ax.plot(s, D[i], color=(0, 0, 0, 0.15), linewidth=1)
    # 50% band
    ax.fill_between(s, q25, q75, color=(0.2, 0.5, 1.0, 0.25), label="IQR (50%)")
    # whiskers
    ax.plot(s, vmin, color=(0.6, 0.6, 0.6, 1.0), linewidth=1, label="Whisker (min/max)")
    ax.plot(s, vmax, color=(0.6, 0.6, 0.6, 1.0), linewidth=1)
    # deepest bold
    ax.plot(s, deepest, color=(0.85, 0.1, 0.1, 1.0), linewidth=2.5, label="Deepest (MBD)")
    ax.set_xlabel("Alongshore station s (m)")
    ax.set_ylabel("Cross-shore distance d (m, + seaward)")
    ax.set_title(f"Curve Boxplot: {clip_dir.name} (N={D.shape[0]})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{clip_dir.name}_boxplot.png"
    csv_sd_path = out_dir / f"{clip_dir.name}_deepest_sd.csv"
    csv_world_path = out_dir / f"{clip_dir.name}_deepest_world.csv"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    sd_df.to_csv(csv_sd_path, index=False)
    world_df.to_csv(csv_world_path, index=False)

    return sd_df, world_df, fig_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute the deepest shoreline curve per clip using curve boxplot + MBD.")
    ap.add_argument("--root", required=True, help="Root folder with baseline_warped.csv and clip subfolders.")
    ap.add_argument("--baseline", default=None, help="Path to baseline_warped.csv; default: <root>/baseline_warped.csv")
    ap.add_argument("--out", default=None, help="Output folder; default: <root>/averaged")
    ap.add_argument("--spacing", type=float, default=4.0, help="Baseline sampling spacing in meters (default 4.0).")
    ap.add_argument("--length", type=float, default=200.0, help="Transect forward length in meters (default 200).")
    ap.add_argument("--min-coverage", type=float, default=0.8, help="Min fraction of stations with valid intersections (default 0.8).")
    ap.add_argument("--max-gap", type=int, default=12, help="Max gap length (in samples) to fill by interpolation (default 12 samples).")
    ap.add_argument("--invert-normal", action="store_true", help="Force normal direction flip (if seaward detection seems wrong).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    baseline_path = Path(args.baseline) if args.baseline else (root / "baseline_warped.csv")
    if not baseline_path.exists():
        raise SystemExit(f"Baseline CSV not found: {baseline_path}")
    out_root = Path(args.out) if args.out else (root / "averaged")

    # Load baseline
    bdf = pd.read_csv(baseline_path)
    lower = {c.lower(): c for c in bdf.columns}
    if not {"x_warped", "y_warped"}.issubset(lower.keys()):
        raise SystemExit("baseline_warped.csv must contain X_warped,Y_warped columns")
    bx = bdf[lower["x_warped"]].to_numpy(float)
    by = bdf[lower["y_warped"]].to_numpy(float)

    # Discover clips: subfolders containing *_warped.csv files
    clip_dirs: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            if any(child.name.endswith("_warped.csv") for child in p.glob("*.csv")):
                clip_dirs.append(p)

    if not clip_dirs:
        print("No clip subfolders found with *_warped.csv files.")
        return

    print(f"Found {len(clip_dirs)} clip folder(s). Processing…")
    for clip in clip_dirs:
        out_dir = out_root / clip.name
        try:
            process_clip(
                clip_dir=clip,
                baseline_xy=(bx, by),
                spacing=float(args.spacing),
                length=float(args.length),
                min_coverage=float(args.min_coverage),
                max_gap_samples=int(args.max_gap),
                invert_normal=bool(args.invert_normal),
                out_dir=out_dir,
            )
            print(f"✅ {clip.name} → {out_dir}")
        except Exception as e:
            print(f"❌ {clip.name}: {e}")


if __name__ == "__main__":
    main()
