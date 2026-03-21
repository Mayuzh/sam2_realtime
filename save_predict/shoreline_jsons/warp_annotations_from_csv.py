#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist

def _read_any_table(path):
    # Try flexible parsing (handles commas, tabs, multiple spaces)
    return pd.read_csv(path, sep=None, engine="python", comment="#", header=None)

def load_control_points(path):
    """
    Load control points from:
      - ArcGIS link table (columns: sourceX, sourceY, mapX, mapY)
      - Generic CSV (columns: src_x, src_y, dst_x, dst_y)
      - Headerless 4-column numeric file: [src_x, src_y, dst_x, dst_y]
    """
    # First try with header inference
    try:
        df_h = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        df_h = None

    if df_h is not None:
        cols = {c.lower(): c for c in df_h.columns}
        # ArcGIS link table with headers
        if {"sourcex", "sourcey", "mapx", "mapy"}.issubset(cols.keys()):
            sx = df_h[cols["sourcex"]].to_numpy(float)
            sy = df_h[cols["sourcey"]].to_numpy(float)
            dx = df_h[cols["mapx"]].to_numpy(float)
            dy = df_h[cols["mapy"]].to_numpy(float)
            return sx, sy, dx, dy
        # Generic headered CSV
        if {"src_x", "src_y", "dst_x", "dst_y"}.issubset(cols.keys()):
            sx = df_h[cols["src_x"]].to_numpy(float)
            sy = df_h[cols["src_y"]].to_numpy(float)
            dx = df_h[cols["dst_x"]].to_numpy(float)
            dy = df_h[cols["dst_y"]].to_numpy(float)
            return sx, sy, dx, dy

    # Fallback: headerless file with exactly 4 numeric columns
    df = _read_any_table(path)
    if df.shape[1] != 4:
        raise ValueError(
            "Unrecognized control-point schema. Expected headered "
            "(sourceX,sourceY,mapX,mapY) or (src_x,src_y,dst_x,dst_y), "
            "or a headerless 4-column file [src_x, src_y, dst_x, dst_y]."
        )

    # Ensure numeric
    for i in range(4):
        df[i] = pd.to_numeric(df[i], errors="raise")
    sx, sy, dx, dy = (df[0].to_numpy(float),
                      df[1].to_numpy(float),
                      df[2].to_numpy(float),
                      df[3].to_numpy(float))
    return sx, sy, dx, dy

def _phi(r: np.ndarray):
    # Thin-plate spline radial basis: r^2 * log(r), define phi(0)=0
    with np.errstate(divide='ignore', invalid='ignore'):
        out = r*r * np.where(r>0, np.log(r), 0.0)
    out[r==0] = 0.0
    return out

def fit_tps_explicit(src_x, src_y, dst_x, dst_y, smooth=0.0):
    """Solve classic TPS system explicitly to mimic ArcGIS Spline behavior.

    Returns coefficients (w_x, a_x) and (w_y, a_y) where
        f(x,y) = a0 + a1*x + a2*y + sum_i w_i * phi(||(x,y)-(x_i,y_i)||)
    """
    src_x = np.asarray(src_x, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_x = np.asarray(dst_x, dtype=float)
    dst_y = np.asarray(dst_y, dtype=float)
    n = src_x.shape[0]
    pts = np.stack([src_x, src_y], axis=1)
    # Pairwise distances
    diff = pts[:,None,:] - pts[None,:,:]  # (n,n,2)
    r = np.linalg.norm(diff, axis=2)      # (n,n)
    K = _phi(r)
    if smooth>0:
        K += np.eye(n)*smooth
    P = np.concatenate([np.ones((n,1)), src_x[:,None], src_y[:,None]], axis=1)  # (n,3)
    # Build linear system
    A = np.zeros((n+3, n+3), dtype=float)
    A[:n,:n] = K
    A[:n,n:] = P
    A[n:,:n] = P.T
    # Right-hand sides
    bx = np.zeros(n+3, dtype=float); bx[:n] = dst_x
    by = np.zeros(n+3, dtype=float); by[:n] = dst_y
    # Solve
    try:
        solx = np.linalg.solve(A, bx)
        soly = np.linalg.solve(A, by)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"TPS system solve failed (singular). Try adding --smooth > 0. Error: {e}")
    w_x, a_x = solx[:n], solx[n:]
    w_y, a_y = soly[:n], soly[n:]
    return (w_x, a_x), (w_y, a_y), pts

def eval_tps(model, query_pts):
    (w, a), pts = model
    src_pts = pts  # (n,2)
    q = np.asarray(query_pts, dtype=float)  # (m,2)
    diff = q[:,None,:] - src_pts[None,:,:]  # (m,n,2)
    r = np.linalg.norm(diff, axis=2)        # (m,n)
    Kq = _phi(r)                            # (m,n)
    # f = a0 + a1*x + a2*y + sum_i w_i phi_i
    ones = np.ones((q.shape[0],1))
    Pq = np.concatenate([ones, q], axis=1)  # (m,3)
    return (Pq @ a) + (Kq @ w)

# -----------------------------
# Affine (1st order polynomial) Transform Helpers
# -----------------------------
def _fit_affine(src_x, src_y, dst_x, dst_y):
    """Fit affine transformation (1st order polynomial) using least squares.
    
    X = a0 + a1*x + a2*y
    Y = b0 + b1*x + b2*y
    
    Requires at least 3 control points.
    Returns 2x3 affine matrix [[a1, a2, a0], [b1, b2, b0]].
    """
    src_x = np.asarray(src_x, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_x = np.asarray(dst_x, dtype=float)
    dst_y = np.asarray(dst_y, dtype=float)
    n = src_x.shape[0]
    if n < 3:
        raise ValueError("Affine transform requires at least 3 control points.")
    
    # Build design matrix [x, y, 1] for each point
    A = np.column_stack([src_x, src_y, np.ones(n)])
    
    # Solve for X transformation: X = a0 + a1*x + a2*y
    coeffs_x, residuals_x, rank_x, s_x = np.linalg.lstsq(A, dst_x, rcond=None)
    
    # Solve for Y transformation: Y = b0 + b1*x + b2*y
    coeffs_y, residuals_y, rank_y, s_y = np.linalg.lstsq(A, dst_y, rcond=None)
    
    if rank_x < 3 or rank_y < 3:
        print(f"Warning: affine matrix rank deficient (rank_x={rank_x}, rank_y={rank_y})")
    
    # Return as 2x3 matrix: [[a1, a2, a0], [b1, b2, b0]]
    affine_matrix = np.array([
        [coeffs_x[0], coeffs_x[1], coeffs_x[2]],
        [coeffs_y[0], coeffs_y[1], coeffs_y[2]]
    ], dtype=float)
    
    return affine_matrix

def _eval_affine(affine_matrix, x, y):
    """Apply affine transformation to points.
    
    Args:
        affine_matrix: 2x3 matrix [[a1, a2, a0], [b1, b2, b0]]
        x, y: input coordinates (arrays or scalars)
    
    Returns:
        X, Y: transformed coordinates
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # X = a1*x + a2*y + a0
    X = affine_matrix[0, 0] * x + affine_matrix[0, 1] * y + affine_matrix[0, 2]
    
    # Y = b1*x + b2*y + b0
    Y = affine_matrix[1, 0] * x + affine_matrix[1, 1] * y + affine_matrix[1, 2]
    
    return X, Y

# -----------------------------
# Projective (Homography) Transform Helpers
# -----------------------------
def _fit_projective(src_x, src_y, dst_x, dst_y):
    src_x = np.asarray(src_x, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_x = np.asarray(dst_x, dtype=float)
    dst_y = np.asarray(dst_y, dtype=float)
    n = src_x.shape[0]
    if n < 4:
        raise ValueError("Projective transform requires at least 4 control points.")
    A = np.zeros((2*n, 8), dtype=float)
    b = np.zeros(2*n, dtype=float)
    for i, (x, y, X, Y) in enumerate(zip(src_x, src_y, dst_x, dst_y)):
        A[2*i, 0] = x; A[2*i, 1] = y; A[2*i, 2] = 1.0
        A[2*i, 6] = -X * x; A[2*i, 7] = -X * y; b[2*i] = X
        A[2*i+1, 3] = x; A[2*i+1, 4] = y; A[2*i+1, 5] = 1.0
        A[2*i+1, 6] = -Y * x; A[2*i+1, 7] = -Y * y; b[2*i+1] = Y
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        print(f"Warning: homography matrix rank deficient (rank={rank})")
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=float)
    return H

def _eval_projective(H, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones_like(x)
    pts = np.stack([x, y, ones], axis=0)  # (3,N)
    warped = H @ pts
    w = warped[2]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    X = warped[0] / w_safe
    Y = warped[1] / w_safe
    return X, Y

def _gather_points_files(points_path: Path):
    if points_path.is_file() and points_path.suffix.lower() == ".csv":
        return [points_path]
    if points_path.is_dir():
        return sorted([p for p in points_path.glob("*.csv")])
    raise FileNotFoundError(f"Could not find CSV(s) at {points_path}")


def _find_csv_dirs(root: Path) -> Dict[Path, List[Path]]:
    """Recursively find directories under root that directly contain .csv files.

    Returns: mapping of directory path -> sorted list of CSV files in that directory.
    """
    result: Dict[Path, List[Path]] = {}
    if not root.exists():
        return result
    for dirpath, _, filenames in os.walk(root):
        csvs = [Path(dirpath) / fn for fn in filenames if fn.lower().endswith('.csv')]
        if csvs:
            result[Path(dirpath)] = sorted(csvs)
    return result


# -----------------------------
# Smoothing and Filtering Functions
# -----------------------------
def smooth_moving_average(x, y, window=5):
    """Apply moving average smoothing to x, y coordinates."""
    if len(x) < window:
        return x, y
    x_smooth = np.convolve(x, np.ones(window)/window, mode='same')
    y_smooth = np.convolve(y, np.ones(window)/window, mode='same')
    return x_smooth, y_smooth


def smooth_savgol(x, y, window=11, polyorder=3):
    """Apply Savitzky-Golay filter for smoothing."""
    if len(x) < window:
        window = len(x) if len(x) % 2 == 1 else len(x) - 1
        if window < polyorder + 2:
            return x, y
    x_smooth = savgol_filter(x, window, polyorder)
    y_smooth = savgol_filter(y, window, polyorder)
    return x_smooth, y_smooth


def smooth_spline(x, y, s=None, k=3):
    """Fit a spline through the points and resample.
    
    Args:
        x, y: coordinates
        s: smoothing factor (if None, uses len(x))
        k: spline degree (1=linear, 2=quadratic, 3=cubic)
    """
    if len(x) < k + 1:
        return x, y
    # Create parametric representation using cumulative distance
    t = np.zeros(len(x))
    t[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.cumsum(t)
    
    if s is None:
        s = len(x)
    
    try:
        spl_x = UnivariateSpline(t, x, s=s, k=k)
        spl_y = UnivariateSpline(t, y, s=s, k=k)
        x_smooth = spl_x(t)
        y_smooth = spl_y(t)
        return x_smooth, y_smooth
    except Exception as e:
        print(f"Warning: Spline smoothing failed: {e}. Returning original points.")
        return x, y


def remove_outliers_zscore(x, y, threshold=3.0):
    """Remove points that are outliers based on distance from neighbors.
    
    Uses z-score of point-to-point distances to identify outliers.
    """
    if len(x) < 3:
        return x, y
    
    # Compute distances between consecutive points
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    # Z-score of distances
    if len(dists) > 0 and np.std(dists) > 0:
        z_scores = np.abs((dists - np.mean(dists)) / np.std(dists))
        # Keep points where both incoming and outgoing distances are reasonable
        keep_mask = np.ones(len(x), dtype=bool)
        # Mark points with abnormally large distances
        for i in range(len(z_scores)):
            if z_scores[i] > threshold:
                # Remove the point after the large jump
                if i + 1 < len(keep_mask):
                    keep_mask[i + 1] = False
        
        x_filtered = x[keep_mask]
        y_filtered = y[keep_mask]
        return x_filtered, y_filtered
    
    return x, y


def remove_outliers_ransac(x, y, max_trials=100, residual_threshold=None):
    """Remove outliers using RANSAC-like approach fitting a polynomial."""
    if len(x) < 10:
        return x, y
    
    # Create parametric representation
    t = np.arange(len(x))
    
    if residual_threshold is None:
        # Auto-set threshold based on data spread
        residual_threshold = np.std(np.sqrt(np.diff(x)**2 + np.diff(y)**2)) * 2
    
    # Fit polynomial
    try:
        poly_x = np.polyfit(t, x, deg=3)
        poly_y = np.polyfit(t, y, deg=3)
        x_fit = np.polyval(poly_x, t)
        y_fit = np.polyval(poly_y, t)
        
        # Compute residuals
        residuals = np.sqrt((x - x_fit)**2 + (y - y_fit)**2)
        
        # Keep inliers
        inlier_mask = residuals < residual_threshold
        return x[inlier_mask], y[inlier_mask]
    except:
        return x, y


def resample_uniform(x, y, num_points=None, spacing=None):
    """Resample points uniformly along the curve.
    
    Args:
        x, y: input coordinates
        num_points: number of output points (mutually exclusive with spacing)
        spacing: desired spacing between points
    """
    if len(x) < 2:
        return x, y
    
    # Compute cumulative distance
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_dist[-1]
    
    # Determine sample points
    if spacing is not None:
        num_points = int(total_length / spacing) + 1
    elif num_points is None:
        num_points = len(x)
    
    # Interpolate uniformly
    t_uniform = np.linspace(0, total_length, num_points)
    x_uniform = np.interp(t_uniform, cum_dist, x)
    y_uniform = np.interp(t_uniform, cum_dist, y)
    
    return x_uniform, y_uniform


def _to_pca_frame(X, Y):
    """Project 2D points into PCA frame (u alongshore, v cross-shore)."""
    pts = np.column_stack([X, Y])
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e1 = vt[0]
    e2 = vt[1]
    u = centered @ e1
    v = centered @ e2
    return u, v, center, e1, e2


def _from_pca_frame(u, v, center, e1, e2):
    """Map points from PCA frame back to XY coordinates."""
    pts = center + np.outer(u, e1) + np.outer(v, e2)
    return pts[:, 0], pts[:, 1]


def apply_smoothing_pipeline(Xw, Yw, args, method_name):
    """Apply the complete smoothing and filtering pipeline to warped coordinates.
    
    Args:
        Xw, Yw: warped coordinates (numpy arrays)
        args: parsed command-line arguments
    
    Returns:
        X_smooth, Y_smooth, keeps_row_alignment
    """
    X, Y = Xw.copy(), Yw.copy()
    keeps_row_alignment = True
    used_pca_frame = False

    frame = args.smooth_frame
    if frame == "auto":
        frame = "pca" if method_name == "projective" else "index"

    # In PCA frame, sort by alongshore coordinate so filters act along the shoreline,
    # which avoids zig-zag stretching for projective outputs.
    if frame == "pca" and len(X) >= 3:
        u, v, center, e1, e2 = _to_pca_frame(X, Y)
        order = np.argsort(u)
        X = u[order]
        Y = v[order]
        keeps_row_alignment = False
        used_pca_frame = True
    
    # Step 1: Remove outliers (before smoothing)
    if args.remove_outliers == "zscore":
        X, Y = remove_outliers_zscore(X, Y, threshold=args.outlier_threshold)
        print(f"  Outlier removal (z-score): {len(Xw)} -> {len(X)} points")
    elif args.remove_outliers == "ransac":
        X, Y = remove_outliers_ransac(X, Y, residual_threshold=args.outlier_threshold)
        print(f"  Outlier removal (RANSAC): {len(Xw)} -> {len(X)} points")
    
    # Step 2: Apply smoothing
    if args.smooth_method == "moving_average":
        X, Y = smooth_moving_average(X, Y, window=args.smooth_window)
        print(f"  Smoothing: moving average (window={args.smooth_window})")
    elif args.smooth_method == "savgol":
        X, Y = smooth_savgol(X, Y, window=args.smooth_window, polyorder=args.smooth_polyorder)
        print(f"  Smoothing: Savitzky-Golay (window={args.smooth_window}, order={args.smooth_polyorder})")
    elif args.smooth_method == "spline":
        s = args.smooth_spline_s if args.smooth_spline_s is not None else len(X)
        X, Y = smooth_spline(X, Y, s=s, k=3)
        print(f"  Smoothing: spline (s={s})")
    
    # Step 3: Resample uniformly
    if args.resample:
        X, Y = resample_uniform(X, Y, num_points=args.resample_points, spacing=args.resample_spacing)
        spacing_info = f"spacing={args.resample_spacing}" if args.resample_spacing else f"n={args.resample_points or len(X)}"
        print(f"  Resampling: {spacing_info} -> {len(X)} points")

    if used_pca_frame and len(X) >= 2:
        X, Y = _from_pca_frame(X, Y, center, e1, e2)
    
    return X, Y, keeps_row_alignment


def main():
    ap = argparse.ArgumentParser(description="Warp CSV points using projective (homography) or TPS. Supports recursive input root with mirrored output structure.")
    ap.add_argument("--points", default="./csv/trevone/", help="Input CSV file OR a ROOT folder to scan recursively for folders that directly contain CSVs (with columns x,y).")
    ap.add_argument("--links", default="./links/control_points_trevone2.txt", help="Control points file: ArcGIS link table (.txt) or generic CSV.")
    ap.add_argument("--out", default="./csv/trevone_rec2/", help="Output ROOT folder (for directory input) or output CSV file (for single-file input). When points is a directory, the directory tree is mirrored under this root.")
    ap.add_argument("--method", choices=["affine", "projective", "tps"], default="affine", help="Transformation method: affine (1st order polynomial, min 3 pts), projective homography (min 4 pts), or thin-plate spline (default: projective).")
    ap.add_argument("--smooth", type=float, default=0.0, help="Smoothing (lambda) for TPS. Increase slightly (e.g. 1e-3) if TPS system is near-singular.")
    ap.add_argument("--y-down", action="store_true", help="Indicates source/control pixel coords are in a Y-down system (origin top-left). They will NOT be flipped; this flag only controls how --image-height flipping is interpreted.")
    ap.add_argument("--image-height", type=int, default=None, help="If provided WITH --flip-to-yup, used to convert y_down to y_up via (H-1 - y).")
    ap.add_argument("--flip-to-yup", action="store_true", help="Convert Y-down pixel coords to math Y-up before fitting: y_up = (H-1 - y_down). Provide --image-height.")
    ap.add_argument("--report", action="store_true", help="Print RMS residuals on control points.")
    
    # Smoothing and filtering options
    ap.add_argument("--smooth-method", choices=["none", "moving_average", "savgol", "spline"], default="none", 
                    help="Post-warp smoothing method: moving_average, savgol (Savitzky-Golay), spline, or none.")
    ap.add_argument("--smooth-window", type=int, default=11, help="Window size for moving_average and savgol smoothing (default: 11).")
    ap.add_argument("--smooth-polyorder", type=int, default=3, help="Polynomial order for savgol filter (default: 3).")
    ap.add_argument("--smooth-spline-s", type=float, default=None, help="Smoothing factor for spline (if None, uses data length).")
    ap.add_argument("--remove-outliers", choices=["none", "zscore", "ransac"], default="none",
                    help="Remove outliers before smoothing: zscore (distance-based), ransac (polynomial fit), or none.")
    ap.add_argument("--outlier-threshold", type=float, default=3.0, 
                    help="Threshold for outlier removal: z-score threshold (default: 3.0) or residual distance.")
    ap.add_argument("--resample", action="store_true", help="Resample points uniformly along the curve after smoothing.")
    ap.add_argument("--resample-points", type=int, default=None, help="Number of points for resampling (if not set, keeps original count).")
    ap.add_argument("--resample-spacing", type=float, default=None, help="Spacing between resampled points in output units (overrides --resample-points).")
    ap.add_argument("--smooth-frame", choices=["auto", "index", "pca"], default="auto",
                    help="Coordinate frame for smoothing. 'auto' uses PCA for projective and index-order for affine/TPS.")
    
    args = ap.parse_args()

    # Load control points
    src_x, src_y, dst_x, dst_y = load_control_points(args.links)

    # Optionally convert to Y-up using image height (correct conversion vs simple negation)
    if args.flip_to_yup:
        if args.image_height is None:
            raise SystemExit("--flip-to-yup requires --image-height <H>.")
        H = args.image_height
        src_y = (H - 1) - src_y

    # Fit transform
    transform_payload = None
    if args.method == "affine":
        # Affine (1st order polynomial)
        affine_matrix = _fit_affine(src_x, src_y, dst_x, dst_y)
        if args.report:
            Xc, Yc = _eval_affine(affine_matrix, src_x, src_y)
            rms_x = math.sqrt(np.mean((Xc - dst_x)**2))
            rms_y = math.sqrt(np.mean((Yc - dst_y)**2))
            print(f"[Affine] Control residual RMS: X={rms_x:.4f}  Y={rms_y:.4f}")
        transform_payload = ("affine", affine_matrix)
    elif args.method == "tps":
        (w_x, a_x), (w_y, a_y), ctrl_pts = fit_tps_explicit(src_x, src_y, dst_x, dst_y, smooth=args.smooth)
        model_x = ((w_x, a_x), ctrl_pts)
        model_y = ((w_y, a_y), ctrl_pts)
        if args.report:
            pred_x = eval_tps(model_x, ctrl_pts)
            pred_y = eval_tps(model_y, ctrl_pts)
            rms_x = math.sqrt(np.mean((pred_x - dst_x)**2))
            rms_y = math.sqrt(np.mean((pred_y - dst_y)**2))
            print(f"[TPS] Control residual RMS: X={rms_x:.4f}  Y={rms_y:.4f}")
        transform_payload = ("tps", model_x, model_y)
    else:
        # Projective homography
        H = _fit_projective(src_x, src_y, dst_x, dst_y)
        if args.report:
            Xc, Yc = _eval_projective(H, src_x, src_y)
            rms_x = math.sqrt(np.mean((Xc - dst_x)**2))
            rms_y = math.sqrt(np.mean((Yc - dst_y)**2))
            print(f"[Projective] Control residual RMS: X={rms_x:.4f}  Y={rms_y:.4f}")
        transform_payload = ("projective", H)

    # Determine mode and paths
    points_path = Path(args.points)
    out_path = Path(args.out)

    # Case 1: points is a directory -> recursive scan and mirror structure
    if points_path.is_dir():
        dir_to_csvs = _find_csv_dirs(points_path)
        if not dir_to_csvs:
            print(f"No CSV files found under input root: {points_path}")
            return
        out_root = out_path
        out_root.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        total_files = 0
        touched_dirs = 0
        for csv_dir, csv_list in sorted(dir_to_csvs.items()):
            rel = csv_dir.relative_to(points_path)
            out_dir = out_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            touched_dirs += 1
            for csv_in in csv_list:
                pts = pd.read_csv(csv_in)
                lower = {c.lower(): c for c in pts.columns}
                if "x" not in lower or "y" not in lower:
                    print(f"Skipping {csv_in}: missing x/y columns")
                    continue
                xcol, ycol = lower["x"], lower["y"]
                px = pts[xcol].to_numpy(float)
                py = pts[ycol].to_numpy(float)
                if args.flip_to_yup:
                    if args.image_height is None:
                        raise SystemExit("--flip-to-yup requires --image-height (same as used for control points).")
                    H_img = args.image_height
                    py = (H_img - 1) - py
                if transform_payload[0] == "tps":
                    _, model_x, model_y = transform_payload
                    query = np.stack([px, py], axis=1)
                    Xw = eval_tps(model_x, query)
                    Yw = eval_tps(model_y, query)
                elif transform_payload[0] == "affine":
                    _, affine_matrix = transform_payload
                    Xw, Yw = _eval_affine(affine_matrix, px, py)
                else:
                    _, Hm = transform_payload
                    Xw, Yw = _eval_projective(Hm, px, py)
                
                # Apply smoothing pipeline
                if args.smooth_method != "none" or args.remove_outliers != "none" or args.resample:
                    Xw, Yw, keeps_alignment = apply_smoothing_pipeline(Xw, Yw, args, transform_payload[0])
                else:
                    keeps_alignment = True
                
                # Handle length mismatch if smoothing changed point count
                if (len(Xw) != len(pts)) or (not keeps_alignment):
                    # Create new dataframe with only warped coordinates
                    pts_out = pd.DataFrame({"X_warped": Xw, "Y_warped": Yw})
                else:
                    # Keep original columns and add warped coordinates
                    pts["X_warped"] = Xw
                    pts["Y_warped"] = Yw
                    pts_out = pts
                
                out_csv = out_dir / f"{csv_in.stem}_warped.csv"
                pts_out.to_csv(out_csv, index=False)
                total_rows += len(pts_out)
                total_files += 1
        print(f"✅ Wrote {total_files} file(s) across {touched_dirs} folder(s) under: {str(out_root)} (total rows: {total_rows})")
        return

    # Case 2: Single-file or flat-folder fallback
    points_files = _gather_points_files(points_path)
    # If multiple inputs, treat output as directory in flat mode
    batch_mode = len(points_files) > 1 or (out_path.exists() and out_path.is_dir()) or (out_path.suffix.lower() != ".csv")

    if batch_mode:
        out_dir = out_path if out_path.suffix == "" or out_path.is_dir() else out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for csv_in in points_files:
            pts = pd.read_csv(csv_in)
            lower = {c.lower(): c for c in pts.columns}
            if "x" not in lower or "y" not in lower:
                print(f"Skipping {csv_in}: missing x/y columns")
                continue
            xcol, ycol = lower["x"], lower["y"]
            px = pts[xcol].to_numpy(float)
            py = pts[ycol].to_numpy(float)
            if args.flip_to_yup:
                if args.image_height is None:
                    raise SystemExit("--flip-to-yup requires --image-height (same as used for control points).")
                H_img = args.image_height
                py = (H_img - 1) - py
            if transform_payload[0] == "tps":
                _, model_x, model_y = transform_payload
                query = np.stack([px, py], axis=1)
                Xw = eval_tps(model_x, query)
                Yw = eval_tps(model_y, query)
            elif transform_payload[0] == "affine":
                _, affine_matrix = transform_payload
                Xw, Yw = _eval_affine(affine_matrix, px, py)
            else:
                _, Hm = transform_payload
                Xw, Yw = _eval_projective(Hm, px, py)
            
            # Apply smoothing pipeline
            if args.smooth_method != "none" or args.remove_outliers != "none" or args.resample:
                Xw, Yw, keeps_alignment = apply_smoothing_pipeline(Xw, Yw, args, transform_payload[0])
            else:
                keeps_alignment = True
            
            # Handle length mismatch if smoothing changed point count
            if (len(Xw) != len(pts)) or (not keeps_alignment):
                # Create new dataframe with only warped coordinates
                pts_out = pd.DataFrame({"X_warped": Xw, "Y_warped": Yw})
            else:
                # Keep original columns and add warped coordinates
                pts["X_warped"] = Xw
                pts["Y_warped"] = Yw
                pts_out = pts
            
            out_csv = out_dir / f"{csv_in.stem}_warped.csv"
            pts_out.to_csv(out_csv, index=False)
            total += len(pts_out)
        print(f"✅ Wrote {len(points_files)} file(s) to {str(out_dir)} (total rows: {total})")
    else:
        # Single file mode
        csv_in = points_files[0]
        pts = pd.read_csv(csv_in)
        lower = {c.lower(): c for c in pts.columns}
        if "x" not in lower or "y" not in lower:
            raise ValueError("Points CSV must contain columns 'x' and 'y' (case-insensitive).")
        xcol, ycol = lower["x"], lower["y"]
        px = pts[xcol].to_numpy(float)
        py = pts[ycol].to_numpy(float)
        if args.flip_to_yup:
            if args.image_height is None:
                raise SystemExit("--flip-to-yup requires --image-height (same as used for control points).")
            H = args.image_height
            py = (H - 1) - py
        if transform_payload[0] == "tps":
            _, model_x, model_y = transform_payload
            query = np.stack([px, py], axis=1)
            Xw = eval_tps(model_x, query)
            Yw = eval_tps(model_y, query)
        elif transform_payload[0] == "affine":
            _, affine_matrix = transform_payload
            Xw, Yw = _eval_affine(affine_matrix, px, py)
        else:
            _, H = transform_payload
            Xw, Yw = _eval_projective(H, px, py)
        
        # Apply smoothing pipeline
        if args.smooth_method != "none" or args.remove_outliers != "none" or args.resample:
            Xw, Yw, keeps_alignment = apply_smoothing_pipeline(Xw, Yw, args, transform_payload[0])
        else:
            keeps_alignment = True
        
        # Handle length mismatch if smoothing changed point count
        if (len(Xw) != len(pts)) or (not keeps_alignment):
            # Create new dataframe with only warped coordinates
            pts_out = pd.DataFrame({"X_warped": Xw, "Y_warped": Yw})
        else:
            # Keep original columns and add warped coordinates
            pts["X_warped"] = Xw
            pts["Y_warped"] = Yw
            pts_out = pts
        
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    pts_out.to_csv(args.out, index=False)
    print(f"✅ Wrote warped CSV: {args.out} (columns added: X_warped, Y_warped)")

if __name__ == "__main__":
    main()
