#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

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


def main():
    ap = argparse.ArgumentParser(description="Warp CSV points using projective (homography) or TPS transform with ArcGIS-style control points.")
    ap.add_argument("--points", default="./csv/twinlakes/13/", help="Input points CSV or a folder of CSVs (each must contain columns x,y).")
    ap.add_argument("--links", default="./links/control_points2.txt", help="Control points file: ArcGIS link table (.txt) or generic CSV.")
    ap.add_argument("--out", default="./csv/twinlakes_rec/13/", help="Output warped CSV file, or an output folder when --points is a folder.")
    ap.add_argument("--method", choices=["projective", "tps"], default="tps", help="Transformation method: projective homography or thin-plate spline (default: projective).")
    ap.add_argument("--smooth", type=float, default=0.0, help="Smoothing (lambda) for TPS. Increase slightly (e.g. 1e-3) if TPS system is near-singular.")
    ap.add_argument("--y-down", action="store_true", help="Indicates source/control pixel coords are in a Y-down system (origin top-left). They will NOT be flipped; this flag only controls how --image-height flipping is interpreted.")
    ap.add_argument("--image-height", type=int, default=None, help="If provided WITH --flip-to-yup, used to convert y_down to y_up via (H-1 - y).")
    ap.add_argument("--flip-to-yup", action="store_true", help="Convert Y-down pixel coords to math Y-up before fitting: y_up = (H-1 - y_down). Provide --image-height.")
    ap.add_argument("--report", action="store_true", help="Print RMS residuals on control points.")
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
    if args.method == "tps":
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

    # Determine batch or single mode
    points_path = Path(args.points)
    out_path = Path(args.out)
    points_files = _gather_points_files(points_path)

    # If multiple inputs, treat output as directory
    batch_mode = len(points_files) > 1 or points_path.is_dir() or (out_path.exists() and out_path.is_dir()) or (out_path.suffix.lower() != ".csv")

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
                H = args.image_height
                py = (H - 1) - py
            if transform_payload[0] == "tps":
                _, model_x, model_y = transform_payload
                query = np.stack([px, py], axis=1)
                Xw = eval_tps(model_x, query)
                Yw = eval_tps(model_y, query)
            else:
                _, H = transform_payload
                Xw, Yw = _eval_projective(H, px, py)
            pts["X_warped"] = Xw
            pts["Y_warped"] = Yw
            out_csv = out_dir / f"{csv_in.stem}_warped.csv"
            pts.to_csv(out_csv, index=False)
            total += len(pts)
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
        else:
            _, H = transform_payload
            Xw, Yw = _eval_projective(H, px, py)
        pts["X_warped"] = Xw
        pts["Y_warped"] = Yw
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        pts.to_csv(args.out, index=False)
        print(f"✅ Wrote warped CSV: {args.out} (columns added: X_warped, Y_warped)")

if __name__ == "__main__":
    main()
