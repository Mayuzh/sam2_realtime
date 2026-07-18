#!/usr/bin/env python3
"""Compute representative shorelines by binning in a PCA alongshore frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_points(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    if "x_warped" not in lower or "y_warped" not in lower:
        return None
    x = pd.to_numeric(df[lower["x_warped"]], errors="coerce")
    y = pd.to_numeric(df[lower["y_warped"]], errors="coerce")
    out = pd.DataFrame({"X_warped": x, "Y_warped": y}).dropna()
    return out if len(out) >= 2 else None


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    if len(values) < window:
        return values
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")
    out[0] = values[0]
    out[-1] = values[-1]
    return out


def process_clip(
    clip_dir: Path,
    out_dir: Path,
    bins: int,
    min_points_per_bin: int,
    max_files: int | None,
    smooth_window: int,
) -> bool:
    csvs = sorted([p for p in clip_dir.glob("*_warped.csv") if "baseline" not in p.stem.lower()])
    if max_files is not None:
        csvs = csvs[:max_files]

    frames = []
    for csv_path in csvs:
        pts = _load_points(csv_path)
        if pts is not None:
            frames.append(pts)
    if not frames:
        print(f"[skip] {clip_dir.name}: no point CSVs")
        return False

    pts = pd.concat(frames, ignore_index=True)
    xy = pts[["X_warped", "Y_warped"]].to_numpy(float)
    center = xy.mean(axis=0)
    centered = xy - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e1 = vt[0]
    e2 = vt[1]
    u = centered @ e1
    v = centered @ e2

    edges = np.linspace(float(u.min()), float(u.max()), bins + 1)
    bin_index = np.digitize(u, edges) - 1
    binned = pd.DataFrame({"u": u, "v": v, "bin": bin_index})
    grouped = (
        binned[(binned["bin"] >= 0) & (binned["bin"] < bins)]
        .groupby("bin")
        .agg(u=("u", "median"), v=("v", "median"), n=("v", "size"))
        .reset_index(drop=True)
    )
    grouped = grouped[grouped["n"] >= min_points_per_bin].sort_values("u")
    if len(grouped) < 2:
        print(f"[skip] {clip_dir.name}: too few populated bins")
        return False

    u_out = grouped["u"].to_numpy(float)
    v_out = _smooth(grouped["v"].to_numpy(float), smooth_window)
    xy_out = center + np.outer(u_out, e1) + np.outer(v_out, e2)
    s = u_out - u_out.min()
    out = pd.DataFrame({
        "s": s,
        "X_warped": xy_out[:, 0],
        "Y_warped": xy_out[:, 1],
        "n_points": grouped["n"].to_numpy(int),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "rep_shoreline_world.csv", index=False)
    out[["s", "n_points"]].to_csv(out_dir / "rep_shoreline_sd.csv", index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Rectified CSV root with one subfolder per clip.")
    parser.add_argument("--out", required=True, help="Output root for representative shorelines.")
    parser.add_argument("--bins", type=int, default=220)
    parser.add_argument("--min-points-per-bin", type=int, default=3)
    parser.add_argument("--max-files-per-clip", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=11)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    count = 0
    for clip_dir in sorted(root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.lower() == "baseline":
            continue
        if process_clip(
            clip_dir,
            out_root / clip_dir.name,
            bins=args.bins,
            min_points_per_bin=args.min_points_per_bin,
            max_files=args.max_files_per_clip,
            smooth_window=args.smooth_window,
        ):
            count += 1
            print(f"OK {clip_dir.name} -> {out_root / clip_dir.name}")
    print(f"Wrote {count} representative shoreline(s).")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
