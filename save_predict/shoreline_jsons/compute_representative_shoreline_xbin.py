#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute one representative shoreline per clip by binning rectified points in X.

This is useful for Seabright-like plan-view shorelines that are mostly a
single-valued Y=f(X) curve. It avoids fragile transect intersections from a
straight baseline on a curved beach.
"""

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


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        return y
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
    out[0] = y[0]
    out[-1] = y[-1]
    return out


def process_clip(clip_dir: Path, out_dir: Path, bins: int, min_points_per_bin: int,
                 max_files: int | None, smooth_window: int) -> bool:
    csvs = sorted([
        p for p in clip_dir.glob("*_warped.csv")
        if "baseline" not in p.stem.lower()
    ])
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
    edges = np.linspace(float(pts.X_warped.min()), float(pts.X_warped.max()), bins + 1)
    pts["bin"] = np.digitize(pts.X_warped, edges) - 1
    grouped = (
        pts[(pts["bin"] >= 0) & (pts["bin"] < bins)]
        .groupby("bin")
        .agg(X_warped=("X_warped", "median"), Y_warped=("Y_warped", "median"), n=("Y_warped", "size"))
        .reset_index(drop=True)
    )
    grouped = grouped[grouped["n"] >= min_points_per_bin].sort_values("X_warped")
    if len(grouped) < 2:
        print(f"[skip] {clip_dir.name}: too few populated bins")
        return False

    y = _smooth(grouped["Y_warped"].to_numpy(float), smooth_window)
    x = grouped["X_warped"].to_numpy(float)
    s = x.copy()
    out = pd.DataFrame({"s": s, "X_warped": x, "Y_warped": y, "n_points": grouped["n"].to_numpy(int)})

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "rep_shoreline_world.csv", index=False)
    out[["s", "n_points"]].to_csv(out_dir / "rep_shoreline_sd.csv", index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Rectified CSV root with one subfolder per clip.")
    parser.add_argument("--out", required=True, help="Output root for representative shorelines.")
    parser.add_argument("--bins", type=int, default=500, help="Number of X bins per clip.")
    parser.add_argument("--min-points-per-bin", type=int, default=3, help="Minimum aggregate points required per bin.")
    parser.add_argument("--max-files-per-clip", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--smooth-window", type=int, default=9, help="Moving-average smoothing window over binned median Y.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    count = 0
    for clip_dir in sorted(root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.lower() == "baseline":
            continue
        out_dir = out_root / clip_dir.name
        if process_clip(
            clip_dir,
            out_dir,
            bins=int(args.bins),
            min_points_per_bin=int(args.min_points_per_bin),
            max_files=args.max_files_per_clip,
            smooth_window=int(args.smooth_window),
        ):
            count += 1
            print(f"OK {clip_dir.name} -> {out_dir}")
    print(f"Wrote {count} representative shoreline(s).")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
