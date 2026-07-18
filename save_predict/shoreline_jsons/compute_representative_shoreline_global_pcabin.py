#!/usr/bin/env python3
"""Compute representative shorelines using one shared PCA frame for all clips."""

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


def clip_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower() != "baseline")


def load_clip_points(clip_dir: Path, max_files: int | None) -> pd.DataFrame | None:
    csvs = sorted(p for p in clip_dir.glob("*_warped.csv") if "baseline" not in p.stem.lower())
    if max_files is not None:
        csvs = csvs[:max_files]
    frames = []
    for csv_path in csvs:
        pts = _load_points(csv_path)
        if pts is not None:
            frames.append(pts)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Rectified CSV root with one subfolder per clip.")
    parser.add_argument("--out", required=True, help="Output root for representative shorelines.")
    parser.add_argument("--bins", type=int, default=260)
    parser.add_argument("--min-points-per-bin", type=int, default=3)
    parser.add_argument("--max-files-per-clip", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--min-coverage", type=float, default=0.45, help="Minimum fraction of global bins kept per clip.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    loaded: list[tuple[Path, pd.DataFrame]] = []
    for clip_dir in clip_dirs(root):
        pts = load_clip_points(clip_dir, args.max_files_per_clip)
        if pts is not None:
            loaded.append((clip_dir, pts))
    if not loaded:
        print("No clip points found.")
        return 1

    all_xy = pd.concat([pts for _, pts in loaded], ignore_index=True)[["X_warped", "Y_warped"]].to_numpy(float)
    center = all_xy.mean(axis=0)
    _, _, vt = np.linalg.svd(all_xy - center, full_matrices=False)
    e1 = vt[0]
    e2 = vt[1]

    all_u = (all_xy - center) @ e1
    edges = np.linspace(float(all_u.min()), float(all_u.max()), args.bins + 1)
    min_bins = max(2, int(args.min_coverage * args.bins))
    count = 0
    summary_rows = []

    for clip_dir, pts in loaded:
        xy = pts[["X_warped", "Y_warped"]].to_numpy(float)
        centered = xy - center
        u = centered @ e1
        v = centered @ e2
        bin_index = np.digitize(u, edges) - 1
        binned = pd.DataFrame({"u": u, "v": v, "bin": bin_index})
        grouped = (
            binned[(binned["bin"] >= 0) & (binned["bin"] < args.bins)]
            .groupby("bin")
            .agg(u=("u", "median"), v=("v", "median"), n=("v", "size"))
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= args.min_points_per_bin].sort_values("u")
        if len(grouped) < min_bins:
            print(f"[skip] {clip_dir.name}: coverage {len(grouped)}/{args.bins} bins")
            continue

        u_out = grouped["u"].to_numpy(float)
        v_out = _smooth(grouped["v"].to_numpy(float), args.smooth_window)
        xy_out = center + np.outer(u_out, e1) + np.outer(v_out, e2)
        # Shared s coordinate: same global alongshore coordinate for all clips.
        s = u_out - float(all_u.min())
        out = pd.DataFrame({
            "s": s,
            "X_warped": xy_out[:, 0],
            "Y_warped": xy_out[:, 1],
            "n_points": grouped["n"].to_numpy(int),
        })

        out_dir = out_root / clip_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "rep_shoreline_world.csv", index=False)
        out[["s", "n_points"]].to_csv(out_dir / "rep_shoreline_sd.csv", index=False)
        summary_rows.append({
            "clip": clip_dir.name,
            "bins": len(grouped),
            "s_min": float(s.min()),
            "s_max": float(s.max()),
            "points": int(grouped["n"].sum()),
        })
        count += 1
        print(f"OK {clip_dir.name} -> {out_dir}")

    if summary_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_root / "global_pcabin_summary.csv", index=False)
        pd.DataFrame({
            "center_x": [center[0]],
            "center_y": [center[1]],
            "e1_x": [e1[0]],
            "e1_y": [e1[1]],
            "e2_x": [e2[0]],
            "e2_y": [e2[1]],
            "u_min": [float(all_u.min())],
            "u_max": [float(all_u.max())],
        }).to_csv(out_root / "global_pcabin_frame.csv", index=False)
    print(f"Wrote {count} representative shoreline(s).")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
