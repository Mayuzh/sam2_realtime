#!/usr/bin/env python3
"""Generate a clean Trevone camera-view overlay from filtered native points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import UnivariateSpline

from build_beach_surface import load_water_levels, parse_clip_timestamp, tide_at

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


def smooth_xy(x: np.ndarray, y: np.ndarray, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique = np.r_[True, np.diff(x) > 1e-6]
    x = x[unique]
    y = y[unique]
    if len(x) < 4:
        return x, y
    out_x = np.linspace(float(x.min()), float(x.max()), 260)
    k = min(3, len(x) - 1)
    try:
        spline = UnivariateSpline(x, y, s=smoothing, k=k)
        return out_x, spline(out_x)
    except Exception:
        return x, y


def session_median_curve(clip_dir: Path, bins: int, min_points_per_bin: int, smoothing: float) -> tuple[np.ndarray, np.ndarray] | None:
    frames = []
    for csv_path in sorted(clip_dir.glob("*_warped.csv")):
        df = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in df.columns}
        if "x" not in lower or "y" not in lower:
            continue
        x = pd.to_numeric(df[lower["x"]], errors="coerce")
        y = pd.to_numeric(df[lower["y"]], errors="coerce")
        pts = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pts):
            frames.append(pts)
    if not frames:
        return None
    pts = pd.concat(frames, ignore_index=True)
    edges = np.linspace(float(pts.x.min()), float(pts.x.max()), bins + 1)
    pts["bin"] = np.digitize(pts.x, edges) - 1
    grouped = (
        pts[(pts["bin"] >= 0) & (pts["bin"] < bins)]
        .groupby("bin")
        .agg(x=("x", "median"), y=("y", "median"), n=("y", "size"))
        .reset_index(drop=True)
    )
    grouped = grouped[grouped["n"] >= min_points_per_bin].sort_values("x")
    if len(grouped) < 4:
        return None
    return smooth_xy(grouped["x"].to_numpy(float), -grouped["y"].to_numpy(float), smoothing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--water-csv", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Trevone representative shorelines by tide level")
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--min-points-per-bin", type=int, default=3)
    parser.add_argument("--smooth", type=float, default=260.0)
    parser.add_argument("--max-lines", type=int, default=9)
    parser.add_argument("--crop-padding", type=float, default=90.0)
    parser.add_argument("--no-extrapolate", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image)
    water_seconds, water_z = load_water_levels(Path(args.water_csv))
    reps = []
    for clip_dir in sorted(Path(args.root).iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.lower() == "baseline":
            continue
        timestamp = parse_clip_timestamp(clip_dir.name)
        if timestamp is None:
            continue
        tide = tide_at(timestamp, water_seconds, water_z, extrapolate=not args.no_extrapolate)
        if not np.isfinite(tide):
            continue
        curve = session_median_curve(clip_dir, args.bins, args.min_points_per_bin, args.smooth)
        if curve is None:
            continue
        reps.append({"clip": clip_dir.name, "tide": float(tide), "curve": curve})
    reps = sorted(reps, key=lambda item: item["tide"])
    if len(reps) > args.max_lines:
        idx = np.linspace(0, len(reps) - 1, args.max_lines).round().astype(int)
        reps = [reps[i] for i in sorted(set(idx))]
    if not reps:
        raise SystemExit("No Trevone curves found.")

    tide_values = np.array([r["tide"] for r in reps])
    norm = plt.Normalize(float(tide_values.min()), float(tide_values.max()))
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)
    all_x = []
    all_y = []
    for rep in reps:
        x, y = rep["curve"]
        ax.plot(x, y, color=cmap(norm(rep["tide"])), linewidth=3.2, alpha=0.95, solid_capstyle="round")
        all_x.extend(x)
        all_y.extend(y)
    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    ax.set_xlim(max(0, all_x.min() - args.crop_padding), min(image.width, all_x.max() + args.crop_padding))
    ax.set_ylim(min(image.height, all_y.max() + args.crop_padding), max(0, all_y.min() - args.crop_padding))
    ax.set_title(args.title)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Tide height (m)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
