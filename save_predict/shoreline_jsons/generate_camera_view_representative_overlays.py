#!/usr/bin/env python3
"""Plot smooth representative shoreline contours on georectified camera views."""

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


def world_to_pixel(world_file: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, d, b, e, c, f = [float(v.strip()) for v in world_file.read_text().split()]
    transform = np.array([[a, b], [d, e]], dtype=float)
    inv = np.linalg.inv(transform)
    cr = np.column_stack([x - c, y - f]) @ inv.T
    return cr[:, 0], cr[:, 1]


def smooth_curve(x: np.ndarray, y: np.ndarray, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 4:
        return x, y
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, step > 1e-6]
    x = x[keep]
    y = y[keep]
    if len(x) < 4:
        return x, y
    t = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if t[-1] <= 0:
        return x, y
    out_t = np.linspace(0.0, t[-1], 240)
    k = min(3, len(x) - 1)
    try:
        sx = UnivariateSpline(t, x, s=smoothing, k=k)
        sy = UnivariateSpline(t, y, s=smoothing, k=k)
        return sx(out_t), sy(out_t)
    except Exception:
        return x, y


def load_reps(rep_root: Path, water_csv: Path, no_extrapolate: bool) -> list[dict]:
    water_seconds, water_z = load_water_levels(water_csv)
    rows = []
    for rep_path in sorted(rep_root.glob("*/rep_shoreline_world.csv")):
        clip = rep_path.parent.name
        timestamp = parse_clip_timestamp(clip)
        if timestamp is None:
            continue
        tide = tide_at(timestamp, water_seconds, water_z, extrapolate=not no_extrapolate)
        if not np.isfinite(tide):
            continue
        df = pd.read_csv(rep_path)
        lower = {c.lower(): c for c in df.columns}
        x_col = lower.get("x_warped")
        y_col = lower.get("y_warped")
        s_col = lower.get("s")
        if x_col is None or y_col is None:
            continue
        if s_col is not None:
            df = df.sort_values(s_col)
        rows.append({
            "clip": clip,
            "timestamp": timestamp,
            "tide": float(tide),
            "x": pd.to_numeric(df[x_col], errors="coerce").to_numpy(float),
            "y": pd.to_numeric(df[y_col], errors="coerce").to_numpy(float),
        })
    return sorted(rows, key=lambda r: r["tide"])


def select_reps(reps: list[dict], max_lines: int) -> list[dict]:
    if max_lines <= 0 or len(reps) <= max_lines:
        return reps
    idx = np.linspace(0, len(reps) - 1, max_lines).round().astype(int)
    return [reps[i] for i in sorted(set(idx))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-root", required=True)
    parser.add_argument("--water-csv", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--world-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-lines", type=int, default=9)
    parser.add_argument("--smooth", type=float, default=120.0)
    parser.add_argument("--crop-padding", type=float, default=60.0)
    parser.add_argument("--no-extrapolate", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path)
    reps = select_reps(load_reps(Path(args.rep_root), Path(args.water_csv), args.no_extrapolate), args.max_lines)
    if not reps:
        raise SystemExit("No representative shorelines found for overlay.")

    tide_values = np.array([r["tide"] for r in reps], dtype=float)
    norm = plt.Normalize(float(tide_values.min()), float(tide_values.max()))
    cmap = plt.cm.turbo

    curves = []
    all_c = []
    all_r = []
    for rep in reps:
        c, r = world_to_pixel(Path(args.world_file), rep["x"], rep["y"])
        c, r = smooth_curve(c, r, args.smooth)
        inside = (c >= -args.crop_padding) & (c <= image.width + args.crop_padding) & (r >= -args.crop_padding) & (r <= image.height + args.crop_padding)
        c = c[inside]
        r = r[inside]
        if len(c) < 2:
            continue
        curves.append((rep, c, r))
        all_c.extend(c)
        all_r.extend(r)
    if not curves:
        raise SystemExit("Representative shorelines did not overlap the image.")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)
    for rep, c, r in curves:
        color = cmap(norm(rep["tide"]))
        ax.plot(c, r, color=color, linewidth=3.0, alpha=0.95, solid_capstyle="round")

    all_c = np.asarray(all_c)
    all_r = np.asarray(all_r)
    xmin = max(0.0, float(np.nanmin(all_c) - args.crop_padding))
    xmax = min(float(image.width), float(np.nanmax(all_c) + args.crop_padding))
    ymin = max(0.0, float(np.nanmin(all_r) - args.crop_padding))
    ymax = min(float(image.height), float(np.nanmax(all_r) + args.crop_padding))
    if xmax > xmin and ymax > ymin:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
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
