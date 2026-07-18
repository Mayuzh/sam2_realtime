#!/usr/bin/env python3
"""Plot representative shoreline contours in native camera image coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import UnivariateSpline

from build_beach_surface import load_water_levels, parse_clip_timestamp, tide_at

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


def load_warp_module():
    path = Path(__file__).with_name("warp_annotations_from_csv.py")
    spec = importlib.util.spec_from_file_location("warp_annotations_from_csv", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    out_t = np.linspace(0.0, t[-1], 260)
    k = min(3, len(x) - 1)
    try:
        sx = UnivariateSpline(t, x, s=smoothing, k=k)
        sy = UnivariateSpline(t, y, s=smoothing, k=k)
        return sx(out_t), sy(out_t)
    except Exception:
        return x, y


def fit_inverse_transform(method: str, control_points: Path):
    warp = load_warp_module()
    sx, sy, dx, dy = warp.load_control_points(control_points)
    if method == "projective":
        h = warp._fit_projective(sx, sy, dx, dy)
        h_inv = np.linalg.inv(h)

        def inverse(x_map: np.ndarray, y_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return warp._eval_projective(h_inv, x_map, y_map)

        return inverse
    if method == "affine":
        matrix = warp._fit_affine(sx, sy, dx, dy)
        full = np.array([
            [matrix[0, 0], matrix[0, 1], matrix[0, 2]],
            [matrix[1, 0], matrix[1, 1], matrix[1, 2]],
            [0.0, 0.0, 1.0],
        ])
        inv = np.linalg.inv(full)

        def inverse(x_map: np.ndarray, y_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pts = np.column_stack([x_map, y_map, np.ones_like(x_map)])
            out = pts @ inv.T
            return out[:, 0], out[:, 1]

        return inverse
    raise ValueError(f"Unsupported method: {method}")


def load_reps(rep_root: Path, water_csv: Path, no_extrapolate: bool) -> list[dict]:
    water_seconds, water_z = load_water_levels(water_csv)
    reps = []
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
        reps.append({
            "clip": clip,
            "tide": float(tide),
            "x": pd.to_numeric(df[x_col], errors="coerce").to_numpy(float),
            "y": pd.to_numeric(df[y_col], errors="coerce").to_numpy(float),
        })
    return sorted(reps, key=lambda item: item["tide"])


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
    parser.add_argument("--control-points", required=True)
    parser.add_argument("--method", choices=["projective", "affine"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-lines", type=int, default=9)
    parser.add_argument("--smooth", type=float, default=120.0)
    parser.add_argument("--crop-padding", type=float, default=80.0)
    parser.add_argument("--no-extrapolate", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image)
    inverse = fit_inverse_transform(args.method, Path(args.control_points))
    reps = select_reps(load_reps(Path(args.rep_root), Path(args.water_csv), args.no_extrapolate), args.max_lines)
    if not reps:
        raise SystemExit("No representative shorelines found for overlay.")

    tide_values = np.array([r["tide"] for r in reps])
    norm = plt.Normalize(float(tide_values.min()), float(tide_values.max()))
    cmap = plt.cm.turbo
    curves = []
    all_x = []
    all_y = []
    for rep in reps:
        src_x, src_y = inverse(rep["x"], rep["y"])
        col = src_x
        row = -src_y
        col, row = smooth_curve(col, row, args.smooth)
        inside = (
            (col >= -args.crop_padding) &
            (col <= image.width + args.crop_padding) &
            (row >= -args.crop_padding) &
            (row <= image.height + args.crop_padding)
        )
        col = col[inside]
        row = row[inside]
        if len(col) < 2:
            continue
        curves.append((rep, col, row))
        all_x.extend(col)
        all_y.extend(row)
    if not curves:
        raise SystemExit("No representative shorelines overlap the camera image.")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)
    for rep, col, row in curves:
        ax.plot(col, row, color=cmap(norm(rep["tide"])), linewidth=3.0, alpha=0.95, solid_capstyle="round")

    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    xmin = max(0.0, float(np.nanmin(all_x) - args.crop_padding))
    xmax = min(float(image.width), float(np.nanmax(all_x) + args.crop_padding))
    ymin = max(0.0, float(np.nanmin(all_y) - args.crop_padding))
    ymax = min(float(image.height), float(np.nanmax(all_y) + args.crop_padding))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
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
