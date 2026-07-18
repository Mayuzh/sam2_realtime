#!/usr/bin/env python3
"""Build a smoothed gridded 2.5D beach surface from representative shoreline points."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

from build_beach_surface import write_obj, write_ply
from build_beach_surface_tin import load_points
from build_beach_surface import load_water_levels

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def fill_small_holes(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = z.copy()
    rows, cols = np.where(~np.isfinite(out) & valid)
    for r, c in zip(rows, cols):
        r0 = max(0, r - 1)
        r1 = min(out.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(out.shape[1], c + 2)
        vals = out[r0:r1, c0:c1]
        finite = vals[np.isfinite(vals)]
        if finite.size:
            out[r, c] = float(np.nanmean(finite))
    return out


def smooth_grid(z: np.ndarray, valid: np.ndarray, iterations: int) -> np.ndarray:
    out = z.copy()
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="edge")
        smoothed = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
        out = np.where(valid & np.isfinite(out), smoothed, out)
    return out


def build_grid_faces(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = -np.ones(valid.shape, dtype=int)
    index[valid] = np.arange(int(valid.sum()))
    faces = []
    for r in range(valid.shape[0] - 1):
        for c in range(valid.shape[1] - 1):
            a = index[r, c]
            b = index[r + 1, c]
            d = index[r, c + 1]
            e = index[r + 1, c + 1]
            if a >= 0 and b >= 0 and e >= 0:
                faces.append((a, b, e))
            if a >= 0 and e >= 0 and d >= 0:
                faces.append((a, e, d))
    return index, np.asarray(faces, dtype=np.int64)


def write_preview(path: Path, xg: np.ndarray, yg: np.ndarray, z: np.ndarray, vertices: np.ndarray) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.pcolormesh(xg, yg, z, shading="auto", cmap="viridis")
    ax.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], s=4, cmap="viridis", edgecolors="none", alpha=0.45)
    fig.colorbar(image, ax=ax, label="Z_m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Smoothed gridded beach surface")
    ax.set_xlabel("X_warped")
    ax.set_ylabel("Y_warped")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-root", required=True)
    parser.add_argument("--water-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grid-spacing-m", type=float, default=1.5)
    parser.add_argument("--method", choices=["cubic", "linear"], default="cubic")
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--x-min", type=float, default=None, help="Optional output grid minimum X.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional output grid maximum X.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional output grid minimum Y.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional output grid maximum Y.")
    parser.add_argument(
        "--extrapolate-grid",
        action="store_true",
        help="Fill the whole requested grid using nearest-neighbour fallback outside the point hull.",
    )
    parser.add_argument("--no-extrapolate", action="store_true")
    args = parser.parse_args()

    water_seconds, water_z_m = load_water_levels(Path(args.water_csv))
    points, point_table = load_points(
        Path(args.rep_root),
        water_seconds,
        water_z_m,
        extrapolate=not args.no_extrapolate,
    )
    xy = points[:, :2]
    z = points[:, 2]

    spacing = float(args.grid_spacing_m)
    data_min = xy.min(axis=0)
    data_max = xy.max(axis=0)
    xmin = float(data_min[0] if args.x_min is None else args.x_min)
    ymin = float(data_min[1] if args.y_min is None else args.y_min)
    xmax = float(data_max[0] if args.x_max is None else args.x_max)
    ymax = float(data_max[1] if args.y_max is None else args.y_max)
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    xg, yg = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xg.ravel(), yg.ravel()])

    hull = Delaunay(xy)
    inside = hull.find_simplex(grid_xy) >= 0
    if args.method == "cubic":
        interp = CloughTocher2DInterpolator(xy, z)
    else:
        interp = LinearNDInterpolator(xy, z)
    zg = np.asarray(interp(grid_xy), dtype=float).reshape(xg.shape)

    if not np.isfinite(zg).any():
        interp = LinearNDInterpolator(xy, z)
        zg = np.asarray(interp(grid_xy), dtype=float).reshape(xg.shape)
    nearest = NearestNDInterpolator(xy, z)
    if args.extrapolate_grid:
        missing = ~np.isfinite(zg)
        if missing.any():
            zg[missing] = nearest(np.column_stack([xg[missing], yg[missing]]))
        valid = np.isfinite(zg)
    else:
        missing_inside = inside.reshape(xg.shape) & ~np.isfinite(zg)
        if missing_inside.any():
            zg[missing_inside] = nearest(np.column_stack([xg[missing_inside], yg[missing_inside]]))
        valid = inside.reshape(xg.shape) & np.isfinite(zg)
    zg = fill_small_holes(zg, valid)
    zg = smooth_grid(zg, valid, args.smooth_iterations)
    valid = valid & np.isfinite(zg)

    index, faces = build_grid_faces(valid)
    rows, cols = np.where(valid)
    vertices = np.column_stack([xg[rows, cols], yg[rows, cols], zg[rows, cols]])
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("No grid surface could be built from the representative points.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "beach_surface.ply", vertices, faces)
    write_obj(out_dir / "beach_surface.obj", vertices, faces)
    point_table.to_csv(out_dir / "source_representative_shoreline_points_3d.csv", index=False)
    grid_df = pd.DataFrame(vertices, columns=["X_warped", "Y_warped", "Z_m"])
    grid_df.to_csv(out_dir / "gridded_surface_vertices.csv", index=False)
    write_preview(out_dir / "surface_plan_preview.png", xg, yg, np.where(valid, zg, np.nan), points)

    xy_span = math.hypot(float(xmax - xmin), float(ymax - ymin))
    diagnostics = pd.DataFrame({
        "metric": [
            "source_points",
            "grid_vertices",
            "faces",
            "tide_min_m",
            "tide_max_m",
            "tide_span_m",
            "plan_bbox_diagonal_m",
            "grid_spacing_m",
            "interpolation_method",
            "smooth_iterations",
            "extrapolate_grid",
        ],
        "value": [
            len(points),
            len(vertices),
            len(faces),
            float(z.min()),
            float(z.max()),
            float(z.max() - z.min()),
            xy_span,
            spacing,
            args.method,
            args.smooth_iterations,
            bool(args.extrapolate_grid),
        ],
    })
    diagnostics.to_csv(out_dir / "surface_diagnostics.csv", index=False)
    print(f"Wrote {out_dir / 'beach_surface.ply'}")
    print(f"Wrote {out_dir / 'beach_surface.obj'}")
    print(f"Grid vertices: {len(vertices)}; faces: {len(faces)}; tide span: {z.max() - z.min():.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
