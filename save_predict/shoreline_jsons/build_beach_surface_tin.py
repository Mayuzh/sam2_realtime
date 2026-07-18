#!/usr/bin/env python3
"""Build a 2.5D beach surface TIN from representative shoreline contours."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from build_beach_surface import (
    _col,
    _read_csv,
    load_water_levels,
    parse_clip_timestamp,
    tide_at,
    write_obj,
    write_ply,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_points(rep_root: Path, water_seconds: np.ndarray, water_z_m: np.ndarray,
                extrapolate: bool) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    vertices = []
    for rep_path in sorted(rep_root.glob("*/rep_shoreline_world.csv")):
        clip = rep_path.parent.name
        timestamp = parse_clip_timestamp(clip)
        if timestamp is None:
            print(f"[skip] Could not parse timestamp from {clip}")
            continue
        z = tide_at(timestamp, water_seconds, water_z_m, extrapolate)
        if not np.isfinite(z):
            print(f"[skip] No tide value for {clip}")
            continue

        df = _read_csv(rep_path)
        x_col = _col(df, ["x_warped", "x", "X_warped", "X"])
        y_col = _col(df, ["y_warped", "y", "Y_warped", "Y"])
        if x_col is None or y_col is None:
            print(f"[skip] Missing X/Y columns: {rep_path}")
            continue
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < 2:
            continue
        start_idx = len(vertices)
        for i, (xi, yi) in enumerate(zip(x, y)):
            vertices.append((xi, yi, z))
            rows.append({
                "vertex_index": start_idx + i,
                "clip_name": clip,
                "timestamp_utc": timestamp.isoformat(),
                "X_warped": xi,
                "Y_warped": yi,
                "Z_m": z,
                "point_index": i,
                "rep_csv": str(rep_path),
            })
    if not vertices:
        raise ValueError("No representative shoreline points loaded.")
    return np.asarray(vertices, dtype=float), pd.DataFrame(rows)


def filter_faces(vertices: np.ndarray, simplices: np.ndarray, max_edge: float | None,
                 max_z_span: float | None) -> np.ndarray:
    if max_edge is None and max_z_span is None:
        return simplices.astype(np.int64)
    keep = []
    xy = vertices[:, :2]
    z = vertices[:, 2]
    for tri in simplices:
        tri_xy = xy[tri]
        edges = [
            np.linalg.norm(tri_xy[0] - tri_xy[1]),
            np.linalg.norm(tri_xy[1] - tri_xy[2]),
            np.linalg.norm(tri_xy[2] - tri_xy[0]),
        ]
        if max_edge is not None and max(edges) > max_edge:
            continue
        if max_z_span is not None and float(z[tri].max() - z[tri].min()) > max_z_span:
            continue
        keep.append(tuple(int(i) for i in tri))
    return np.asarray(keep, dtype=np.int64)


def write_preview(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(faces):
        ax.triplot(vertices[:, 0], vertices[:, 1], faces, color="0.75", linewidth=0.4)
    scatter = ax.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], s=6, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="Z_m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("TIN surface footprint from representative shoreline points")
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
    parser.add_argument("--no-extrapolate", action="store_true")
    parser.add_argument("--max-edge-m", type=float, default=35.0,
                        help="Drop triangles with an XY edge longer than this many meters.")
    parser.add_argument("--max-z-span-m", type=float, default=None,
                        help="Drop triangles whose vertex Z range exceeds this many meters.")
    args = parser.parse_args()

    water_seconds, water_z_m = load_water_levels(Path(args.water_csv))
    vertices, point_table = load_points(
        Path(args.rep_root),
        water_seconds,
        water_z_m,
        extrapolate=not args.no_extrapolate,
    )
    tri = Delaunay(vertices[:, :2])
    faces = filter_faces(vertices, tri.simplices, args.max_edge_m, args.max_z_span_m)
    if len(faces) == 0:
        raise ValueError("All TIN faces were filtered out. Relax --max-edge-m or --max-z-span-m.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "beach_surface.ply", vertices, faces)
    write_obj(out_dir / "beach_surface.obj", vertices, faces)
    point_table.to_csv(out_dir / "all_representative_shoreline_points_3d.csv", index=False)
    write_preview(out_dir / "surface_plan_preview.png", vertices, faces)

    xy_span = math.hypot(
        float(vertices[:, 0].max() - vertices[:, 0].min()),
        float(vertices[:, 1].max() - vertices[:, 1].min()),
    )
    diagnostics = pd.DataFrame({
        "metric": [
            "points",
            "faces",
            "tide_min_m",
            "tide_max_m",
            "tide_span_m",
            "plan_bbox_diagonal_m",
            "max_edge_filter_m",
            "max_z_span_filter_m",
        ],
        "value": [
            len(vertices),
            len(faces),
            float(vertices[:, 2].min()),
            float(vertices[:, 2].max()),
            float(vertices[:, 2].max() - vertices[:, 2].min()),
            xy_span,
            float(args.max_edge_m) if args.max_edge_m is not None else np.nan,
            float(args.max_z_span_m) if args.max_z_span_m is not None else np.nan,
        ],
    })
    diagnostics.to_csv(out_dir / "surface_diagnostics.csv", index=False)
    print(f"Wrote {out_dir / 'beach_surface.ply'}")
    print(f"Wrote {out_dir / 'beach_surface.obj'}")
    print(f"TIN points: {len(vertices)}; faces: {len(faces)}; tide span: {vertices[:, 2].max() - vertices[:, 2].min():.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
