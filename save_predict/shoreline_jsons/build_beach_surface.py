#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a 3D intertidal surface from representative shoreline contours.

Inputs:
- an averaged/representative root with one subfolder per clip, each containing
  rep_shoreline_world.csv from compute_deepest_shoreline.py
- a water-level file, either CO-OPS-style feet columns or Newlyn-style
  "Time GMT" with tide height in metres to ACD

Each representative shoreline is treated as a horizontal contour whose Z value
is the tide-gauge water level at the clip timestamp. Curves are resampled onto a
common alongshore s grid before meshing, which avoids the thin/warped ribbon
that can happen when each shoreline is parameterized by its own point index.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


FEET_TO_M = 0.3048


@dataclass
class ShorelineContour:
    clip: str
    timestamp: datetime
    tide_m: float
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    d: np.ndarray | None
    path: Path


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=False)
    except Exception:
        return pd.read_csv(path, engine="python", index_col=False)


def parse_clip_timestamp(name: str) -> datetime | None:
    patterns = [
        r"(?P<date>\d{4}-\d{2}-\d{2})[-_](?P<hms>\d{6})Z",
        r"(?P<date>\d{4}-\d{2}-\d{2})[-_](?P<hhmm>\d{4})Z",
        r"(?P<ymd>\d{8})[_-](?P<hms>\d{6})",
        r"(?P<ymd>\d{8})[_-](?P<hhmm>\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        parts = match.groupdict()
        if parts.get("date"):
            date_text = parts["date"]
        else:
            ymd = parts["ymd"]
            date_text = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        time_text = parts.get("hms") or f"{parts['hhmm']}00"
        return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H%M%S").replace(tzinfo=timezone.utc)
    return None


def load_water_levels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = _read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    date_col = _col(df, ["date"])
    time_col = _col(df, ["time (gmt)", "time", "time gmt"])
    datetime_col = _col(df, ["datetime (gmt)", "datetime", "date time (gmt)", "time gmt"])
    if date_col and time_col:
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), utc=True, errors="coerce")
    elif datetime_col:
        dt = pd.to_datetime(df[datetime_col].astype(str), format="%Y/%m/%d %H:%M", utc=True, errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(df[datetime_col].astype(str), utc=True, errors="coerce")
    else:
        raise ValueError(f"Could not find Date/Time columns in {path}")

    meter_cols = [
        _col(df, ["tide height in metres to acd", "tide height metres to acd", "water level (m)", "water level m"]),
    ]
    level_cols = [
        _col(df, ["verified (ft)", "verified", "verified(ft)"]),
        _col(df, ["preliminary (ft)", "preliminary", "preliminary(ft)"]),
        _col(df, ["predicted (ft)", "predicted", "predicted(ft)"]),
    ]
    level = pd.Series(np.nan, index=df.index, dtype=float)
    units = None
    for level_col in meter_cols:
        if level_col is not None:
            level = level.fillna(pd.to_numeric(df[level_col], errors="coerce"))
            units = "m"
    for level_col in level_cols:
        if level_col is not None:
            level = level.fillna(pd.to_numeric(df[level_col], errors="coerce"))
            if units is None:
                units = "ft"

    if units == "m":
        z_m = level
    else:
        z_m = level * FEET_TO_M

    out = pd.DataFrame({"dt": dt, "z_m": z_m}).dropna().sort_values("dt")
    out = out.drop_duplicates("dt", keep="last")
    if len(out) < 2:
        raise ValueError(f"Water-level CSV has fewer than two usable rows: {path}")

    seconds = out["dt"].astype("int64").to_numpy(dtype=float) / 1e9
    return seconds, out["z_m"].to_numpy(float)


def tide_at(timestamp: datetime, water_seconds: np.ndarray, water_z_m: np.ndarray, extrapolate: bool) -> float:
    ts = timestamp.timestamp()
    if not extrapolate and (ts < water_seconds.min() or ts > water_seconds.max()):
        return float("nan")
    return float(np.interp(ts, water_seconds, water_z_m))


def _arc_length_parameter(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.arange(len(x), dtype=float)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.linspace(0.0, 1.0, len(x))
    return s


def load_contours(root: Path, water_seconds: np.ndarray, water_z_m: np.ndarray, extrapolate: bool) -> list[ShorelineContour]:
    contours: list[ShorelineContour] = []
    for rep_path in sorted(root.glob("*/rep_shoreline_world.csv")):
        clip_dir = rep_path.parent
        timestamp = parse_clip_timestamp(clip_dir.name)
        if timestamp is None:
            print(f"[skip] Could not parse timestamp from {clip_dir.name}")
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
        if finite.sum() < 2:
            print(f"[skip] Too few finite points: {rep_path}")
            continue
        x = x[finite]
        y = y[finite]

        s_col = _col(df, ["s"])
        if s_col is not None:
            s_all = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
            s = s_all[finite]
        else:
            s = _arc_length_parameter(x, y)

        d = None
        d_col = _col(df, ["d"])
        if d_col is not None:
            d_all = pd.to_numeric(df[d_col], errors="coerce").to_numpy(float)
            d = d_all[finite]

        order = np.argsort(s)
        s = s[order]
        x = x[order]
        y = y[order]
        d = d[order] if d is not None else None
        unique = np.r_[True, np.diff(s) > 1e-9]
        s = s[unique]
        x = x[unique]
        y = y[unique]
        d = d[unique] if d is not None else None
        if len(s) < 2:
            print(f"[skip] Degenerate s grid: {rep_path}")
            continue

        tide_m = tide_at(timestamp, water_seconds, water_z_m, extrapolate)
        if not np.isfinite(tide_m):
            print(f"[skip] No tide value for {clip_dir.name}")
            continue

        contours.append(ShorelineContour(clip_dir.name, timestamp, tide_m, s, x, y, d, rep_path))
    return contours


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends = np.where(np.diff(padded.astype(int)) == -1)[0]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    keep = np.zeros_like(mask, dtype=bool)
    keep[starts[best]:ends[best]] = True
    return keep


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _window_weight(values: np.ndarray, start: float | None, end: float | None, side: str) -> np.ndarray:
    """Smooth 0..1 spatial weight over X values."""
    if start is None and end is None:
        return np.zeros_like(values, dtype=float)
    x = np.asarray(values, dtype=float)
    if side == "left":
        if end is None:
            end = float(np.nanmin(x))
        if start is None:
            start = end - 1.0
        denom = max(end - start, 1e-9)
        return 1.0 - _smoothstep((x - start) / denom)
    if side == "right":
        if start is None:
            start = float(np.nanmax(x))
        if end is None:
            end = start + 1.0
        denom = max(end - start, 1e-9)
        return _smoothstep((x - start) / denom)
    if start is None or end is None:
        return np.zeros_like(values, dtype=float)
    center = 0.5 * (start + end)
    half = max(0.5 * abs(end - start), 1e-9)
    return np.clip(1.0 - np.abs((x - center) / half), 0.0, 1.0)


def apply_footprint_adjustments(verts_grid: np.ndarray, tide_values: np.ndarray,
                                args: argparse.Namespace) -> np.ndarray:
    """Apply small map-space Y adjustments to the final footprint."""
    adjusted = verts_grid.copy()
    if adjusted.size == 0:
        return adjusted

    tide_min = float(np.nanmin(tide_values))
    tide_span = max(float(np.nanmax(tide_values) - tide_min), 1e-9)
    tide_norm = (tide_values - tide_min) / tide_span

    for i in range(adjusted.shape[0]):
        x = adjusted[i, :, 0]

        if args.left_lift_m:
            w = _window_weight(x, args.left_lift_start_x, args.left_lift_end_x, "left")
            adjusted[i, :, 1] += w * args.left_lift_m

        if args.middle_low_lift_m:
            w = _window_weight(x, args.middle_lift_x_min, args.middle_lift_x_max, "middle")
            low_weight = 1.0 - tide_norm[i]
            adjusted[i, :, 1] += w * low_weight * args.middle_low_lift_m

        if args.right_high_drop_m:
            w = _window_weight(x, args.right_drop_start_x, args.right_drop_end_x, "right")
            high_weight = tide_norm[i]
            adjusted[i, :, 1] -= w * high_weight * args.right_high_drop_m

    return adjusted


def resample_contours(contours: list[ShorelineContour], samples: int,
                      max_cross_shore_span: float | None,
                      trim_longest_valid_run: bool,
                      parameter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if parameter == "baseline-s":
        s_min = max(float(c.s.min()) for c in contours)
        s_max = min(float(c.s.max()) for c in contours)
        if not s_max > s_min:
            raise ValueError("Representative shorelines do not share an overlapping s range.")
        s_grid = np.linspace(s_min, s_max, samples)
    else:
        s_grid = np.linspace(0.0, 1.0, samples)

    vertices = []
    d_grid = []
    for contour in contours:
        if parameter == "baseline-s":
            src_param = contour.s
        else:
            arc = _arc_length_parameter(contour.x, contour.y)
            src_param = arc / arc[-1] if arc[-1] > 0 else np.linspace(0.0, 1.0, len(arc))
        x = np.interp(s_grid, src_param, contour.x)
        y = np.interp(s_grid, src_param, contour.y)
        z = np.full(samples, contour.tide_m, dtype=float)
        vertices.append(np.column_stack([x, y, z]))
        if contour.d is not None:
            d_grid.append(np.interp(s_grid, src_param, contour.d))
    d_arr = np.vstack(d_grid) if len(d_grid) == len(contours) else np.empty((0, samples))
    if max_cross_shore_span is not None:
        if not d_arr.size:
            raise ValueError("--max-cross-shore-span requires rep_shoreline_world.csv files with a d column.")
        span = np.nanmax(d_arr, axis=0) - np.nanmin(d_arr, axis=0)
        keep = np.isfinite(span) & (span <= max_cross_shore_span)
        if trim_longest_valid_run:
            keep = _longest_true_run(keep)
        if keep.sum() < 2:
            raise ValueError(
                f"Cross-shore span filter kept fewer than two samples. "
                f"Relax --max-cross-shore-span; current value: {max_cross_shore_span}"
            )
        s_grid = s_grid[keep]
        vertices = [v[keep] for v in vertices]
        d_arr = d_arr[:, keep]
    return s_grid, np.stack(vertices, axis=0), d_arr, np.array([c.tide_m for c in contours], dtype=float)


def write_plan_preview(path: Path, contours: list[ShorelineContour], verts_grid: np.ndarray) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    tide_values = np.array([c.tide_m for c in contours], dtype=float)
    denom = max(float(tide_values.max() - tide_values.min()), 1e-9)
    for contour, vertices in zip(contours, verts_grid):
        color_value = (contour.tide_m - tide_values.min()) / denom
        ax.plot(vertices[:, 0], vertices[:, 1], color=plt.cm.viridis(color_value), linewidth=1.5)
    outline = np.vstack([verts_grid[0], verts_grid[-1][::-1], verts_grid[0, :1]])
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Surface footprint from representative shoreline contours")
    ax.set_xlabel("X_warped")
    ax.set_ylabel("Y_warped")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_faces(n_contours: int, n_samples: int) -> np.ndarray:
    faces = []
    for i in range(n_contours - 1):
        for j in range(n_samples - 1):
            a = i * n_samples + j
            b = (i + 1) * n_samples + j
            c = (i + 1) * n_samples + (j + 1)
            d = i * n_samples + (j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(faces, dtype=np.int64)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in faces:
            handle.write(f"3 {i} {j} {k}\n")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in faces:
            handle.write(f"f {i + 1} {j + 1} {k + 1}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-root", required=True, help="Root containing <clip>/rep_shoreline_world.csv files.")
    parser.add_argument("--water-csv", required=True, help="CO-OPS water-level CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory for mesh and diagnostics.")
    parser.add_argument("--samples", type=int, default=400, help="Number of alongshore samples per contour.")
    parser.add_argument("--min-contours", type=int, default=3, help="Minimum contours required to build a mesh.")
    parser.add_argument("--sort-by", choices=["tide", "time"], default="tide", help="Contour ordering for mesh connectivity.")
    parser.add_argument("--no-extrapolate", action="store_true", help="Reject clips outside the tide time range.")
    parser.add_argument("--max-cross-shore-span", type=float, default=None,
                        help="Drop samples where max(d)-min(d) across contours exceeds this many meters.")
    parser.add_argument("--keep-disjoint-valid-s", action="store_true",
                        help="Keep all stations passing --max-cross-shore-span instead of only the longest contiguous run.")
    parser.add_argument("--parameter", choices=["baseline-s", "arc"], default="baseline-s",
                        help="How to align points between contours. 'arc' is safer when a straight baseline causes curved-end loops.")
    parser.add_argument("--left-lift-m", type=float, default=0.0, help="Meters added to Y on the left end of the footprint.")
    parser.add_argument("--left-lift-start-x", type=float, default=None, help="Start of left lift blend in X_warped.")
    parser.add_argument("--left-lift-end-x", type=float, default=None, help="End of left lift blend in X_warped.")
    parser.add_argument("--middle-low-lift-m", type=float, default=0.0, help="Meters added to Y in the middle, strongest on low-tide contours.")
    parser.add_argument("--middle-lift-x-min", type=float, default=None, help="Left X_warped edge of middle low-tide lift.")
    parser.add_argument("--middle-lift-x-max", type=float, default=None, help="Right X_warped edge of middle low-tide lift.")
    parser.add_argument("--right-high-drop-m", type=float, default=0.0, help="Meters subtracted from Y on the right, strongest on high-tide contours.")
    parser.add_argument("--right-drop-start-x", type=float, default=None, help="Start of right high-tide drop blend in X_warped.")
    parser.add_argument("--right-drop-end-x", type=float, default=None, help="End of right high-tide drop blend in X_warped.")
    args = parser.parse_args()

    rep_root = Path(args.rep_root)
    out_dir = Path(args.out_dir)
    water_seconds, water_z_m = load_water_levels(Path(args.water_csv))
    contours = load_contours(rep_root, water_seconds, water_z_m, extrapolate=not args.no_extrapolate)
    if len(contours) < args.min_contours:
        raise SystemExit(f"Need at least {args.min_contours} representative shorelines; found {len(contours)}.")

    if args.sort_by == "tide":
        contours = sorted(contours, key=lambda c: (c.tide_m, c.timestamp))
    else:
        contours = sorted(contours, key=lambda c: c.timestamp)

    s_grid, verts_grid, d_grid, tide_values = resample_contours(
        contours,
        int(args.samples),
        max_cross_shore_span=args.max_cross_shore_span,
        trim_longest_valid_run=not args.keep_disjoint_valid_s,
        parameter=args.parameter,
    )
    verts_grid = apply_footprint_adjustments(verts_grid, tide_values, args)
    vertices = verts_grid.reshape(-1, 3)
    faces = build_faces(len(contours), len(s_grid))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "beach_surface.ply", vertices, faces)
    write_obj(out_dir / "beach_surface.obj", vertices, faces)
    write_plan_preview(out_dir / "surface_plan_preview.png", contours, verts_grid)

    combined_rows = []
    for contour, contour_vertices in zip(contours, verts_grid):
        frame = pd.DataFrame(contour_vertices, columns=["X_warped", "Y_warped", "Z_m"])
        frame.insert(0, "s", s_grid)
        frame["clip_name"] = contour.clip
        frame["timestamp_utc"] = contour.timestamp.isoformat()
        combined_rows.append(frame)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(out_dir / "all_representative_shorelines_3d.csv", index=False)

    xy_span = math.hypot(float(vertices[:, 0].max() - vertices[:, 0].min()), float(vertices[:, 1].max() - vertices[:, 1].min()))
    d_span = float(np.nanmax(d_grid) - np.nanmin(d_grid)) if d_grid.size else float("nan")
    diagnostics = pd.DataFrame({
        "metric": [
            "contours",
            "samples_per_contour",
            "tide_min_m",
            "tide_max_m",
            "tide_span_m",
            "s_min_m",
            "s_max_m",
            "s_span_m",
            "parameter",
            "plan_bbox_diagonal_m",
            "cross_shore_d_span_m",
            "cross_shore_span_filter_m",
            "vertices",
            "faces",
        ],
        "value": [
            len(contours),
            len(s_grid),
            float(tide_values.min()),
            float(tide_values.max()),
            float(tide_values.max() - tide_values.min()),
            float(s_grid.min()),
            float(s_grid.max()),
            float(s_grid.max() - s_grid.min()),
            args.parameter,
            xy_span,
            d_span,
            float(args.max_cross_shore_span) if args.max_cross_shore_span is not None else np.nan,
            len(vertices),
            len(faces),
        ],
    })
    diagnostics.to_csv(out_dir / "surface_diagnostics.csv", index=False)

    contour_summary = pd.DataFrame({
        "clip_name": [c.clip for c in contours],
        "timestamp_utc": [c.timestamp.isoformat() for c in contours],
        "tide_m": [c.tide_m for c in contours],
        "rep_csv": [str(c.path) for c in contours],
    })
    contour_summary.to_csv(out_dir / "surface_contours.csv", index=False)

    print(f"Wrote {out_dir / 'beach_surface.ply'}")
    print(f"Wrote {out_dir / 'beach_surface.obj'}")
    print(f"Tide span: {tide_values.max() - tide_values.min():.3f} m across {len(contours)} contours")
    print(f"Cross-shore d span: {d_span:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
