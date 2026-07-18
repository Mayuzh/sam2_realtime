"""Generate curve boxplots from shoreline LabelMe JSONs, grouped by sea-state bin or clip.

This script treats each shoreline JSON as an open shoreline polyline, not as a
closed polygon. It converts each shoreline into a 1D function by casting
transects from either a fixed left vertical baseline or an annotated polyline
baseline. Each transect uses the first shoreline intersection it sees from the
baseline, then removes sharp local spikes before building the boxplot.

Default input layout:
    shoreline_outputs/jennettes_pier/<direction_group>/bin_<n>/<video_stem>/*.json

Default output layout:
    curve_boxplot_outputs/jennettes_pier/<direction_group>/bin_<n>/
        curve_boxplot.png
        curves.csv
        boxplot_stats.csv

Per-clip output layout with --group-by clip:
    curve_boxplot_outputs_per_clip/jennettes_pier/<direction_group>/bin_<n>/<video_stem>/
        curve_boxplot.png
        curves.csv
        boxplot_stats.csv

The final summary CSV contains one row per generated curve boxplot, including
a scalar shoreline variance value for later sea_state correlation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SHORELINE_ROOT = SCRIPT_DIR / "shoreline_outputs" / "jennettes_pier"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "curve_boxplot_outputs" / "jennettes_pier"
DEFAULT_PER_CLIP_OUTPUT_ROOT = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "jennettes_pier"
DEFAULT_MANIFEST = SCRIPT_DIR / "downloaded_webcoos_clips" / "download_manifest.csv"
DEFAULT_CANDIDATES = SCRIPT_DIR / "candidate_clip_outputs" / "candidate_clip_table.csv"


@dataclass
class CurveRecord:
    json_path: Path
    video_stem: str
    direction_group: str
    bin_name: str
    curve: np.ndarray
    valid_fraction: float


@dataclass
class BaselineGeometry:
    coordinate: np.ndarray
    origins: np.ndarray
    normals: np.ndarray
    points: np.ndarray
    mode: str
    source_path: Path | None = None
    image_path: Path | None = None
    image_width: int = 1280
    image_height: int = 960


def trim_baseline_geometry(
    baseline: BaselineGeometry,
    coordinate_min: float | None,
    coordinate_max: float | None,
) -> BaselineGeometry:
    keep = np.ones(len(baseline.coordinate), dtype=bool)
    if coordinate_min is not None:
        keep &= baseline.coordinate >= coordinate_min
    if coordinate_max is not None:
        keep &= baseline.coordinate <= coordinate_max
    if keep.sum() < 2:
        raise ValueError(
            "Baseline coordinate crop leaves fewer than two transects. "
            "Relax --baseline-coordinate-min/--baseline-coordinate-max."
        )
    return BaselineGeometry(
        coordinate=baseline.coordinate[keep],
        origins=baseline.origins[keep],
        normals=baseline.normals[keep],
        points=baseline.points,
        mode=baseline.mode,
        source_path=baseline.source_path,
        image_path=baseline.image_path,
        image_width=baseline.image_width,
        image_height=baseline.image_height,
    )


def resolve_path(path_value: str | None, default: Path) -> Path:
    if path_value is None:
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def natural_key(path: Path) -> list[Any]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(path))]


def load_labelme_shorelines(json_path: Path) -> tuple[list[np.ndarray], int, int]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    polylines = []

    for shape in data.get("shapes", []):
        if shape.get("label") != "shoreline":
            continue
        pts = np.asarray(shape.get("points", []), dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
            polylines.append(pts[:, :2])

    return polylines, width, height


def load_baseline_points(
    json_path: Path,
    label: str,
    order: str,
    min_vertex_spacing: float,
) -> tuple[np.ndarray, int, int, Path | None]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    points = None
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        candidate = np.asarray(shape.get("points", []), dtype=float)
        if candidate.ndim == 2 and candidate.shape[0] >= 2 and candidate.shape[1] >= 2:
            points = candidate[:, :2]
            break
    if points is None:
        raise ValueError(f"No {label!r} polyline with at least two points in {json_path}")

    if order == "x":
        points = points[np.argsort(points[:, 0])]
    elif order == "y":
        points = points[np.argsort(points[:, 1])]

    if min_vertex_spacing > 0 and len(points) > 2:
        kept = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - kept[-1]) >= min_vertex_spacing:
                kept.append(point)
        if len(kept) == 1 or not np.array_equal(kept[-1], points[-1]):
            kept.append(points[-1])
        points = np.asarray(kept, dtype=float)

    width = int(data.get("imageWidth") or 1280)
    height = int(data.get("imageHeight") or 960)
    image_name = str(data.get("imagePath") or "").strip()
    image_path = json_path.parent / image_name if image_name else None
    if image_path is not None and not image_path.exists():
        image_path = None
    return points, width, height, image_path


def sample_polyline_baseline(
    points: np.ndarray,
    samples: int,
    normal_direction: str,
    normal_mode: str = "perpendicular",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = np.diff(points, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    keep = lengths > 1e-9
    segments = segments[keep]
    lengths = lengths[keep]
    starts = points[:-1][keep]
    if not len(segments):
        raise ValueError("Baseline points do not define a nonzero segment")

    cumulative = np.r_[0.0, np.cumsum(lengths)]
    coordinate = np.linspace(0.0, cumulative[-1], max(2, samples))
    origins = np.empty((len(coordinate), 2), dtype=float)
    normals = np.empty_like(origins)
    desired = {
        "down": np.array([0.0, 1.0]),
        "up": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }[normal_direction]

    fixed_normal = desired / np.linalg.norm(desired)

    for index, distance in enumerate(coordinate):
        segment_index = min(
            int(np.searchsorted(cumulative[1:], distance, side="right")),
            len(segments) - 1,
        )
        fraction = (distance - cumulative[segment_index]) / lengths[segment_index]
        fraction = float(np.clip(fraction, 0.0, 1.0))
        origins[index] = starts[segment_index] + fraction * segments[segment_index]
        if normal_mode == "fixed":
            normal = fixed_normal
        else:
            tangent = segments[segment_index] / lengths[segment_index]
            normal = np.array([-tangent[1], tangent[0]])
            if float(np.dot(normal, desired)) < 0:
                normal *= -1.0
        normals[index] = normal
    return coordinate, origins, normals


def build_baseline_geometry(args: argparse.Namespace) -> BaselineGeometry:
    if args.baseline_json:
        baseline_path = resolve_path(args.baseline_json, Path(args.baseline_json))
        points, width, height, image_path = load_baseline_points(
            baseline_path,
            label=args.baseline_label,
            order=args.baseline_order,
            min_vertex_spacing=args.baseline_min_vertex_spacing,
        )
        coordinate, origins, normals = sample_polyline_baseline(
            points,
            samples=args.baseline_samples,
            normal_direction=args.normal_direction,
            normal_mode=args.baseline_normal_mode,
        )
        return BaselineGeometry(
            coordinate=coordinate,
            origins=origins,
            normals=normals,
            points=points,
            mode="polyline",
            source_path=baseline_path,
            image_path=image_path,
            image_width=width,
            image_height=height,
        )

    coordinate = np.linspace(args.baseline_y_min, args.baseline_y_max, args.baseline_samples)
    origins = np.column_stack([np.full_like(coordinate, args.baseline_x), coordinate])
    normals = np.tile(np.array([[1.0, 0.0]]), (len(coordinate), 1))
    points = np.array(
        [[args.baseline_x, args.baseline_y_min], [args.baseline_x, args.baseline_y_max]],
        dtype=float,
    )
    return BaselineGeometry(
        coordinate=coordinate,
        origins=origins,
        normals=normals,
        points=points,
        mode="vertical",
    )


def scale_baseline_to_frame(
    baseline: BaselineGeometry,
    width: int,
    height: int,
) -> BaselineGeometry:
    if width == baseline.image_width and height == baseline.image_height:
        return baseline
    scale = np.array(
        [width / float(baseline.image_width), height / float(baseline.image_height)],
        dtype=float,
    )
    normals = baseline.normals * scale
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return BaselineGeometry(
        coordinate=baseline.coordinate,
        origins=baseline.origins * scale,
        normals=normals,
        points=baseline.points * scale,
        mode=baseline.mode,
        source_path=baseline.source_path,
        image_path=baseline.image_path,
        image_width=width,
        image_height=height,
    )


def split_polyline_on_gaps(points: np.ndarray, max_gap: float) -> list[np.ndarray]:
    if len(points) < 2:
        return []

    gaps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    split_after = np.where(gaps > max_gap)[0] + 1
    chunks = np.split(points, split_after)
    return [chunk for chunk in chunks if len(chunk) >= 2]


def smooth_polyline_points(points: np.ndarray, window: int) -> np.ndarray:
    """Smooth small local wiggles while preserving the open shoreline ends."""
    if window <= 2 or len(points) < 3:
        return points
    if window % 2 == 0:
        window += 1
    if len(points) < window:
        return points

    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def clean_polylines(
    polylines: list[np.ndarray],
    width: int,
    height: int,
    top_margin: float,
    bottom_margin: float,
    left_margin: float,
    right_margin: float,
    max_gap: float,
    min_points: int,
    simplify_tolerance: float,
    polyline_smooth_window: int,
) -> list[np.ndarray]:
    cleaned: list[np.ndarray] = []

    for pts in polylines:
        x = pts[:, 0]
        y = pts[:, 1]

        # Drop likely image-border artifacts. This also removes the visual
        # polygon-closing edge when LabelMe displays an open shoreline as a polygon.
        keep = (
            (x >= left_margin)
            & (x <= max(left_margin, width - right_margin))
            & (y >= top_margin)
            & (y <= max(top_margin, height - bottom_margin))
        )
        pts = pts[keep]
        if len(pts) < min_points:
            continue

        for chunk in split_polyline_on_gaps(pts, max_gap=max_gap):
            if len(chunk) >= min_points:
                chunk = smooth_polyline_points(chunk, polyline_smooth_window)
                if simplify_tolerance > 0:
                    chunk = cv2.approxPolyDP(
                        chunk.astype(np.float32).reshape(-1, 1, 2),
                        epsilon=simplify_tolerance,
                        closed=False,
                    ).reshape(-1, 2).astype(float)
                if len(chunk) < 2:
                    continue
                cleaned.append(chunk)

    return cleaned


def horizontal_intersections_for_y(polyline: np.ndarray, y0: float, baseline_x: float) -> list[float]:
    pts = polyline
    intersections: list[float] = []

    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y_a = p0
        x1, y_b = p1

        if y_a == y_b:
            if abs(y0 - y_a) < 1e-9:
                # Horizontal segment on the transect: use both endpoints.
                for x in (x0, x1):
                    if x >= baseline_x:
                        intersections.append(float(x))
            continue

        ymin = min(y_a, y_b)
        ymax = max(y_a, y_b)
        if y0 < ymin or y0 > ymax:
            continue

        u = (y0 - y_a) / (y_b - y_a)
        if 0.0 <= u <= 1.0:
            x = x0 + u * (x1 - x0)
            if x >= baseline_x:
                intersections.append(float(x))

    return intersections


def shoreline_to_curve(
    polylines: list[np.ndarray],
    y_grid: np.ndarray,
    baseline_x: float,
    prefer: str,
) -> np.ndarray:
    curve = np.full(len(y_grid), np.nan, dtype=float)

    for i, y0 in enumerate(y_grid):
        xs: list[float] = []
        for line in polylines:
            xs.extend(horizontal_intersections_for_y(line, y0, baseline_x))

        if not xs:
            continue

        xs_arr = np.asarray(xs, dtype=float)
        if prefer in {"first", "nearest"}:
            x_pick = np.nanmin(xs_arr)
        elif prefer == "farthest":
            x_pick = np.nanmax(xs_arr)
        else:
            x_pick = np.nanmedian(xs_arr)
        curve[i] = x_pick - baseline_x

    return curve


def ray_intersections(
    polyline: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    segment_starts = polyline[:-1]
    segment_vectors = polyline[1:] - polyline[:-1]
    relative = segment_starts - origin
    denominator = (
        direction[0] * segment_vectors[:, 1]
        - direction[1] * segment_vectors[:, 0]
    )
    nonparallel = np.abs(denominator) > 1e-9
    if not nonparallel.any():
        return np.empty(0, dtype=float)

    distance = np.full(len(segment_vectors), np.nan, dtype=float)
    fraction = np.full(len(segment_vectors), np.nan, dtype=float)
    distance[nonparallel] = (
        relative[nonparallel, 0] * segment_vectors[nonparallel, 1]
        - relative[nonparallel, 1] * segment_vectors[nonparallel, 0]
    ) / denominator[nonparallel]
    fraction[nonparallel] = (
        relative[nonparallel, 0] * direction[1]
        - relative[nonparallel, 1] * direction[0]
    ) / denominator[nonparallel]
    valid = (
        nonparallel
        & (distance >= 0.0)
        & (fraction >= 0.0)
        & (fraction <= 1.0)
    )
    return distance[valid]


def shoreline_to_baseline_curve(
    polylines: list[np.ndarray],
    baseline: BaselineGeometry,
    prefer: str,
    max_distance: float | None,
    reference_curve: np.ndarray | None = None,
    max_reference_deviation: float | None = None,
) -> np.ndarray:
    curve = np.full(len(baseline.coordinate), np.nan, dtype=float)
    for index, (origin, normal) in enumerate(zip(baseline.origins, baseline.normals)):
        hits = [ray_intersections(line, origin, normal) for line in polylines]
        hits = [values for values in hits if values.size]
        if not hits:
            continue
        distances = np.concatenate(hits)
        if max_distance is not None:
            distances = distances[distances <= max_distance]
        if (
            reference_curve is not None
            and max_reference_deviation is not None
            and np.isfinite(reference_curve[index])
        ):
            distances = distances[
                np.abs(distances - reference_curve[index]) <= max_reference_deviation
            ]
        if not distances.size:
            continue
        if prefer in {"first", "nearest"}:
            curve[index] = float(np.min(distances))
        elif prefer == "farthest":
            curve[index] = float(np.max(distances))
        else:
            curve[index] = float(np.median(distances))
    return curve


def remove_local_spikes(
    curve: np.ndarray,
    window: int,
    threshold_px: float,
    mad_scale: float,
) -> np.ndarray:
    """Drop isolated x-distance jumps caused by knots, people, or stray lines."""
    if window <= 2 or threshold_px <= 0:
        return curve
    if window % 2 == 0:
        window += 1

    out = curve.copy()
    half = window // 2
    for i, value in enumerate(curve):
        if not np.isfinite(value):
            continue

        start = max(0, i - half)
        end = min(len(curve), i + half + 1)
        local = np.concatenate([curve[start:i], curve[i + 1 : end]])
        local = local[np.isfinite(local)]
        if local.size < max(3, half):
            continue

        local_median = float(np.median(local))
        mad = float(np.median(np.abs(local - local_median)))
        robust_limit = threshold_px
        if mad > 0:
            robust_limit = max(threshold_px, mad_scale * 1.4826 * mad)

        if abs(value - local_median) > robust_limit:
            out[i] = np.nan

    return out


def interpolate_short_gaps(curve: np.ndarray, max_gap_samples: int) -> np.ndarray:
    if max_gap_samples <= 0:
        return curve

    out = curve.copy()
    valid = np.isfinite(out)
    if valid.sum() < 2:
        return out

    idx = np.arange(len(out))
    missing = ~valid
    starts = np.where(np.diff(np.r_[False, missing, False]) == 1)[0]
    ends = np.where(np.diff(np.r_[False, missing, False]) == -1)[0]

    for start, end in zip(starts, ends):
        if end - start > max_gap_samples:
            continue
        if start == 0 or end >= len(out):
            continue
        out[start:end] = np.interp(idx[start:end], idx[valid], out[valid])

    return out


def moving_median(curve: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return curve
    if window % 2 == 0:
        window += 1

    half = window // 2
    padded = np.pad(curve, (half, half), mode="edge")
    smoothed = np.full_like(curve, np.nan, dtype=float)
    for i in range(len(curve)):
        vals = padded[i : i + window]
        if np.isfinite(vals).any():
            smoothed[i] = np.nanmedian(vals)
    return smoothed


def weighted_moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    """Smooth finite curve sections without filling long missing regions."""
    if window <= 1:
        return curve
    if window % 2 == 0:
        window += 1

    weights = np.hanning(window)
    if not np.any(weights > 0):
        weights = np.ones(window, dtype=float)
    finite = np.isfinite(curve)
    values = np.where(finite, curve, 0.0)
    numerator = np.convolve(values, weights, mode="same")
    denominator = np.convolve(finite.astype(float), weights, mode="same")
    result = np.full_like(curve, np.nan, dtype=float)
    usable = finite & (denominator > 0)
    result[usable] = numerator[usable] / denominator[usable]
    return result


def save_baseline_preview(
    baseline: BaselineGeometry,
    output_path: Path,
    transect_length: float,
) -> None:
    if baseline.image_path is None:
        return
    image = plt.imread(baseline.image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.plot(baseline.points[:, 0], baseline.points[:, 1], color="yellow", linewidth=2)
    step = max(1, len(baseline.coordinate) // 24)
    for origin, normal in zip(baseline.origins[::step], baseline.normals[::step]):
        end = origin + normal * transect_length
        plt.plot([origin[0], end[0]], [origin[1], end[1]], color="cyan", linewidth=0.8, alpha=0.8)
    plt.xlim(0, baseline.image_width - 1)
    plt.ylim(baseline.image_height - 1, 0)
    plt.title("Baseline and transect directions")
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def functional_depth_rank(curves: np.ndarray) -> np.ndarray:
    """Fraiman-Muniz style depth; higher means more central."""
    n_curves, n_samples = curves.shape
    filled = curves.copy()

    med = np.nanmedian(filled, axis=0)
    inds = np.where(~np.isfinite(filled))
    if inds[0].size:
        filled[inds] = np.take(med, inds[1])

    depths = np.zeros(n_curves, dtype=float)
    for j in range(n_samples):
        col = filled[:, j]
        order = np.argsort(col)
        ranks = np.empty(n_curves, dtype=int)
        ranks[order] = np.arange(n_curves) + 1

        center = (n_curves + 1) / 2.0
        denom = (n_curves - 1) / 2.0 if n_curves > 1 else 1.0
        depths += 1.0 - (np.abs(ranks - center) / denom)

    return depths / max(1, n_samples)


def column_nan_extreme(values: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(values)
    has_value = finite.any(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=float)
    if not has_value.any():
        return result

    if mode == "min":
        masked = np.where(finite, values, np.inf)
        result[has_value] = masked[:, has_value].min(axis=0)
    elif mode == "max":
        masked = np.where(finite, values, -np.inf)
        result[has_value] = masked[:, has_value].max(axis=0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return result


def build_curve_boxplot(curves: np.ndarray, central_band_percent: float = 50.0) -> dict[str, np.ndarray]:
    depths = functional_depth_rank(curves)
    order = np.argsort(-depths)
    sorted_curves = curves[order]
    pointwise_median = np.nanmedian(curves, axis=0)
    median_curve = np.where(np.isfinite(sorted_curves[0]), sorted_curves[0], pointwise_median)

    central_fraction = float(np.clip(central_band_percent, 1.0, 100.0)) / 100.0
    n_central = max(1, int(math.ceil(central_fraction * curves.shape[0])))
    central = sorted_curves[:n_central]
    lower = column_nan_extreme(central, "min")
    upper = column_nan_extreme(central, "max")
    fallback_lower = column_nan_extreme(curves, "min")
    fallback_upper = column_nan_extreme(curves, "max")
    lower = np.where(np.isfinite(lower), lower, fallback_lower)
    upper = np.where(np.isfinite(upper), upper, fallback_upper)

    band_width = upper - lower
    positive = band_width[np.isfinite(band_width) & (band_width > 0)]
    fallback_width = float(np.nanmedian(positive)) if positive.size else 1.0
    band_width = np.where((~np.isfinite(band_width)) | (band_width == 0), fallback_width, band_width)

    outliers = []
    for curve in curves:
        too_low = curve < (median_curve - 3.0 * band_width)
        too_high = curve > (median_curve + 3.0 * band_width)
        outliers.append(bool(np.any(too_low | too_high)))

    return {
        "median": median_curve,
        "lower": lower,
        "upper": upper,
        "depths": depths,
        "order": order,
        "outliers": np.asarray(outliers, dtype=bool),
    }


def smooth_boxplot_statistics(
    box: dict[str, np.ndarray],
    window: int,
) -> dict[str, np.ndarray]:
    if window <= 1:
        return box
    result = dict(box)
    median = weighted_moving_average(box["median"], window)
    lower = weighted_moving_average(box["lower"], window)
    upper = weighted_moving_average(box["upper"], window)
    result["median"] = median
    result["lower"] = np.minimum(lower, median)
    result["upper"] = np.maximum(upper, median)
    return result


def curve_variance_scalar(curves: np.ndarray, outliers: np.ndarray | None = None) -> float:
    if outliers is not None and (~outliers).sum() >= 2:
        curves = curves[~outliers]
    usable_columns = np.isfinite(curves).any(axis=0)
    if not usable_columns.any():
        return float("nan")
    pointwise_var = np.nanvar(curves[:, usable_columns], axis=0, ddof=0)
    return float(np.nanmean(pointwise_var))


def central_band_area(y_grid: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    width = upper - lower
    valid = np.isfinite(width)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapezoid(width[valid], y_grid[valid]))


def discover_groups(root: Path) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for json_path in sorted(root.rglob("*.json"), key=natural_key):
        parts = json_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        direction_group = parts[0]
        bin_name = parts[1]
        if not bin_name.startswith("bin_"):
            continue
        groups.setdefault((direction_group, bin_name), []).append(json_path)
    return groups


def discover_clip_groups(root: Path) -> dict[tuple[str, str, str], list[Path]]:
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for json_path in sorted(root.rglob("*.json"), key=natural_key):
        parts = json_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        direction_group = parts[0]
        bin_name = parts[1]
        video_stem = parts[2]
        if not bin_name.startswith("bin_"):
            continue
        groups.setdefault((direction_group, bin_name, video_stem), []).append(json_path)
    return groups


def build_manifest_lookup(manifest_path: Path, candidate_path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "clip_path" in manifest.columns:
            for _, row in manifest.iterrows():
                clip_path = str(row.get("clip_path", ""))
                if not clip_path or clip_path == "nan":
                    continue
                stem = Path(clip_path).stem
                lookup[stem] = row.to_dict()

    # Fallback for cases where only candidate rows exist and output folder stems
    # match downloaded clip stems poorly. The group/bin aggregation still works.
    if not lookup and candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
        for _, row in candidates.iterrows():
            key = f"{safe_name(row.get('direction_group', ''))}/bin_{int(row.get('sea_state_bin', 0))}"
            lookup[key] = row.to_dict()

    return lookup


def metadata_for_records(records: list[CurveRecord], metadata_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    matched_stems: set[str] = set()
    for record in records:
        if record.video_stem in metadata_lookup and record.video_stem not in matched_stems:
            rows.append(metadata_lookup[record.video_stem])
            matched_stems.add(record.video_stem)

    if not rows:
        key = f"{records[0].direction_group}/{records[0].bin_name}"
        if key in metadata_lookup:
            rows.append(metadata_lookup[key])

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for col in ["location", "direction_group", "final_plot_group"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique()
            out[col] = vals[0] if len(vals) else ""

    for col in ["H", "T", "sea_state", "direction_degrees", "sea_state_bin"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(numeric.mean()) if numeric.notna().any() else np.nan
            out[f"{col}_min"] = float(numeric.min()) if numeric.notna().any() else np.nan
            out[f"{col}_max"] = float(numeric.max()) if numeric.notna().any() else np.nan

    out["metadata_rows_matched"] = len(rows)
    return out


def build_reference_curve(
    json_path: Path,
    baseline: BaselineGeometry,
    args: argparse.Namespace,
) -> np.ndarray:
    polylines, width, height = load_labelme_shorelines(json_path)
    cleaned = clean_polylines(
        polylines,
        width=width,
        height=height,
        top_margin=args.top_margin,
        bottom_margin=args.bottom_margin,
        left_margin=args.left_margin,
        right_margin=args.right_margin,
        max_gap=args.max_gap,
        min_points=args.min_points,
        simplify_tolerance=args.simplify_tolerance,
        polyline_smooth_window=args.polyline_smooth_window,
    )
    if not cleaned:
        raise ValueError(f"No usable shoreline in reference JSON: {json_path}")
    frame_baseline = scale_baseline_to_frame(baseline, width, height)
    curve = shoreline_to_baseline_curve(
        cleaned,
        baseline=frame_baseline,
        prefer="first",
        max_distance=args.max_transect_distance,
    )
    curve = remove_local_spikes(
        curve,
        window=args.despike_window,
        threshold_px=args.despike_threshold_px,
        mad_scale=args.despike_mad_scale,
    )
    curve = interpolate_short_gaps(curve, args.interpolate_gap)
    curve = moving_median(curve, args.smooth_window)
    return weighted_moving_average(curve, args.smooth_mean_window)


def curves_from_jsons(
    json_paths: list[Path],
    shoreline_root: Path,
    baseline: BaselineGeometry,
    reference_curve: np.ndarray | None,
    args: argparse.Namespace,
) -> list[CurveRecord]:
    records: list[CurveRecord] = []

    for json_path in json_paths:
        polylines, width, height = load_labelme_shorelines(json_path)
        if width <= 0 or height <= 0:
            continue

        cleaned = clean_polylines(
            polylines,
            width=width,
            height=height,
            top_margin=args.top_margin,
            bottom_margin=args.bottom_margin,
            left_margin=args.left_margin,
            right_margin=args.right_margin,
            max_gap=args.max_gap,
            min_points=args.min_points,
        simplify_tolerance=args.simplify_tolerance,
        polyline_smooth_window=args.polyline_smooth_window,
        )
        if not cleaned:
            continue

        frame_baseline = scale_baseline_to_frame(baseline, width, height)
        curve = shoreline_to_baseline_curve(
            cleaned,
            baseline=frame_baseline,
            prefer=args.intersection_choice,
            max_distance=args.max_transect_distance,
            reference_curve=reference_curve,
            max_reference_deviation=args.max_reference_deviation,
        )
        curve = remove_local_spikes(
            curve,
            window=args.despike_window,
            threshold_px=args.despike_threshold_px,
            mad_scale=args.despike_mad_scale,
        )
        curve = interpolate_short_gaps(curve, args.interpolate_gap)
        curve = moving_median(curve, args.smooth_window)
        curve = weighted_moving_average(curve, args.smooth_mean_window)

        valid_fraction = float(np.isfinite(curve).mean())
        if valid_fraction < args.min_curve_valid_fraction:
            continue

        parts = json_path.relative_to(shoreline_root).parts
        direction_group = parts[0]
        bin_name = parts[1]
        video_stem = parts[2]

        records.append(
            CurveRecord(
                json_path=json_path,
                video_stem=video_stem,
                direction_group=direction_group,
                bin_name=bin_name,
                curve=curve,
                valid_fraction=valid_fraction,
            )
        )

    return records


def plot_boxplot(
    baseline_coordinate: np.ndarray,
    curves: np.ndarray,
    box: dict[str, np.ndarray],
    title: str,
    output_path: Path,
    x_min: float,
    x_max: float,
    y_max: float | None,
    baseline_mode: str,
    central_band_percent: float,
    show_outlier_lines: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.fill_between(
        baseline_coordinate,
        box["lower"],
        box["upper"],
        alpha=0.3,
        label=f"Central {central_band_percent:.0f}% band",
    )
    plt.plot(baseline_coordinate, box["median"], linewidth=2, label="Median")

    if show_outlier_lines:
        outlier_indices = np.where(box["outliers"])[0]
        for idx in outlier_indices[:20]:
            plt.plot(baseline_coordinate, curves[idx], linewidth=0.7, alpha=0.35)

    if baseline_mode == "polyline":
        plt.xlabel("Alongshore distance on baseline (pixels)")
        plt.ylabel("First shoreline intersection distance (pixels)")
    else:
        plt.xlabel("Alongshore baseline coordinate (pixel y)")
        plt.ylabel("Distance offshore from left baseline (pixels)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xlim(x_min, x_max)
    if y_max is not None:
        plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_group_outputs(
    output_dir: Path,
    baseline_coordinate: np.ndarray,
    records: list[CurveRecord],
    curves: np.ndarray,
    box: dict[str, np.ndarray],
    baseline_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "s" if baseline_mode == "polyline" else "y"
    curve_df = pd.DataFrame(
        curves,
        columns=[f"{prefix}_{int(round(value))}" for value in baseline_coordinate],
    )
    curve_df.insert(0, "json_path", [str(r.json_path) for r in records])
    curve_df.insert(1, "video_stem", [r.video_stem for r in records])
    curve_df.insert(2, "valid_fraction", [r.valid_fraction for r in records])
    curve_df.insert(3, "is_outlier", box["outliers"])
    curve_df.to_csv(output_dir / "curves.csv", index=False)

    stats_df = pd.DataFrame(
        {
            "baseline_coordinate": baseline_coordinate,
            "median": box["median"],
            "lower": box["lower"],
            "upper": box["upper"],
        }
    )
    stats_df.to_csv(output_dir / "boxplot_stats.csv", index=False)


def run(args: argparse.Namespace) -> int:
    shoreline_root = resolve_path(args.shoreline_root, DEFAULT_SHORELINE_ROOT)
    default_output = DEFAULT_PER_CLIP_OUTPUT_ROOT if args.group_by == "clip" else DEFAULT_OUTPUT_ROOT
    output_root = resolve_path(args.output_root, default_output)
    manifest_path = resolve_path(args.manifest, DEFAULT_MANIFEST)
    candidate_path = resolve_path(args.candidate_csv, DEFAULT_CANDIDATES)

    baseline = build_baseline_geometry(args)
    baseline = trim_baseline_geometry(
        baseline,
        args.baseline_coordinate_min,
        args.baseline_coordinate_max,
    )
    baseline_coordinate = baseline.coordinate
    reference_curve = None
    if args.reference_shoreline_json:
        reference_path = resolve_path(
            args.reference_shoreline_json,
            Path(args.reference_shoreline_json),
        )
        reference_curve = build_reference_curve(reference_path, baseline, args)
        print(f"Using shoreline reference envelope: {reference_path}")
    if args.group_by == "clip":
        groups = discover_clip_groups(shoreline_root)
    else:
        groups = discover_groups(shoreline_root)
    if args.direction_group:
        groups = {
            key: paths for key, paths in groups.items() if key[0] == args.direction_group
        }
    if args.sea_state_bin is not None:
        wanted_bin = f"bin_{args.sea_state_bin}"
        groups = {key: paths for key, paths in groups.items() if key[1] == wanted_bin}
    metadata_lookup = build_manifest_lookup(manifest_path, candidate_path)

    if not groups:
        print(f"No shoreline JSON groups found under {shoreline_root}")
        return 1

    save_baseline_preview(
        baseline,
        output_root / "baseline_transects.png",
        transect_length=args.transect_preview_length,
    )
    if baseline.source_path:
        print(f"Using polyline baseline: {baseline.source_path}")

    summary_rows: list[dict[str, Any]] = []

    for group_key, json_paths in sorted(groups.items()):
        direction_group = group_key[0]
        bin_name = group_key[1]
        video_stem = group_key[2] if args.group_by == "clip" else ""
        if args.max_json_per_bin is not None:
            json_paths = json_paths[: args.max_json_per_bin]

        group_label = (
            f"{direction_group}/{bin_name}/{video_stem}"
            if args.group_by == "clip"
            else f"{direction_group}/{bin_name}"
        )
        print(f"[start] {group_label}: {len(json_paths)} JSON files")
        records = curves_from_jsons(
            json_paths,
            shoreline_root,
            baseline,
            reference_curve,
            args,
        )
        if len(records) < args.min_curves:
            print(f"[skip] {group_label}: {len(records)} usable curves")
            continue

        curves = np.vstack([r.curve for r in records])
        sample_valid_fraction = np.isfinite(curves).mean(axis=0)
        valid_samples = sample_valid_fraction >= args.min_sample_valid_fraction
        if valid_samples.sum() < args.min_samples:
            print(f"[skip] {group_label}: {valid_samples.sum()} usable baseline samples")
            continue

        curves_valid = curves[:, valid_samples]
        coordinate_valid = baseline_coordinate[valid_samples]
        keep_frames = np.isfinite(curves_valid).mean(axis=1) >= args.min_curve_valid_fraction
        curves_valid = curves_valid[keep_frames]
        kept_records = [record for record, keep in zip(records, keep_frames) if keep]

        if curves_valid.shape[0] < args.min_curves:
            print(f"[skip] {group_label}: {curves_valid.shape[0]} curves after sample mask")
            continue

        box = build_curve_boxplot(curves_valid, args.central_band_percent)
        box = smooth_boxplot_statistics(box, args.box_smooth_window)
        variance_value = curve_variance_scalar(curves_valid, box["outliers"])
        band_area = central_band_area(coordinate_valid, box["lower"], box["upper"])

        group_output = output_root / direction_group / bin_name
        if args.group_by == "clip":
            group_output = group_output / video_stem
        title = f"Curve Boxplot - {group_label.replace('/', ' ')}"
        plot_boxplot(
            coordinate_valid,
            curves_valid,
            box,
            title,
            group_output / "curve_boxplot.png",
            float(baseline_coordinate.min()),
            float(baseline_coordinate.max()),
            args.y_max,
            baseline.mode,
            args.central_band_percent,
            args.show_outlier_lines,
        )
        save_group_outputs(
            group_output,
            coordinate_valid,
            kept_records,
            curves_valid,
            box,
            baseline.mode,
        )

        meta = metadata_for_records(kept_records, metadata_lookup)
        summary = {
            "location": meta.get("location", shoreline_root.name),
            "direction_group": direction_group,
            "final_plot_group": meta.get("final_plot_group", ""),
            "sea_state_bin": int(bin_name.replace("bin_", "")) if bin_name.replace("bin_", "").isdigit() else bin_name,
            "video_stem": video_stem,
            "group_by": args.group_by,
            "n_json_files": len(json_paths),
            "n_usable_curves": int(curves_valid.shape[0]),
            "n_baseline_samples": int(curves_valid.shape[1]),
            "baseline_mode": baseline.mode,
            "baseline_source": str(baseline.source_path or ""),
            "baseline_x": args.baseline_x if baseline.mode == "vertical" else np.nan,
            "baseline_coordinate_min": float(coordinate_valid.min()),
            "baseline_coordinate_max": float(coordinate_valid.max()),
            "curve_boxplot_variance": variance_value,
            "central_band_area": band_area,
            "central_band_percent": args.central_band_percent,
            "outlier_count": int(box["outliers"].sum()),
            "plot_path": str(group_output / "curve_boxplot.png"),
        }
        summary.update(meta)
        summary_rows.append(summary)

        print(
            f"[ok] {group_label}: "
            f"{curves_valid.shape[0]} curves, variance={variance_value:.3f}"
        )

    if not summary_rows:
        print("No curve boxplots were generated.")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "curve_boxplot_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    for row in summary_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoreline-root", default=None, help="Root containing shoreline JSON outputs.")
    parser.add_argument("--output-root", default=None, help="Output root for plots, curve CSVs, and summary.")
    parser.add_argument(
        "--group-by",
        choices=["bin", "clip"],
        default="bin",
        help="Generate one curve boxplot per sea-state bin or one per clip folder inside each bin.",
    )
    parser.add_argument("--direction-group", default=None, help="Optionally process one direction group.")
    parser.add_argument("--sea-state-bin", type=int, default=None, help="Optionally process one bin number.")
    parser.add_argument("--manifest", default=None, help="Download manifest CSV for matching sea-state metadata.")
    parser.add_argument("--candidate-csv", default=None, help="Candidate table CSV fallback for sea-state metadata.")
    parser.add_argument("--baseline-json", default=None, help="Optional LabelMe baseline annotation JSON.")
    parser.add_argument(
        "--reference-shoreline-json",
        default=None,
        help="Known-good shoreline JSON used to reject intersections on wrong contour branches.",
    )
    parser.add_argument("--baseline-label", default="baseline", help="Shape label to read from --baseline-json.")
    parser.add_argument(
        "--baseline-min-vertex-spacing",
        type=float,
        default=5.0,
        help="Drop near-duplicate baseline vertices closer than this many pixels after ordering.",
    )
    parser.add_argument(
        "--baseline-order",
        choices=["annotation", "x", "y"],
        default="x",
        help="Order annotated baseline vertices; x is appropriate for Walton Lighthouse.",
    )
    parser.add_argument(
        "--normal-direction",
        choices=["down", "up", "left", "right"],
        default="down",
        help="Side of an annotated baseline toward which transects are cast.",
    )
    parser.add_argument(
        "--baseline-normal-mode",
        choices=["perpendicular", "fixed"],
        default="perpendicular",
        help=(
            "For annotated baselines, cast transects perpendicular to each "
            "baseline segment or in the fixed --normal-direction. Use fixed "
            "for Seabright's nearly horizontal vegetation baseline."
        ),
    )
    parser.add_argument("--baseline-x", type=float, default=0.0, help="Left vertical baseline x coordinate in pixels.")
    parser.add_argument("--baseline-y-min", type=float, default=0.0, help="Top of fixed baseline y range.")
    parser.add_argument("--baseline-y-max", type=float, default=959.0, help="Bottom of fixed baseline y range.")
    parser.add_argument("--baseline-samples", type=int, default=240, help="Number of horizontal transects.")
    parser.add_argument(
        "--baseline-coordinate-min",
        type=float,
        default=None,
        help="Ignore annotated-baseline transects before this alongshore distance.",
    )
    parser.add_argument(
        "--baseline-coordinate-max",
        type=float,
        default=None,
        help="Ignore annotated-baseline transects after this alongshore distance.",
    )
    parser.add_argument(
        "--intersection-choice",
        choices=["first", "nearest", "median", "farthest"],
        default="first",
        help="Which horizontal-transect intersection to use. 'first'/'nearest' means first hit from the baseline side.",
    )
    parser.add_argument(
        "--max-transect-distance",
        type=float,
        default=None,
        help="Reject intersections farther than this many pixels from the baseline.",
    )
    parser.add_argument(
        "--max-reference-deviation",
        type=float,
        default=None,
        help="Maximum distance from the reference shoreline at each transect.",
    )
    parser.add_argument("--top-margin", type=float, default=10.0, help="Drop shoreline points this close to image top.")
    parser.add_argument("--bottom-margin", type=float, default=10.0, help="Drop shoreline points this close to image bottom.")
    parser.add_argument("--left-margin", type=float, default=0.0, help="Drop shoreline points this close to image left.")
    parser.add_argument("--right-margin", type=float, default=10.0, help="Drop shoreline points this close to image right.")
    parser.add_argument("--max-gap", type=float, default=80.0, help="Split polylines at point-to-point jumps above this many pixels.")
    parser.add_argument("--min-points", type=int, default=25, help="Minimum points in a cleaned shoreline component.")
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0,
        help="OpenCV polyline simplification tolerance in pixels before intersections.",
    )
    parser.add_argument(
        "--polyline-smooth-window",
        type=int,
        default=1,
        help="Moving-average window for shoreline coordinates before intersections.",
    )
    parser.add_argument("--interpolate-gap", type=int, default=3, help="Interpolate NaN gaps up to this many samples.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving median window along each curve.")
    parser.add_argument(
        "--smooth-mean-window",
        type=int,
        default=1,
        help="Hann-weighted smoothing window applied after the moving median.",
    )
    parser.add_argument(
        "--box-smooth-window",
        type=int,
        default=1,
        help="Smoothing window for plotted median and central-band boundaries.",
    )
    parser.add_argument("--despike-window", type=int, default=9, help="Local window used to remove shoreline knots/spikes.")
    parser.add_argument("--despike-threshold-px", type=float, default=90.0, help="Minimum local jump in pixels before a point is treated as noise.")
    parser.add_argument("--despike-mad-scale", type=float, default=6.0, help="Robust MAD multiplier for local spike removal.")
    parser.add_argument("--min-curve-valid-fraction", type=float, default=0.08, help="Minimum valid transect fraction per frame curve.")
    parser.add_argument("--min-sample-valid-fraction", type=float, default=0.10, help="Minimum valid frame fraction per baseline sample.")
    parser.add_argument("--min-curves", type=int, default=10, help="Minimum usable curves required for a bin boxplot.")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum valid baseline samples required for a bin boxplot.")
    parser.add_argument("--max-json-per-bin", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional fixed y-axis max for distance plots.")
    parser.add_argument(
        "--central-band-percent",
        type=float,
        default=50.0,
        help="Functional-depth central band percentage to shade.",
    )
    parser.add_argument(
        "--show-outlier-lines",
        action="store_true",
        help="Draw outlier shoreline curves on the curve-boxplot PNG.",
    )
    parser.add_argument(
        "--transect-preview-length",
        type=float,
        default=300.0,
        help="Transect length shown in baseline_transects.png.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
