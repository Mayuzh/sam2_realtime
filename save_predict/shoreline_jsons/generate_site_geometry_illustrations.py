"""Generate matched transect and spaghetti illustrations for Jennette's Pier and Seabright."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_curve_boxplots_by_bin import load_baseline_points, sample_polyline_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "curve_boxplot_outputs" / "matched_site_illustrations"
DEFAULT_SEABRIGHT_BASELINE = SCRIPT_DIR / "baseline" / "walton_lighthouse-2025-07-26_012153Z_000701.json"
DEFAULT_SEABRIGHT_CURVES = (
    SCRIPT_DIR
    / "curve_boxplot_outputs"
    / "seabright"
    / "south_easterly_onshore"
    / "bin_2"
    / "curves.csv"
)
DEFAULT_JENNETTE_CURVES = (
    SCRIPT_DIR
    / "curve_boxplot_outputs_per_clip"
    / "jennettes_pier"
    / "north_easterly_onshore"
    / "bin_10"
    / "jennette_north-2025-08-23_165135Z"
    / "curves.csv"
)


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def curve_columns(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = []
    for column in frame.columns:
        if not re.match(r"^[ys]_-?\d+(?:\.\d+)?$", column):
            continue
        try:
            columns.append((column, float(column.split("_", 1)[1])))
        except ValueError:
            continue
    columns = sorted(columns, key=lambda item: item[1])
    return [column for column, _ in columns], np.asarray([value for _, value in columns], dtype=float)


def read_image_path_from_json(json_path: Path) -> Path | None:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    image_name = str(data.get("imagePath") or "").strip()
    if image_name:
        candidate = json_path.parent / image_name
        if candidate.exists():
            return candidate
    candidate = json_path.with_suffix(".png")
    if candidate.exists():
        return candidate
    candidate = json_path.with_suffix(".jpg")
    if candidate.exists():
        return candidate
    return None


def video_path_for_json(json_path: Path) -> Path | None:
    parts = json_path.parts
    try:
        idx = parts.index("shoreline_outputs")
        location = parts[idx + 1]
        direction_group = parts[idx + 2]
        bin_name = parts[idx + 3]
        video_stem = parts[idx + 4]
    except (ValueError, IndexError):
        return None
    candidate = SCRIPT_DIR / "downloaded_webcoos_clips" / location / direction_group / bin_name / f"{video_stem}.mp4"
    return candidate if candidate.exists() else None


def frame_number_from_json(json_path: Path) -> int:
    match = re.search(r"_(\d+)$", json_path.stem)
    return int(match.group(1)) if match else 0


def read_video_frame(json_path: Path) -> tuple[np.ndarray | None, str]:
    video_path = video_path_for_json(json_path)
    if video_path is None:
        return None, ""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, str(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number_from_json(json_path))
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        return None, str(video_path)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), str(video_path)


def load_background(json_path: Path) -> tuple[np.ndarray | None, int, int, str]:
    image_path = read_image_path_from_json(json_path)
    if image_path is not None:
        image = plt.imread(image_path)
        height, width = image.shape[:2]
        return image, width, height, str(image_path)

    image, source = read_video_frame(json_path)
    if image is not None:
        height, width = image.shape[:2]
        return image, width, height, source

    return None, 1280, 960, source


def evenly_sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit).round().astype(int)


def smooth_points(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 2 or len(points) < window:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
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


def trim_curve_tail(points: np.ndarray, site_key: str) -> np.ndarray:
    if site_key != "seabright":
        return points
    keep = points[:, 1] <= 700
    if keep.sum() >= 2:
        points = points[keep]
    return points


def select_seabright_shoreline_component(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    split_after = np.where(step > 100.0)[0] + 1
    components = [component for component in np.split(points, split_after) if len(component) >= 2]
    components = [component for component in components if float(np.nanmedian(component[:, 1])) <= 520.0]
    if not components:
        return np.empty((0, 2), dtype=float)
    return max(components, key=len)


def load_curves(curves_csv: Path) -> tuple[pd.DataFrame, list[str], np.ndarray, list[Path]]:
    frame = pd.read_csv(curves_csv)
    columns, coordinates = curve_columns(frame)
    if not columns:
        raise ValueError(f"No y_* or s_* columns found in {curves_csv}")
    json_paths = [Path(value) for value in frame["json_path"].dropna().astype(str)]
    json_paths = [path for path in json_paths if path.exists()]
    if not json_paths:
        raise ValueError(f"No existing source JSON paths found in {curves_csv}")
    return frame, columns, coordinates, json_paths


def seabright_geometry(coordinates: np.ndarray, baseline_json: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_points, _, _, _ = load_baseline_points(
        baseline_json,
        label="baseline",
        order="x",
        min_vertex_spacing=0.0,
    )
    full_coordinate, origins, normals = sample_polyline_baseline(
        baseline_points,
        samples=max(2, len(coordinates)),
        normal_direction="down",
        normal_mode="fixed",
    )
    origin_x = np.interp(coordinates, full_coordinate, origins[:, 0])
    origin_y = np.interp(coordinates, full_coordinate, origins[:, 1])
    normal_x = np.interp(coordinates, full_coordinate, normals[:, 0])
    normal_y = np.interp(coordinates, full_coordinate, normals[:, 1])
    return np.column_stack([origin_x, origin_y]), np.column_stack([normal_x, normal_y]), baseline_points


def jennette_geometry(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.column_stack([np.zeros_like(coordinates), coordinates])
    normals = np.tile(np.array([[1.0, 0.0]], dtype=float), (len(coordinates), 1))
    baseline_points = np.array([[0.0, float(np.nanmin(coordinates))], [0.0, float(np.nanmax(coordinates))]])
    return origins, normals, baseline_points


def points_from_distances(
    distances: np.ndarray,
    origins: np.ndarray,
    normals: np.ndarray,
    site_key: str,
    smooth_window: int,
) -> np.ndarray:
    valid = np.isfinite(distances)
    if valid.sum() < 2:
        return np.empty((0, 2), dtype=float)
    points = origins[valid] + normals[valid] * distances[valid, None]
    points = trim_curve_tail(points, site_key)
    if site_key == "seabright":
        points = select_seabright_shoreline_component(points)
    return smooth_points(points, smooth_window)


def curve_is_displayable(points: np.ndarray, site_key: str) -> bool:
    if len(points) < 2:
        return False
    if site_key != "seabright":
        return True
    y_span = float(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))
    median_y = float(np.nanmedian(points[:, 1]))
    step = np.diff(points, axis=0)
    max_step = float(np.nanmax(np.hypot(step[:, 0], step[:, 1]))) if len(step) else 0.0
    return median_y <= 520.0 and y_span <= 280.0 and max_step <= 90.0


def median_curve(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(values, axis=0)


def add_background(ax: plt.Axes, image: np.ndarray | None) -> None:
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("white")


def set_image_axes(ax: plt.Axes, width: int, height: int, site_key: str) -> None:
    ax.set_xlim(0, width - 1)
    if site_key == "seabright":
        ax.set_ylim(min(720, height - 1), 0)
    else:
        ax.set_ylim(height - 1, 0)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")


def save_transects(
    site_label: str,
    site_key: str,
    curves_csv: Path,
    baseline_json: Path | None,
    output_path: Path,
    transect_step_count: int,
    transect_length: float,
    smooth_window: int,
) -> None:
    frame, columns, coordinates, json_paths = load_curves(curves_csv)
    image, width, height, _ = load_background(json_paths[0])
    if site_key == "seabright":
        origins, normals, baseline_points = seabright_geometry(coordinates, baseline_json or DEFAULT_SEABRIGHT_BASELINE)
    else:
        origins, normals, baseline_points = jennette_geometry(coordinates)

    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    representative = median_curve(values)
    shoreline_points = points_from_distances(representative, origins, normals, site_key, smooth_window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    add_background(ax, image)
    ax.plot(baseline_points[:, 0], baseline_points[:, 1], color="red", linewidth=7, label="Baseline")

    valid = np.isfinite(representative)
    valid_indices = np.where(valid)[0]
    selected = valid_indices[evenly_sample_indices(len(valid_indices), transect_step_count)]
    first_ray = True
    first_hit = True
    for index in selected:
        origin = origins[index]
        normal = normals[index]
        ray_end = origin + normal * transect_length
        ax.plot(
            [origin[0], ray_end[0]],
            [origin[1], ray_end[1]],
            color="limegreen",
            linewidth=0.9,
            alpha=0.62,
            label="Perpendicular transect" if first_ray else None,
        )
        first_ray = False
        hit = origin + normal * representative[index]
        show_hit = True
        if site_key == "seabright":
            if hit[1] > 700 or len(shoreline_points) < 2:
                show_hit = False
            else:
                distance_to_drawn_curve = np.nanmin(np.hypot(shoreline_points[:, 0] - hit[0], shoreline_points[:, 1] - hit[1]))
                show_hit = bool(distance_to_drawn_curve <= 40.0)
        if show_hit:
            ax.scatter(
                hit[0],
                hit[1],
                s=38,
                color="orange",
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
                label="First intersection" if first_hit else None,
            )
            first_hit = False

    if len(shoreline_points):
        ax.plot(shoreline_points[:, 0], shoreline_points[:, 1], color="blue", linewidth=3.5, label="Shoreline")

    set_image_axes(ax, width, height, site_key)
    ax.set_title(f"{site_label}: baseline, transects, and first shoreline intersections")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_spaghetti(
    site_label: str,
    site_key: str,
    curves_csv: Path,
    baseline_json: Path | None,
    output_path: Path,
    max_curves: int,
    smooth_window: int,
) -> None:
    frame, columns, coordinates, json_paths = load_curves(curves_csv)
    image, width, height, _ = load_background(json_paths[0])
    if site_key == "seabright":
        origins, normals, baseline_points = seabright_geometry(coordinates, baseline_json or DEFAULT_SEABRIGHT_BASELINE)
    else:
        origins, normals, baseline_points = jennette_geometry(coordinates)

    selected = evenly_sample_indices(len(frame), max_curves)
    values = frame.iloc[selected][columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    add_background(ax, image)
    ax.plot(baseline_points[:, 0], baseline_points[:, 1], color="red", linewidth=7, label="Baseline")
    colors = plt.cm.turbo(np.linspace(0, 1, len(values)))
    line_count = 0
    for distances, color in zip(values, colors):
        points = points_from_distances(distances, origins, normals, site_key, smooth_window)
        if not curve_is_displayable(points, site_key):
            continue
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.25, alpha=0.28)
        line_count += 1

    ax.plot([], [], color=plt.cm.turbo(0.55), linewidth=1.25, alpha=0.55, label="Shoreline curves")
    set_image_axes(ax, width, height, site_key)
    ax.set_title(f"{site_label}: shoreline ensemble overlaid on frame ({line_count} curves shown)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    seabright_curves = resolve_path(args.seabright_curves, DEFAULT_SEABRIGHT_CURVES)
    jennette_curves = resolve_path(args.jennette_curves, DEFAULT_JENNETTE_CURVES)
    seabright_baseline = resolve_path(args.seabright_baseline, DEFAULT_SEABRIGHT_BASELINE)

    jobs = [
        (
            save_transects,
            "Seabright",
            "seabright",
            seabright_curves,
            seabright_baseline,
            output_dir / "seabright_transects_with_baseline.png",
        ),
        (
            save_spaghetti,
            "Seabright",
            "seabright",
            seabright_curves,
            seabright_baseline,
            output_dir / "seabright_spaghetti_with_baseline.png",
        ),
        (
            save_transects,
            "Jennette's Pier",
            "jennette",
            jennette_curves,
            None,
            output_dir / "jennettes_pier_transects_with_baseline.png",
        ),
        (
            save_spaghetti,
            "Jennette's Pier",
            "jennette",
            jennette_curves,
            None,
            output_dir / "jennettes_pier_spaghetti_with_baseline.png",
        ),
    ]

    for function, site_label, site_key, curves_csv, baseline_json, output_path in jobs:
        if function is save_transects:
            function(
                site_label,
                site_key,
                curves_csv,
                baseline_json,
                output_path,
                args.transect_count,
                args.transect_length,
                args.smooth_window,
            )
        else:
            function(
                site_label,
                site_key,
                curves_csv,
                baseline_json,
                output_path,
                args.max_curves,
                args.smooth_window,
            )
        print(f"Wrote: {output_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seabright-curves", default=None)
    parser.add_argument("--jennette-curves", default=None)
    parser.add_argument("--seabright-baseline", default=None)
    parser.add_argument("--max-curves", type=int, default=300)
    parser.add_argument("--transect-count", type=int, default=22)
    parser.add_argument("--transect-length", type=float, default=430.0)
    parser.add_argument("--smooth-window", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
