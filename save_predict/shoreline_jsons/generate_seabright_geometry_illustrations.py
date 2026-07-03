"""Generate Seabright baseline/intersection and shoreline-ensemble illustrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_curve_boxplots_by_bin import (
    BaselineGeometry,
    clean_polylines,
    load_baseline_points,
    load_labelme_shorelines,
    sample_polyline_baseline,
    shoreline_to_baseline_curve,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_JSON = (
    SCRIPT_DIR / "baseline" / "walton_lighthouse-2025-07-26_012153Z_000701.json"
)
DEFAULT_REFERENCE_JSON = (
    SCRIPT_DIR
    / "shoreline_outputs"
    / "seabright"
    / "south_easterly_onshore"
    / "bin_2"
    / "walton_lighthouse-2025-07-26_012153Z"
    / "walton_lighthouse-2025-07-26_012153Z_000701.json"
)
DEFAULT_CURVES_CSV = (
    SCRIPT_DIR
    / "curve_boxplot_outputs"
    / "seabright"
    / "south_easterly_onshore"
    / "bin_2"
    / "curves.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "curve_boxplot_outputs" / "seabright"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def load_geometry(path: Path, samples: int) -> BaselineGeometry:
    points, width, height, image_path = load_baseline_points(
        path,
        label="baseline",
        order="x",
    )
    coordinate, origins, normals = sample_polyline_baseline(
        points,
        samples=samples,
        normal_direction="down",
    )
    return BaselineGeometry(
        coordinate=coordinate,
        origins=origins,
        normals=normals,
        points=points,
        mode="polyline",
        source_path=path,
        image_path=image_path,
        image_width=width,
        image_height=height,
    )


def cleaned_reference(reference_json: Path) -> tuple[list[np.ndarray], int, int, Path]:
    polylines, width, height = load_labelme_shorelines(reference_json)
    cleaned = clean_polylines(
        polylines,
        width=width,
        height=height,
        top_margin=10,
        bottom_margin=15,
        left_margin=5,
        right_margin=15,
        max_gap=80,
        min_points=25,
        simplify_tolerance=1.5,
    )
    image_path = reference_json.with_suffix(".png")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing reference PNG: {image_path}")
    return cleaned, width, height, image_path


def save_debug_rays(
    baseline: BaselineGeometry,
    polylines: list[np.ndarray],
    image_path: Path,
    output_path: Path,
) -> None:
    curve = shoreline_to_baseline_curve(
        polylines,
        baseline=baseline,
        prefer="first",
        max_distance=350,
    )
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for line in polylines:
        ax.plot(line[:, 0], line[:, 1], color="blue", linewidth=1.5, alpha=0.9)
    ax.plot(
        baseline.points[:, 0],
        baseline.points[:, 1],
        color="red",
        linewidth=5,
        label="Baseline",
    )

    step = max(1, len(curve) // 18)
    first_ray = True
    first_point = True
    for origin, normal, distance in zip(
        baseline.origins[::step],
        baseline.normals[::step],
        curve[::step],
    ):
        ray_end = origin + normal * 350
        ax.plot(
            [origin[0], ray_end[0]],
            [origin[1], ray_end[1]],
            color="lime",
            linewidth=0.7,
            alpha=0.55,
            label="Perpendicular transect" if first_ray else None,
        )
        first_ray = False
        if np.isfinite(distance):
            hit = origin + normal * distance
            ax.scatter(
                hit[0],
                hit[1],
                s=35,
                color="orange",
                edgecolor="black",
                zorder=5,
                label="First intersection" if first_point else None,
            )
            first_point = False

    ax.plot([], [], color="blue", linewidth=1.5, label="Shoreline")
    ax.set_xlim(0, baseline.image_width - 1)
    ax.set_ylim(baseline.image_height - 1, 0)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title("Seabright baseline, transects, and first shoreline intersections")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def curve_columns(frame: pd.DataFrame) -> list[tuple[str, float]]:
    columns = []
    for column in frame.columns:
        if not column.startswith("s_"):
            continue
        try:
            columns.append((column, float(column[2:])))
        except ValueError:
            continue
    return sorted(columns, key=lambda item: item[1])


def save_ensemble(
    baseline: BaselineGeometry,
    curves_csv: Path,
    image_path: Path,
    output_path: Path,
    max_curves: int,
) -> None:
    frame = pd.read_csv(curves_csv)
    columns = curve_columns(frame)
    if not columns:
        raise ValueError(f"No s_* curve columns in {curves_csv}")
    if len(frame) > max_curves:
        indexes = np.linspace(0, len(frame) - 1, max_curves, dtype=int)
        frame = frame.iloc[indexes]

    coordinate = np.asarray([value for _, value in columns], dtype=float)
    baseline_indexes = np.asarray(
        [int(np.argmin(np.abs(baseline.coordinate - value))) for value in coordinate]
    )
    origins = baseline.origins[baseline_indexes]
    normals = baseline.normals[baseline_indexes]
    values = frame[[column for column, _ in columns]].to_numpy(dtype=float)

    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    colors = plt.cm.turbo(np.linspace(0, 1, len(values)))
    for distances, color in zip(values, colors):
        valid = np.isfinite(distances)
        if valid.sum() < 2:
            continue
        points = origins[valid] + normals[valid] * distances[valid, None]
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.55, alpha=0.14)

    ax.plot(
        baseline.points[:, 0],
        baseline.points[:, 1],
        color="red",
        linewidth=5,
        label="Baseline",
    )
    ax.set_xlim(0, baseline.image_width - 1)
    ax.set_ylim(baseline.image_height - 1, 0)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title(f"Seabright extracted shoreline ensemble ({len(values)} curves shown)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    baseline_path = resolve_path(args.baseline_json, DEFAULT_BASELINE_JSON)
    reference_path = resolve_path(args.reference_json, DEFAULT_REFERENCE_JSON)
    curves_path = resolve_path(args.curves_csv, DEFAULT_CURVES_CSV)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_geometry(baseline_path, args.baseline_samples)
    polylines, _, _, image_path = cleaned_reference(reference_path)
    debug_path = output_dir / "seabright_baseline_shoreline_intersections.png"
    ensemble_path = output_dir / "seabright_shoreline_ensemble_overlay.png"
    save_debug_rays(baseline, polylines, image_path, debug_path)
    save_ensemble(baseline, curves_path, image_path, ensemble_path, args.max_curves)
    print(f"Wrote: {debug_path}")
    print(f"Wrote: {ensemble_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--reference-json", default=None)
    parser.add_argument("--curves-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-samples", type=int, default=240)
    parser.add_argument("--max-curves", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
