"""Generate beach-view shoreline spaghetti overlays for selected Jennette clips."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "jennettes_pier" / "curve_boxplot_summary.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "jennettes_pier_diagnostics"

DEFAULT_CASES = [
    ("offshore", 2, "jennette_north-2025-11-22_155334Z"),
    ("offshore", 4, "jennette_north-2025-04-29_232044Z"),
    ("offshore", 6, "jennette_north-2025-11-05_205212Z"),
    ("south_easterly_onshore", 2, "jennette_north-2025-05-20_000401Z"),
    ("north_easterly_onshore", 10, "jennette_north-2025-08-23_165135Z"),
    ("south_easterly_onshore", 8, "jennette_north-2025-04-11_172322Z"),
    ("south_easterly_onshore", 3, "jennette_north-2025-04-23_101948Z"),
]


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_shoreline_polylines(json_path: Path) -> tuple[list[np.ndarray], int, int, Path | None]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    width = int(data.get("imageWidth") or 1280)
    height = int(data.get("imageHeight") or 960)
    image_path = None
    image_name = str(data.get("imagePath") or "").strip()
    if image_name:
        candidate = json_path.parent / image_name
        if candidate.exists():
            image_path = candidate
    if image_path is None:
        candidate = json_path.with_suffix(".png")
        if candidate.exists():
            image_path = candidate

    polylines = []
    for shape in data.get("shapes", []):
        if shape.get("label") != "shoreline":
            continue
        points = np.asarray(shape.get("points", []), dtype=float)
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] >= 2:
            polylines.append(points[:, :2])
    return polylines, width, height, image_path


def evenly_sample(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit).round().astype(int)
    return [paths[int(index)] for index in indices]


def coordinate_columns(columns: list[str]) -> tuple[list[str], np.ndarray]:
    curve_cols = [col for col in columns if re.match(r"^[ys]_-?\d+", col)]
    curve_cols = sorted(curve_cols, key=lambda col: float(col.split("_", 1)[1]))
    coordinate = np.asarray([float(col.split("_", 1)[1]) for col in curve_cols], dtype=float)
    return curve_cols, coordinate


def plot_overlay(
    curves_csv: Path,
    output_path: Path,
    title: str,
    max_lines: int,
) -> dict[str, object]:
    curves = pd.read_csv(curves_csv)
    json_paths = [Path(value) for value in curves["json_path"].dropna().astype(str)]
    json_paths = [path for path in json_paths if path.exists()]
    if not json_paths:
        raise ValueError(f"No source JSON paths found in {curves_csv}")

    _, width, height, image_path = load_shoreline_polylines(json_paths[0])
    if image_path is None:
        raise ValueError(f"No frame image found for {json_paths[0]}")

    curve_cols, baseline_coordinate = coordinate_columns(list(curves.columns))
    selected = evenly_sample(list(range(len(curves))), max_lines)

    image = plt.imread(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    for index in selected:
        distances = curves.iloc[index][curve_cols].to_numpy(dtype=float)
        valid = np.isfinite(distances)
        if valid.sum() < 2:
            continue
        # For Jennette's fixed left baseline, x is distance from baseline and
        # y is the baseline coordinate. This draws the exact cleaned curves
        # used by the curve boxplot back onto the camera image.
        plt.plot(distances[valid], baseline_coordinate[valid], color="lime", linewidth=0.8, alpha=0.25)

    plt.axvline(0, color="red", linewidth=3, alpha=0.8)
    plt.title(title)
    plt.xlim(0, width - 1)
    plt.ylim(height - 1, 0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "overlay_path": str(output_path),
        "frame_path": str(image_path),
        "n_source_jsons": len(json_paths),
        "n_overlay_jsons": len(selected),
        "overlay_type": "cleaned_curve_coordinates",
    }


def run(args: argparse.Namespace) -> int:
    summary_path = resolve_path(args.summary_csv, DEFAULT_SUMMARY)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT)
    summary = pd.read_csv(summary_path)
    summary["sea_state_bin"] = pd.to_numeric(summary["sea_state_bin"], errors="coerce").astype("Int64")

    requested = DEFAULT_CASES
    rows = []
    for direction_group, sea_state_bin, video_stem in requested:
        match = summary[
            (summary["direction_group"] == direction_group)
            & (summary["sea_state_bin"] == sea_state_bin)
            & (summary["video_stem"] == video_stem)
        ]
        if match.empty:
            print(f"[missing] {direction_group}/bin_{sea_state_bin}/{video_stem}")
            continue
        row = match.iloc[0]
        output_path = (
            output_root
            / f"{direction_group}_bin_{sea_state_bin:02d}_{video_stem}_shoreline_overlay.png"
        )
        title = f"{direction_group} bin_{sea_state_bin} {video_stem}"
        info = plot_overlay(Path(str(row["curves_csv"])), output_path, title, args.max_lines)
        info.update(
            {
                "direction_group": direction_group,
                "sea_state_bin": sea_state_bin,
                "video_stem": video_stem,
                "curve_boxplot_path": row["plot_path"],
                "curve_boxplot_variance": row.get("curve_boxplot_variance", ""),
                "sea_state_mean": row.get("sea_state_mean", ""),
            }
        )
        rows.append(info)
        print(f"[ok] {output_path}")

    if rows:
        pd.DataFrame(rows).to_csv(output_root / "diagnostic_overlay_index.csv", index=False)
    return 0 if rows else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-lines", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
