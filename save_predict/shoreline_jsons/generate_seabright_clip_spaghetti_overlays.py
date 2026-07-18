"""Generate Seabright beach-view spaghetti overlays from cleaned per-clip curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from generate_curve_boxplots_by_bin import load_baseline_points, sample_polyline_baseline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "seabright" / "curve_boxplot_summary.csv"
DEFAULT_BASELINE = SCRIPT_DIR / "baseline" / "walton_lighthouse-2025-07-26_012153Z_000701.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "seabright_spaghetti_overlays"
DEFAULT_VIDEO_ROOT = SCRIPT_DIR / "downloaded_webcoos_clips" / "seabright"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def coordinate_columns(columns: list[str]) -> tuple[list[str], np.ndarray]:
    curve_cols = [col for col in columns if re.match(r"^[ys]_-?\d+", col)]
    curve_cols = sorted(curve_cols, key=lambda col: float(col.split("_", 1)[1]))
    coordinate = np.asarray([float(col.split("_", 1)[1]) for col in curve_cols], dtype=float)
    return curve_cols, coordinate


def evenly_sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit).round().astype(int)


def load_frame_image(json_path: Path) -> Path | None:
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
    if location != "seabright":
        return None
    candidate = DEFAULT_VIDEO_ROOT / direction_group / bin_name / f"{video_stem}.mp4"
    return candidate if candidate.exists() else None


def frame_number_from_json(json_path: Path) -> int:
    match = re.search(r"_(\d+)$", json_path.stem)
    return int(match.group(1)) if match else 0


def load_video_frame(json_path: Path) -> tuple[np.ndarray, str] | tuple[None, str]:
    video_path = video_path_for_json(json_path)
    if video_path is None:
        return None, ""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, str(video_path)
    frame_index = frame_number_from_json(json_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        return None, str(video_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, str(video_path)


def baseline_for_coordinates(baseline_json: Path, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, _, _, _ = load_baseline_points(baseline_json, label="baseline", order="x")
    full_coordinate, origins, normals = sample_polyline_baseline(
        points,
        samples=max(2, len(coordinates)),
        normal_direction="down",
        normal_mode="fixed",
    )
    origin_x = np.interp(coordinates, full_coordinate, origins[:, 0])
    origin_y = np.interp(coordinates, full_coordinate, origins[:, 1])
    normal_x = np.interp(coordinates, full_coordinate, normals[:, 0])
    normal_y = np.interp(coordinates, full_coordinate, normals[:, 1])
    return np.column_stack([origin_x, origin_y]), np.column_stack([normal_x, normal_y]), points


def plot_overlay(
    curves_csv: Path,
    baseline_json: Path,
    output_path: Path,
    title: str,
    max_lines: int,
) -> dict[str, object]:
    curves = pd.read_csv(curves_csv)
    json_paths = [Path(value) for value in curves["json_path"].dropna().astype(str)]
    json_paths = [path for path in json_paths if path.exists()]
    if not json_paths:
        raise ValueError(f"No source JSON paths found in {curves_csv}")

    curve_cols, coordinates = coordinate_columns(list(curves.columns))
    origins, normals, baseline_points = baseline_for_coordinates(baseline_json, coordinates)
    selected = evenly_sample_indices(len(curves), max_lines)

    frame_path = load_frame_image(json_paths[0])
    frame_source = ""
    if frame_path is not None:
        image = plt.imread(frame_path)
        frame_source = str(frame_path)
        height, width = image.shape[:2]
    else:
        image, frame_source = load_video_frame(json_paths[0])
        if image is not None:
            height, width = image.shape[:2]
        else:
            width = int(max(1280, np.nanmax(origins[:, 0] + 900)))
            height = int(max(960, np.nanmax(origins[:, 1] + 900)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    if image is not None:
        plt.imshow(image)
    else:
        plt.gca().set_facecolor("white")
    plt.plot(baseline_points[:, 0], baseline_points[:, 1], color="red", linewidth=2.5, alpha=0.9)

    for index in selected:
        distances = curves.iloc[int(index)][curve_cols].to_numpy(dtype=float)
        valid = np.isfinite(distances)
        if valid.sum() < 2:
            continue
        pts = origins[valid] + normals[valid] * distances[valid, None]
        plt.plot(pts[:, 0], pts[:, 1], color="lime", linewidth=0.8, alpha=0.22)

    plt.title(title)
    plt.xlim(0, width - 1)
    plt.ylim(height - 1, 0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "overlay_path": str(output_path),
        "frame_path": frame_source,
        "frame_found": bool(frame_source),
        "n_source_jsons": len(json_paths),
        "n_overlay_curves": int(len(selected)),
    }


def run(args: argparse.Namespace) -> int:
    summary_path = resolve_path(args.summary_csv, DEFAULT_SUMMARY)
    baseline_json = resolve_path(args.baseline_json, DEFAULT_BASELINE)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT)
    summary = pd.read_csv(summary_path)

    rows = []
    for _, row in summary.iterrows():
        direction_group = str(row["direction_group"])
        sea_state_bin = int(row["sea_state_bin"])
        video_stem = str(row["video_stem"])
        output_path = output_root / f"{direction_group}_bin_{sea_state_bin:02d}_{video_stem}_spaghetti.png"
        title = f"{direction_group} bin_{sea_state_bin} {video_stem}"
        try:
            info = plot_overlay(Path(str(row["curves_csv"])), baseline_json, output_path, title, args.max_lines)
        except Exception as exc:
            print(f"[skip] {video_stem}: {exc}")
            continue
        info.update(
            {
                "direction_group": direction_group,
                "sea_state_bin": sea_state_bin,
                "video_stem": video_stem,
                "curve_boxplot_path": row.get("plot_path", ""),
                "curve_boxplot_variance": row.get("curve_boxplot_variance", ""),
            }
        )
        rows.append(info)
        print(f"[ok] {output_path}")

    if rows:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_root / "spaghetti_overlay_index.csv", index=False)
    return 0 if rows else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-lines", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
