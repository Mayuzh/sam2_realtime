"""Generate one Jennette's curve boxplot per clip from existing per-bin curves.

This is the fast path after ``generate_curve_boxplots_by_bin.py`` has already
converted shoreline JSONs into cleaned 1D curves. It regroups those cleaned
curves by ``video_stem`` so each 10-minute clip gets its own curve boxplot.

Default input:
    curve_boxplot_outputs/jennettes_pier/<direction_group>/bin_<n>/curves.csv

Default output:
    curve_boxplot_outputs_per_clip/jennettes_pier/<direction_group>/bin_<n>/<video_stem>/
        curve_boxplot.png
        curves.csv
        boxplot_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_curve_boxplots_by_bin import (
    CurveRecord,
    build_curve_boxplot,
    build_manifest_lookup,
    central_band_area,
    curve_variance_scalar,
    metadata_for_records,
    natural_key,
    save_group_outputs,
    smooth_boxplot_statistics,
    weighted_moving_average,
)

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BIN_CURVES_ROOT = SCRIPT_DIR / "curve_boxplot_outputs" / "jennettes_pier"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "curve_boxplot_outputs_per_clip" / "jennettes_pier"
DEFAULT_MANIFEST = SCRIPT_DIR / "downloaded_webcoos_clips" / "download_manifest.csv"
DEFAULT_CANDIDATES = SCRIPT_DIR / "candidate_clip_outputs" / "candidate_clip_table.csv"
DEFAULT_X_MAX = 959.0
DEFAULT_Y_MAX = 1280.0


def resolve_path(path_value: str | None, default: Path) -> Path:
    if path_value is None:
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def coordinate_columns(columns: list[str]) -> tuple[list[str], np.ndarray, str]:
    curve_cols = [col for col in columns if re.match(r"^[ys]_-?\d+", col)]
    if not curve_cols:
        raise ValueError("No y_* or s_* curve coordinate columns found")

    def coord_value(name: str) -> float:
        return float(name.split("_", 1)[1])

    curve_cols = sorted(curve_cols, key=coord_value)
    baseline_coordinate = np.asarray([coord_value(col) for col in curve_cols], dtype=float)
    baseline_mode = "polyline" if curve_cols[0].startswith("s_") else "vertical"
    return curve_cols, baseline_coordinate, baseline_mode


def load_clip_records(
    curves_csv: Path,
    direction_group: str,
    bin_name: str,
    curve_smooth_window: int,
) -> tuple[np.ndarray, str, dict[str, list[CurveRecord]]]:
    df = pd.read_csv(curves_csv)
    curve_cols, baseline_coordinate, baseline_mode = coordinate_columns(list(df.columns))

    grouped: dict[str, list[CurveRecord]] = {}
    for _, row in df.iterrows():
        video_stem = str(row.get("video_stem", "")).strip()
        if not video_stem or video_stem == "nan":
            continue
        curve = row[curve_cols].to_numpy(dtype=float)
        if curve_smooth_window > 1:
            curve = weighted_moving_average(curve, curve_smooth_window)
        valid_fraction = float(np.isfinite(curve).mean())
        json_path = Path(str(row.get("json_path", "")))
        grouped.setdefault(video_stem, []).append(
            CurveRecord(
                json_path=json_path,
                video_stem=video_stem,
                direction_group=direction_group,
                bin_name=bin_name,
                curve=curve,
                valid_fraction=valid_fraction,
            )
        )

    return baseline_coordinate, baseline_mode, grouped


def reject_upper_branch_artifacts(
    curves: np.ndarray,
    reference_quantile: float,
    reference_smooth_window: int,
    max_above_reference: float,
) -> tuple[np.ndarray, int]:
    """Remove far upper branches caused by mask edges or closed-contour artifacts.

    The real shoreline branch is usually the lower/nearer family of
    intersections. Wrong polygon edges appear as tall plateaus or horns far
    above that branch. This uses a smoothed lower quantile as the shoreline
    reference and drops only values that sit far above it.
    """
    if max_above_reference <= 0:
        return curves, 0

    cleaned = curves.copy()
    finite_columns = np.isfinite(cleaned).any(axis=0)
    if not finite_columns.any():
        return cleaned, 0

    reference = np.full(cleaned.shape[1], np.nan, dtype=float)
    with np.errstate(all="ignore"):
        reference[finite_columns] = np.nanpercentile(
            cleaned[:, finite_columns],
            reference_quantile,
            axis=0,
        )
    reference = weighted_moving_average(reference, reference_smooth_window)

    high_branch = np.isfinite(cleaned) & np.isfinite(reference)[None, :]
    high_branch &= cleaned > (reference[None, :] + max_above_reference)
    removed_count = int(high_branch.sum())
    cleaned[high_branch] = np.nan
    return cleaned, removed_count


def trim_low_support_tail(
    curves: np.ndarray,
    coordinate: np.ndarray,
    min_tail_support: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if min_tail_support <= 0 or curves.shape[1] < 2:
        return curves, coordinate, 0

    support = np.isfinite(curves).mean(axis=0)
    strong = support >= min_tail_support
    if not strong.any():
        return curves, coordinate, 0

    last_strong_index = int(np.where(strong)[0].max())
    if last_strong_index >= curves.shape[1] - 1:
        return curves, coordinate, 0

    removed = curves.shape[1] - last_strong_index - 1
    return curves[:, : last_strong_index + 1], coordinate[: last_strong_index + 1], int(removed)


def keep_longest_contiguous_coordinate_span(
    curves: np.ndarray,
    coordinate: np.ndarray,
    gap_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if gap_multiplier <= 0 or len(coordinate) < 3:
        return curves, coordinate, 0

    steps = np.diff(coordinate)
    finite_steps = steps[np.isfinite(steps) & (steps > 0)]
    if not len(finite_steps):
        return curves, coordinate, 0

    expected_step = float(np.nanmedian(finite_steps))
    if expected_step <= 0:
        return curves, coordinate, 0

    break_indices = np.where(steps > expected_step * gap_multiplier)[0]
    if not len(break_indices):
        return curves, coordinate, 0

    starts = np.r_[0, break_indices + 1]
    ends = np.r_[break_indices + 1, len(coordinate)]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    removed = int(len(coordinate) - lengths[best])
    return curves[:, starts[best] : ends[best]], coordinate[starts[best] : ends[best]], removed


def truncate_after_large_upward_jump(
    curves: np.ndarray,
    coordinate: np.ndarray,
    max_upward_jump_px: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Trim the plot when the dominant branch jumps to a different contour.

    A true shoreline curve should not jump hundreds of pixels offshore between
    adjacent transects. That pattern means the first-intersection logic has
    switched to a noisy edge or another mask boundary.
    """
    if max_upward_jump_px <= 0 or curves.shape[1] < 2:
        return curves, coordinate, 0

    with np.errstate(all="ignore"):
        median = np.nanmedian(curves, axis=0)
    finite = np.isfinite(median)
    if finite.sum() < 2:
        return curves, coordinate, 0

    finite_indices = np.where(finite)[0]
    median_finite = median[finite]
    jumps = np.diff(median_finite)
    bad = np.where(jumps > max_upward_jump_px)[0]
    if not bad.size:
        return curves, coordinate, 0

    # Drop the destination of the first bad jump and everything after it.
    cut_index = int(finite_indices[bad[0] + 1])
    if cut_index < 2:
        return curves, coordinate, 0
    removed = curves.shape[1] - cut_index
    return curves[:, :cut_index], coordinate[:cut_index], int(removed)


def truncate_after_post_minimum_rise(
    curves: np.ndarray,
    coordinate: np.ndarray,
    max_rise_after_min_px: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_rise_after_min_px <= 0 or curves.shape[1] < 3:
        return curves, coordinate, 0

    with np.errstate(all="ignore"):
        median = np.nanmedian(curves, axis=0)
    finite = np.isfinite(median)
    if finite.sum() < 3:
        return curves, coordinate, 0

    finite_indices = np.where(finite)[0]
    median_finite = median[finite]
    min_pos = int(np.nanargmin(median_finite))
    min_index = int(finite_indices[min_pos])
    if min_pos >= len(median_finite) - 2:
        return curves, coordinate, 0

    later_max = float(np.nanmax(median_finite[min_pos + 1 :]))
    if later_max - float(median_finite[min_pos]) <= max_rise_after_min_px:
        return curves, coordinate, 0

    cut_index = max(2, min_index + 1)
    removed = curves.shape[1] - cut_index
    return curves[:, :cut_index], coordinate[:cut_index], int(removed)


def apply_pointwise_central_band(
    box: dict[str, np.ndarray],
    curves: np.ndarray,
    central_percent: float,
) -> dict[str, np.ndarray]:
    if central_percent <= 0:
        return box
    lower_q = (100.0 - central_percent) / 2.0
    upper_q = 100.0 - lower_q
    result = dict(box)
    with np.errstate(all="ignore"):
        result["lower"] = np.nanpercentile(curves, lower_q, axis=0)
        result["upper"] = np.nanpercentile(curves, upper_q, axis=0)
        result["median"] = np.nanmedian(curves, axis=0)
    return result


def trim_leading_high_spread(
    curves: np.ndarray,
    coordinate: np.ndarray,
    central_percent: float,
    max_band_width: float,
    stable_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_band_width <= 0 or curves.shape[1] < max(3, stable_samples):
        return curves, coordinate, 0

    lower_q = (100.0 - central_percent) / 2.0
    upper_q = 100.0 - lower_q
    with np.errstate(all="ignore"):
        lower = np.nanpercentile(curves, lower_q, axis=0)
        upper = np.nanpercentile(curves, upper_q, axis=0)
    width = upper - lower
    stable = np.isfinite(width) & (width <= max_band_width)
    if stable[: min(stable_samples, len(stable))].all():
        return curves, coordinate, 0

    stable_samples = max(1, stable_samples)
    for start in range(0, len(stable) - stable_samples + 1):
        if stable[start : start + stable_samples].all():
            if start == 0:
                return curves, coordinate, 0
            return curves[:, start:], coordinate[start:], int(start)
    return curves, coordinate, 0


def plot_clean_boxplot(
    baseline_coordinate: np.ndarray,
    box: dict[str, np.ndarray],
    title: str,
    output_path: Path,
    x_min: float,
    x_max: float,
    y_max: float,
    band_percent: float,
    baseline_mode: str,
    visual_band_scale: float,
    visual_min_band_width: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_box = dict(box)
    if visual_band_scale > 1.0 or visual_min_band_width > 0:
        median = np.asarray(plot_box["median"], dtype=float)
        lower = np.asarray(plot_box["lower"], dtype=float)
        upper = np.asarray(plot_box["upper"], dtype=float)
        half_width = np.maximum(upper - median, median - lower)
        if visual_band_scale > 1.0:
            half_width = half_width * visual_band_scale
        if visual_min_band_width > 0:
            half_width = np.maximum(half_width, visual_min_band_width / 2.0)
        plot_box["lower"] = median - half_width
        plot_box["upper"] = median + half_width

    plt.figure(figsize=(10, 5))
    plt.fill_between(
        baseline_coordinate,
        plot_box["lower"],
        plot_box["upper"],
        alpha=0.3,
        label=f"Central {band_percent:.0f}% band",
    )
    plt.plot(baseline_coordinate, plot_box["median"], linewidth=2, label="Median")
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
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def process_clip_group(
    records: list[CurveRecord],
    baseline_coordinate: np.ndarray,
    baseline_mode: str,
    output_dir: Path,
    title: str,
    min_curves: int,
    min_samples: int,
    min_curve_valid_fraction: float,
    min_sample_valid_fraction: float,
    box_smooth_window: int,
    branch_cleanup: bool,
    branch_reference_quantile: float,
    branch_reference_smooth_window: int,
    branch_max_above_reference: float,
    use_pointwise_median: bool,
    tail_trim: bool,
    tail_min_sample_valid_fraction: float,
    jump_cleanup: bool,
    jump_max_upward_px: float,
    post_min_rise_cleanup: bool,
    post_min_max_rise_px: float,
    central_band_percent: float,
    variance_reference_x: float,
    require_reference_coverage: bool,
    min_coordinate_span: float,
    longest_component_gap_multiplier: float,
    leading_spread_cleanup: bool,
    leading_spread_threshold: float,
    leading_stable_samples: int,
    x_min: float,
    x_max: float,
    y_max: float | None,
    visual_band_scale: float,
    visual_min_band_width: float,
) -> tuple[dict[str, Any] | None, list[CurveRecord], np.ndarray | None, dict[str, np.ndarray] | None]:
    if len(records) < min_curves:
        return None, [], None, None

    curves = np.vstack([record.curve for record in records])
    sample_valid_fraction = np.isfinite(curves).mean(axis=0)
    valid_samples = sample_valid_fraction >= min_sample_valid_fraction
    if valid_samples.sum() < min_samples:
        return None, [], None, None

    curves_valid = curves[:, valid_samples]
    coordinate_valid = baseline_coordinate[valid_samples]
    keep_frames = np.isfinite(curves_valid).mean(axis=1) >= min_curve_valid_fraction
    curves_valid = curves_valid[keep_frames]
    kept_records = [record for record, keep in zip(records, keep_frames) if keep]
    if curves_valid.shape[0] < min_curves:
        return None, [], None, None

    branch_artifact_count = 0
    if branch_cleanup:
        curves_valid, branch_artifact_count = reject_upper_branch_artifacts(
            curves_valid,
            reference_quantile=branch_reference_quantile,
            reference_smooth_window=branch_reference_smooth_window,
            max_above_reference=branch_max_above_reference,
        )
        keep_frames = np.isfinite(curves_valid).mean(axis=1) >= min_curve_valid_fraction
        curves_valid = curves_valid[keep_frames]
        kept_records = [record for record, keep in zip(kept_records, keep_frames) if keep]
        sample_valid_fraction = np.isfinite(curves_valid).mean(axis=0)
        valid_samples = sample_valid_fraction >= min_sample_valid_fraction
        if valid_samples.sum() < min_samples:
            return None, [], None, None
        curves_valid = curves_valid[:, valid_samples]
        coordinate_valid = coordinate_valid[valid_samples]
        if curves_valid.shape[0] < min_curves:
            return None, [], None, None

    tail_trimmed_samples = 0
    if tail_trim:
        curves_valid, coordinate_valid, tail_trimmed_samples = trim_low_support_tail(
            curves_valid,
            coordinate_valid,
            min_tail_support=tail_min_sample_valid_fraction,
        )
        if curves_valid.shape[1] < min_samples:
            return None, [], None, None

    jump_trimmed_samples = 0
    if jump_cleanup:
        curves_valid, coordinate_valid, jump_trimmed_samples = truncate_after_large_upward_jump(
            curves_valid,
            coordinate_valid,
            max_upward_jump_px=jump_max_upward_px,
        )
        if curves_valid.shape[1] < min_samples:
            return None, [], None, None

    post_min_trimmed_samples = 0
    if post_min_rise_cleanup:
        curves_valid, coordinate_valid, post_min_trimmed_samples = truncate_after_post_minimum_rise(
            curves_valid,
            coordinate_valid,
            max_rise_after_min_px=post_min_max_rise_px,
        )
        if curves_valid.shape[1] < min_samples:
            return None, [], None, None

    disconnected_samples_removed = 0
    if longest_component_gap_multiplier > 0:
        curves_valid, coordinate_valid, disconnected_samples_removed = keep_longest_contiguous_coordinate_span(
            curves_valid,
            coordinate_valid,
            gap_multiplier=longest_component_gap_multiplier,
        )
        if curves_valid.shape[1] < min_samples:
            return None, [], None, None

    leading_spread_trimmed_samples = 0
    if leading_spread_cleanup:
        before_trim_curves = curves_valid
        before_trim_coordinate = coordinate_valid
        curves_valid, coordinate_valid, leading_spread_trimmed_samples = trim_leading_high_spread(
            curves_valid,
            coordinate_valid,
            central_percent=central_band_percent,
            max_band_width=leading_spread_threshold,
            stable_samples=leading_stable_samples,
        )
        if curves_valid.shape[1] < min_samples:
            curves_valid = before_trim_curves
            coordinate_valid = before_trim_coordinate
            leading_spread_trimmed_samples = 0

    coordinate_span = float(coordinate_valid.max() - coordinate_valid.min())
    if min_coordinate_span > 0 and coordinate_span < min_coordinate_span:
        return None, [], None, None
    if require_reference_coverage and not (
        float(coordinate_valid.min()) <= variance_reference_x <= float(coordinate_valid.max())
    ):
        return None, [], None, None

    box = build_curve_boxplot(curves_valid)
    if use_pointwise_median:
        with np.errstate(all="ignore"):
            box["median"] = np.nanmedian(curves_valid, axis=0)
    box = apply_pointwise_central_band(box, curves_valid, central_band_percent)
    box = smooth_boxplot_statistics(box, box_smooth_window)
    variance_value = curve_variance_scalar(curves_valid, box["outliers"])
    variance_reference_index = int(np.argmin(np.abs(coordinate_valid - variance_reference_x)))
    variance_at_reference = float(
        np.nanvar(curves_valid[:, variance_reference_index], ddof=0)
    )
    band_area = central_band_area(coordinate_valid, box["lower"], box["upper"])

    plot_clean_boxplot(
        coordinate_valid,
        box,
        title,
        output_dir / "curve_boxplot.png",
        x_min,
        x_max,
        y_max if y_max is not None else DEFAULT_Y_MAX,
        central_band_percent,
        baseline_mode,
        visual_band_scale,
        visual_min_band_width,
    )
    save_group_outputs(output_dir, coordinate_valid, kept_records, curves_valid, box, baseline_mode)

    summary = {
        "n_json_files": len(records),
        "n_usable_curves": int(curves_valid.shape[0]),
        "n_baseline_samples": int(curves_valid.shape[1]),
        "baseline_mode": baseline_mode,
        "baseline_x": 0.0 if baseline_mode == "vertical" else np.nan,
        "baseline_coordinate_min": float(coordinate_valid.min()),
        "baseline_coordinate_max": float(coordinate_valid.max()),
        "baseline_coordinate_span": coordinate_span,
        "curve_boxplot_variance": variance_value,
        "variance_reference_x": variance_reference_x,
        "variance_at_reference_x": variance_at_reference,
        "variance_reference_actual_x": float(coordinate_valid[variance_reference_index]),
        "central_band_area": band_area,
        "outlier_count": int(box["outliers"].sum()),
        "branch_artifact_count": branch_artifact_count,
        "branch_cleanup": branch_cleanup,
        "tail_trimmed_samples": tail_trimmed_samples,
        "tail_trim": tail_trim,
        "jump_trimmed_samples": jump_trimmed_samples,
        "jump_cleanup": jump_cleanup,
        "post_min_trimmed_samples": post_min_trimmed_samples,
        "post_min_rise_cleanup": post_min_rise_cleanup,
        "disconnected_samples_removed": disconnected_samples_removed,
        "longest_component_gap_multiplier": longest_component_gap_multiplier,
        "leading_spread_trimmed_samples": leading_spread_trimmed_samples,
        "leading_spread_cleanup": leading_spread_cleanup,
        "leading_spread_threshold": leading_spread_threshold,
        "central_band_percent": central_band_percent,
        "require_reference_coverage": require_reference_coverage,
        "min_coordinate_span": min_coordinate_span,
        "plot_x_min": x_min,
        "plot_x_max": x_max,
        "plot_y_max": y_max if y_max is not None else DEFAULT_Y_MAX,
        "visual_band_scale": visual_band_scale,
        "visual_min_band_width": visual_min_band_width,
        "plot_path": str(output_dir / "curve_boxplot.png"),
        "curves_csv": str(output_dir / "curves.csv"),
    }
    return summary, kept_records, curves_valid, box


def run(args: argparse.Namespace) -> int:
    bin_curves_root = resolve_path(args.bin_curves_root, DEFAULT_BIN_CURVES_ROOT)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    manifest_path = resolve_path(args.manifest, DEFAULT_MANIFEST)
    candidate_path = resolve_path(args.candidate_csv, DEFAULT_CANDIDATES)
    metadata_lookup = build_manifest_lookup(manifest_path, candidate_path)

    curves_files = sorted(bin_curves_root.rglob("curves.csv"), key=natural_key)
    if args.direction_group:
        curves_files = [p for p in curves_files if p.relative_to(bin_curves_root).parts[0] == args.direction_group]
    if args.sea_state_bin is not None:
        wanted_bin = f"bin_{args.sea_state_bin}"
        curves_files = [p for p in curves_files if p.relative_to(bin_curves_root).parts[1] == wanted_bin]

    if not curves_files:
        print(f"No per-bin curves.csv files found under {bin_curves_root}")
        return 1

    summary_rows: list[dict[str, Any]] = []
    for curves_csv in curves_files:
        parts = curves_csv.relative_to(bin_curves_root).parts
        if len(parts) < 3:
            continue
        direction_group = parts[0]
        bin_name = parts[1]
        baseline_coordinate, baseline_mode, grouped = load_clip_records(
            curves_csv,
            direction_group=direction_group,
            bin_name=bin_name,
            curve_smooth_window=args.curve_smooth_window,
        )

        for video_stem, records in sorted(grouped.items()):
            output_dir = output_root / direction_group / bin_name / video_stem
            title = f"Curve Boxplot - {direction_group} {bin_name} {video_stem}"
            result, kept_records, _, _ = process_clip_group(
                records,
                baseline_coordinate,
                baseline_mode,
                output_dir,
                title,
                min_curves=args.min_curves,
                min_samples=args.min_samples,
                min_curve_valid_fraction=args.min_curve_valid_fraction,
                min_sample_valid_fraction=args.min_sample_valid_fraction,
                box_smooth_window=args.box_smooth_window,
                branch_cleanup=not args.disable_branch_cleanup,
                branch_reference_quantile=args.branch_reference_quantile,
                branch_reference_smooth_window=args.branch_reference_smooth_window,
                branch_max_above_reference=args.branch_max_above_reference,
                use_pointwise_median=not args.use_depth_median_curve,
                tail_trim=not args.disable_tail_trim,
                tail_min_sample_valid_fraction=args.tail_min_sample_valid_fraction,
                jump_cleanup=not args.disable_jump_cleanup,
                jump_max_upward_px=args.jump_max_upward_px,
                post_min_rise_cleanup=not args.disable_post_min_rise_cleanup,
                post_min_max_rise_px=args.post_min_max_rise_px,
                central_band_percent=args.central_band_percent,
                variance_reference_x=args.variance_reference_x,
                require_reference_coverage=not args.disable_reference_coverage_qc,
                min_coordinate_span=args.min_coordinate_span,
                longest_component_gap_multiplier=args.longest_component_gap_multiplier,
                leading_spread_cleanup=not args.disable_leading_spread_cleanup,
                leading_spread_threshold=args.leading_spread_threshold,
                leading_stable_samples=args.leading_stable_samples,
                x_min=args.x_min,
                x_max=args.x_max,
                y_max=args.y_max,
                visual_band_scale=args.visual_band_scale,
                visual_min_band_width=args.visual_min_band_width,
            )
            label = f"{direction_group}/{bin_name}/{video_stem}"
            if result is None:
                print(f"[skip] {label}: not enough usable curves/samples")
                continue

            meta = metadata_for_records(kept_records, metadata_lookup)
            row = {
                "location": meta.get("location", bin_curves_root.name),
                "direction_group": direction_group,
                "final_plot_group": meta.get("final_plot_group", ""),
                "sea_state_bin": int(bin_name.replace("bin_", "")) if bin_name.replace("bin_", "").isdigit() else bin_name,
                "video_stem": video_stem,
                "group_by": "clip",
            }
            row.update(result)
            row.update(meta)
            summary_rows.append(row)
            print(f"[ok] {label}: {result['n_usable_curves']} curves, variance={result['curve_boxplot_variance']:.3f}")

    if not summary_rows:
        print("No per-clip curve boxplots were generated.")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "curve_boxplot_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    for row in summary_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-curves-root", default=None, help="Root containing per-bin curve_boxplot outputs.")
    parser.add_argument("--output-root", default=None, help="Output root for per-clip plots and summary.")
    parser.add_argument("--direction-group", default=None, help="Optionally process one direction group.")
    parser.add_argument("--sea-state-bin", type=int, default=None, help="Optionally process one bin number.")
    parser.add_argument("--manifest", default=None, help="Download manifest CSV for matching sea-state metadata.")
    parser.add_argument("--candidate-csv", default=None, help="Candidate table CSV fallback for sea-state metadata.")
    parser.add_argument("--curve-smooth-window", type=int, default=5, help="Extra Hann smoothing on cleaned input curves.")
    parser.add_argument("--box-smooth-window", type=int, default=5, help="Smoothing window for plotted median and central band.")
    parser.add_argument(
        "--central-band-percent",
        type=float,
        default=80.0,
        help="Pointwise central band percentage to shade on the review plot.",
    )
    parser.add_argument(
        "--disable-branch-cleanup",
        action="store_true",
        help="Disable post-cleaning that removes wrong high branches/horns from existing curves.",
    )
    parser.add_argument(
        "--branch-reference-quantile",
        type=float,
        default=20.0,
        help="Lower quantile used as the expected real shoreline branch.",
    )
    parser.add_argument(
        "--branch-reference-smooth-window",
        type=int,
        default=21,
        help="Smoothing window for the lower-branch reference curve.",
    )
    parser.add_argument(
        "--branch-max-above-reference",
        type=float,
        default=350.0,
        help="Drop curve values this many pixels above the lower-branch reference.",
    )
    parser.add_argument(
        "--use-depth-median-curve",
        action="store_true",
        help="Plot the original curve-boxplot depth median instead of the smoother pointwise median.",
    )
    parser.add_argument(
        "--disable-tail-trim",
        action="store_true",
        help="Disable removal of low-support rightmost baseline chunks.",
    )
    parser.add_argument(
        "--tail-min-sample-valid-fraction",
        type=float,
        default=0.25,
        help="Trim trailing baseline samples after support falls below this fraction.",
    )
    parser.add_argument(
        "--disable-jump-cleanup",
        action="store_true",
        help="Disable trimming after a large upward median branch jump.",
    )
    parser.add_argument(
        "--jump-max-upward-px",
        type=float,
        default=180.0,
        help="Trim after the median jumps upward by more than this many pixels between adjacent samples.",
    )
    parser.add_argument(
        "--disable-post-min-rise-cleanup",
        action="store_true",
        help="Disable trimming after a large rise following the near-shore median minimum.",
    )
    parser.add_argument(
        "--post-min-max-rise-px",
        type=float,
        default=180.0,
        help="Trim after the median rises this many pixels above its near-shore minimum.",
    )
    parser.add_argument("--min-curve-valid-fraction", type=float, default=0.08)
    parser.add_argument("--min-sample-valid-fraction", type=float, default=0.10)
    parser.add_argument("--min-curves", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--variance-reference-x", type=float, default=500.0)
    parser.add_argument(
        "--disable-reference-coverage-qc",
        action="store_true",
        help="Allow plots that do not cover the variance reference coordinate.",
    )
    parser.add_argument(
        "--min-coordinate-span",
        type=float,
        default=300.0,
        help="Skip clips whose cleaned curve covers less than this many baseline pixels.",
    )
    parser.add_argument(
        "--longest-component-gap-multiplier",
        type=float,
        default=0.0,
        help=(
            "If positive, keep only the longest continuous baseline-coordinate "
            "span, splitting where gaps exceed this multiplier times the normal "
            "sample spacing."
        ),
    )
    parser.add_argument(
        "--disable-leading-spread-cleanup",
        action="store_true",
        help="Disable trimming of noisy high-spread samples at the beginning of the baseline.",
    )
    parser.add_argument(
        "--leading-spread-threshold",
        type=float,
        default=350.0,
        help="Trim leading samples until the central band width is below this value for a stable run.",
    )
    parser.add_argument(
        "--leading-stable-samples",
        type=int,
        default=8,
        help="Number of consecutive low-spread samples required before keeping the curve-boxplot span.",
    )
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX)
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX)
    parser.add_argument(
        "--visual-band-scale",
        type=float,
        default=1.0,
        help="Display-only multiplier for the shaded band width; saved curves and variance are unchanged.",
    )
    parser.add_argument(
        "--visual-min-band-width",
        type=float,
        default=0.0,
        help="Display-only minimum shaded band width in pixels; saved curves and variance are unchanged.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
