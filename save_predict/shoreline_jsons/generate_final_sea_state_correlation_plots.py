"""Generate final sea-state versus shoreline-variance correlation plots.

Input is the per-bin curve boxplot output from generate_curve_boxplots_by_bin.py.
For each direction/bin curve boxplot, this script extracts one scalar shoreline
variance from the middle valid baseline region, pairs it with buoy sea_state,
and writes separate plots plus one combined comparison plot:

    onshore  = combined onshore direction groups
    offshore = offshore direction group
    combined = onshore and offshore on the same x/y scales

The middle-window scalar avoids using the far-right part of the baseline when
many shoreline curves do not cover that region.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOXPLOT_ROOT = SCRIPT_DIR / "curve_boxplot_outputs" / "jennettes_pier"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "final_correlation_outputs" / "jennettes_pier"


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


def infer_final_plot_group(row: pd.Series) -> str:
    existing = str(row.get("final_plot_group", "")).strip().lower()
    if existing in {"onshore", "offshore"}:
        return existing

    direction = str(row.get("direction_group", "")).lower()
    if "offshore" in direction:
        return "offshore"
    return "onshore"


def sea_state_value(row: pd.Series) -> float:
    for col in ["sea_state_mean", "sea_state", "sea_state_bin_mean"]:
        if col in row and pd.notna(row[col]):
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(value):
                return float(value)

    h = pd.to_numeric(row.get("H_mean", np.nan), errors="coerce")
    t = pd.to_numeric(row.get("T_mean", np.nan), errors="coerce")
    if pd.notna(h) and pd.notna(t):
        return float((h**2) * t)
    return float("nan")


def bin_folder_from_summary_row(row: pd.Series, boxplot_root: Path) -> Path:
    plot_path = str(row.get("plot_path", "")).strip()
    if plot_path and plot_path.lower() != "nan":
        path = Path(plot_path)
        if path.name:
            return path.parent

    direction_group = str(row["direction_group"])
    sea_state_bin = row["sea_state_bin"]
    if pd.notna(sea_state_bin):
        try:
            bin_name = f"bin_{int(float(sea_state_bin))}"
        except ValueError:
            bin_name = str(sea_state_bin)
    else:
        bin_name = ""

    return boxplot_root / direction_group / bin_name


def curve_columns(curves_df: pd.DataFrame) -> list[tuple[str, float]]:
    cols: list[tuple[str, float]] = []
    for col in curves_df.columns:
        if not col.startswith(("y_", "s_")):
            continue
        try:
            cols.append((col, float(col[2:])))
        except ValueError:
            continue
    return sorted(cols, key=lambda item: item[1])


def middle_window_variance(
    curves_csv: Path,
    middle_fraction: float,
    min_sample_valid_fraction: float,
    window_position: str = "middle",
) -> dict[str, Any]:
    curves_df = pd.read_csv(curves_csv)
    y_cols = curve_columns(curves_df)
    if not y_cols:
        raise ValueError(f"No y_* or s_* curve columns found in {curves_csv}")

    col_names = [col for col, _ in y_cols]
    y_values = np.asarray([y for _, y in y_cols], dtype=float)
    curve_values = curves_df[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    sample_valid_fraction = np.isfinite(curve_values).mean(axis=0)
    eligible = sample_valid_fraction >= min_sample_valid_fraction
    if not eligible.any():
        eligible = np.isfinite(curve_values).any(axis=0)
    if not eligible.any():
        raise ValueError(f"No usable curve samples found in {curves_csv}")

    eligible_y = y_values[eligible]
    full_span = float(y_values.max() - y_values.min())
    window_width = max(1.0, middle_fraction * full_span)
    if window_position == "right":
        edge = float(eligible_y.max())
        in_window = eligible & (y_values >= edge - window_width)
        fallback_target = edge
    elif window_position == "left":
        edge = float(eligible_y.min())
        in_window = eligible & (y_values <= edge + window_width)
        fallback_target = edge
    else:
        fallback_target = float(np.median(eligible_y))
        half_width = 0.5 * window_width
        in_window = eligible & (np.abs(y_values - fallback_target) <= half_width)
    if not in_window.any():
        nearest_idx = int(np.argmin(np.abs(y_values - fallback_target)))
        in_window[nearest_idx] = True
    center_y = float(np.median(y_values[in_window]))

    selected_values = curve_values[:, in_window]
    pointwise_variance = np.nanvar(selected_values, axis=0, ddof=0)
    finite_var = pointwise_variance[np.isfinite(pointwise_variance)]
    scalar = float(np.nanmean(finite_var)) if finite_var.size else float("nan")

    return {
        "shoreline_variance": scalar,
        "variance_window_position": window_position,
        "variance_center_y": center_y,
        "variance_window_y_min": float(y_values[in_window].min()),
        "variance_window_y_max": float(y_values[in_window].max()),
        "variance_window_samples": int(in_window.sum()),
        "variance_curve_count": int(curve_values.shape[0]),
    }


def target_coordinate_variance(
    curves_csv: Path,
    target_coordinate: float,
    min_sample_valid_fraction: float,
) -> dict[str, Any]:
    curves_df = pd.read_csv(curves_csv)
    y_cols = curve_columns(curves_df)
    if not y_cols:
        raise ValueError(f"No y_* or s_* curve columns found in {curves_csv}")

    col_names = [col for col, _ in y_cols]
    y_values = np.asarray([y for _, y in y_cols], dtype=float)
    curve_values = curves_df[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    sample_valid_fraction = np.isfinite(curve_values).mean(axis=0)
    eligible = sample_valid_fraction >= min_sample_valid_fraction
    if not eligible.any():
        eligible = np.isfinite(curve_values).any(axis=0)
    if not eligible.any():
        raise ValueError(f"No usable curve samples found in {curves_csv}")

    eligible_indices = np.where(eligible)[0]
    nearest_idx = eligible_indices[int(np.argmin(np.abs(y_values[eligible] - target_coordinate)))]
    selected_values = curve_values[:, nearest_idx]
    finite_values = selected_values[np.isfinite(selected_values)]
    scalar = float(np.nanvar(finite_values, ddof=0)) if finite_values.size else float("nan")
    center_y = float(y_values[nearest_idx])

    return {
        "shoreline_variance": scalar,
        "variance_window_position": "coordinate",
        "variance_target_coordinate": float(target_coordinate),
        "variance_center_y": center_y,
        "variance_window_y_min": center_y,
        "variance_window_y_max": center_y,
        "variance_window_samples": 1,
        "variance_curve_count": int(curve_values.shape[0]),
        "variance_finite_curve_count": int(finite_values.size),
    }


def add_scalar_rows(
    summary: pd.DataFrame,
    boxplot_root: Path,
    middle_fraction: float,
    min_sample_valid_fraction: float,
    window_position: str,
    target_coordinate: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in summary.iterrows():
        out = row.to_dict()
        out["final_plot_group"] = infer_final_plot_group(row)
        out["sea_state_for_plot"] = sea_state_value(row)

        bin_folder = bin_folder_from_summary_row(row, boxplot_root)
        curves_csv = bin_folder / "curves.csv"
        out["curves_csv"] = str(curves_csv)

        if curves_csv.exists():
            try:
                if target_coordinate is not None:
                    out.update(
                        target_coordinate_variance(
                            curves_csv,
                            target_coordinate=target_coordinate,
                            min_sample_valid_fraction=min_sample_valid_fraction,
                        )
                    )
                else:
                    out.update(
                        middle_window_variance(
                            curves_csv,
                            middle_fraction=middle_fraction,
                            min_sample_valid_fraction=min_sample_valid_fraction,
                            window_position=window_position,
                        )
                    )
            except Exception as exc:
                out["shoreline_variance"] = np.nan
                out["variance_error"] = str(exc)
        else:
            out["shoreline_variance"] = np.nan
            out["variance_error"] = f"Missing {curves_csv}"

        rows.append(out)

    return pd.DataFrame(rows)


def mark_variance_outliers(
    points: pd.DataFrame,
    mad_scale: float,
    enabled: bool,
) -> pd.DataFrame:
    points = points.copy()
    points["included_in_correlation"] = True
    points["cleanup_reason"] = ""
    if not enabled:
        return points

    for group, idx in points.groupby("final_plot_group").groups.items():
        values = pd.to_numeric(points.loc[idx, "shoreline_variance"], errors="coerce")
        finite = values.dropna()
        if len(finite) < 6:
            continue

        median = float(finite.median())
        mad = float((finite - median).abs().median())
        if mad > 0:
            robust_sigma = 1.4826 * mad
            high_limit = median + mad_scale * robust_sigma
            low_limit = max(0.0, median - mad_scale * robust_sigma)
        else:
            q1 = float(finite.quantile(0.25))
            q3 = float(finite.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            low_limit = max(0.0, q1 - 1.5 * iqr)
            high_limit = q3 + 1.5 * iqr

        is_outlier = (values < low_limit) | (values > high_limit)
        outlier_idx = values[is_outlier].index
        points.loc[outlier_idx, "included_in_correlation"] = False
        points.loc[outlier_idx, "cleanup_reason"] = (
            f"{group} shoreline variance outside robust range "
            f"[{low_limit:.3f}, {high_limit:.3f}]"
        )

    return points


def mark_target_coordinate_quality(
    points: pd.DataFrame,
    target_coordinate: float | None,
    coordinate_tolerance: float | None,
    min_finite_curves: int,
) -> pd.DataFrame:
    points = points.copy()
    if target_coordinate is None:
        return points

    if "included_in_correlation" not in points.columns:
        points["included_in_correlation"] = True
    if "cleanup_reason" not in points.columns:
        points["cleanup_reason"] = ""

    if coordinate_tolerance is not None:
        center = pd.to_numeric(points.get("variance_center_y"), errors="coerce")
        too_far = (center - target_coordinate).abs() > coordinate_tolerance
        points.loc[too_far, "included_in_correlation"] = False
        reason = (
            f"nearest valid coordinate more than {coordinate_tolerance:g} px "
            f"from target {target_coordinate:g}"
        )
        points.loc[too_far, "cleanup_reason"] = points.loc[too_far, "cleanup_reason"].where(
            points.loc[too_far, "cleanup_reason"].astype(str).str.len() > 0,
            reason,
        )

    finite_count = pd.to_numeric(points.get("variance_finite_curve_count"), errors="coerce")
    too_few = finite_count < min_finite_curves
    points.loc[too_few, "included_in_correlation"] = False
    reason = f"fewer than {min_finite_curves} finite curves at target coordinate"
    points.loc[too_few, "cleanup_reason"] = points.loc[too_few, "cleanup_reason"].where(
        points.loc[too_few, "cleanup_reason"].astype(str).str.len() > 0,
        reason,
    )

    return points


def apply_manual_qc_exclusions(points: pd.DataFrame, exclusions_csv: Path | None) -> pd.DataFrame:
    points = points.copy()
    if exclusions_csv is None:
        return points
    if not exclusions_csv.exists():
        raise FileNotFoundError(f"Manual exclusion CSV not found: {exclusions_csv}")

    exclusions = pd.read_csv(exclusions_csv).fillna("")
    if "video_stem" not in exclusions.columns:
        raise ValueError("Manual exclusion CSV must include a video_stem column")

    if "included_in_correlation" not in points.columns:
        points["included_in_correlation"] = True
    if "cleanup_reason" not in points.columns:
        points["cleanup_reason"] = ""

    for _, exclusion in exclusions.iterrows():
        video_stem = str(exclusion.get("video_stem", "")).strip()
        if not video_stem:
            continue

        mask = points["video_stem"].astype(str).str.strip() == video_stem
        direction_group = str(exclusion.get("direction_group", "")).strip()
        if direction_group:
            mask &= points["direction_group"].astype(str).str.strip() == direction_group

        sea_state_bin = str(exclusion.get("sea_state_bin", "")).strip()
        if sea_state_bin and "sea_state_bin" in points.columns:
            point_bins = pd.to_numeric(points["sea_state_bin"], errors="coerce")
            try:
                bin_value = float(sea_state_bin)
                mask &= point_bins == bin_value
            except ValueError:
                mask &= points["sea_state_bin"].astype(str).str.strip() == sea_state_bin

        reason = str(exclusion.get("reason", "")).strip() or "manual QC exclusion"
        points.loc[mask, "included_in_correlation"] = False
        points.loc[mask, "cleanup_reason"] = points.loc[mask, "cleanup_reason"].where(
            points.loc[mask, "cleanup_reason"].astype(str).str.len() > 0,
            reason,
        )

    return points


def correlation_stats(df: pd.DataFrame) -> dict[str, float]:
    clean = df[df["included_in_correlation"]][["sea_state_for_plot", "shoreline_variance"]].dropna()
    if len(clean) < 2:
        return {
            "n": float(len(clean)),
            "excluded_n": float((~df["included_in_correlation"]).sum()),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }

    x = clean["sea_state_for_plot"].to_numpy(dtype=float)
    y = clean["shoreline_variance"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(clean["sea_state_for_plot"].corr(clean["shoreline_variance"], method="spearman"))

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "n": float(len(clean)),
        "excluded_n": float((~df["included_in_correlation"]).sum()),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def plot_group(
    df: pd.DataFrame,
    group: str,
    output_path: Path,
    title_prefix: str,
    window_position: str,
    variance_label: str,
) -> dict[str, float]:
    plot_df = df[df["final_plot_group"] == group].copy()
    plot_df = plot_df.dropna(subset=["sea_state_for_plot", "shoreline_variance"])
    stats = correlation_stats(plot_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))

    if plot_df.empty:
        plt.text(0.5, 0.5, f"No {group} data available", ha="center", va="center")
        plt.axis("off")
    else:
        included = plot_df[plot_df["included_in_correlation"]]
        excluded = plot_df[~plot_df["included_in_correlation"]]

        for direction_group, sub in included.groupby("direction_group"):
            plt.scatter(
                sub["sea_state_for_plot"],
                sub["shoreline_variance"],
                s=70,
                alpha=0.85,
                label=str(direction_group),
            )

        if not excluded.empty:
            plt.scatter(
                excluded["sea_state_for_plot"],
                excluded["shoreline_variance"],
                s=95,
                marker="x",
                color="0.45",
                linewidths=2,
                label="excluded by cleanup",
            )

        if len(included) >= 2 and np.isfinite(stats["slope"]):
            x = included["sea_state_for_plot"].to_numpy(dtype=float)
            x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            y_line = stats["slope"] * x_line + stats["intercept"]
            plt.plot(x_line, y_line, color="black", linewidth=2, label="Linear trend")

        stats_text = (
            f"n={int(stats['n'])}\n"
            f"excluded={int(stats['excluded_n'])}\n"
            f"Pearson r={stats['pearson_r']:.2f}\n"
            f"Spearman r={stats['spearman_r']:.2f}\n"
            f"R^2={stats['r_squared']:.2f}"
        )
        plt.gca().text(
            0.03,
            0.97,
            stats_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

        plt.xlabel("Sea state = H^2 x T")
        plt.ylabel(f"{variance_label} shoreline variance (pixels^2)")
        plt.title(f"{title_prefix} {group.capitalize()}: sea state vs shoreline variance")
        plt.grid(True, alpha=0.35)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return stats


def plot_combined_groups(
    df: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    window_position: str,
    variance_label: str,
) -> dict[str, dict[str, float]]:
    """Plot onshore and offshore points together with shared axis limits."""
    plot_df = df.dropna(subset=["sea_state_for_plot", "shoreline_variance"]).copy()
    group_colors = {"onshore": "#D55E00", "offshore": "#0072B2"}
    group_stats: dict[str, dict[str, float]] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    if plot_df.empty:
        ax.text(0.5, 0.5, "No onshore or offshore data available", ha="center", va="center")
        ax.axis("off")
    else:
        for group in ["onshore", "offshore"]:
            group_df = plot_df[plot_df["final_plot_group"] == group]
            stats = correlation_stats(group_df)
            group_stats[group] = stats
            color = group_colors[group]
            included = group_df[group_df["included_in_correlation"]]
            excluded = group_df[~group_df["included_in_correlation"]]

            if not included.empty:
                ax.scatter(
                    included["sea_state_for_plot"],
                    included["shoreline_variance"],
                    s=78,
                    alpha=0.85,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    label=f"{group.capitalize()} points",
                )

            if not excluded.empty:
                ax.scatter(
                    excluded["sea_state_for_plot"],
                    excluded["shoreline_variance"],
                    s=100,
                    marker="x",
                    color=color,
                    linewidths=2,
                    label=f"{group.capitalize()} excluded by cleanup",
                )

            if len(included) >= 2 and np.isfinite(stats["slope"]):
                x = included["sea_state_for_plot"].to_numpy(dtype=float)
                x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                ax.plot(
                    x_line,
                    stats["slope"] * x_line + stats["intercept"],
                    color=color,
                    linewidth=2.2,
                    label=f"{group.capitalize()} linear trend",
                )

        x_max = float(plot_df["sea_state_for_plot"].max())
        y_max = float(plot_df["shoreline_variance"].max())
        ax.set_xlim(0, max(1.0, x_max * 1.05))
        ax.set_ylim(0, max(1.0, y_max * 1.08))

        stats_lines = []
        for group in ["onshore", "offshore"]:
            stats = group_stats.get(group, {})
            if stats:
                stats_lines.append(
                    f"{group.capitalize()}: n={int(stats['n'])}, "
                    f"Pearson r={stats['pearson_r']:.2f}, "
                    f"Spearman r={stats['spearman_r']:.2f}"
                )
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

        ax.set_xlabel("Sea state = H^2 x T")
        ax.set_ylabel(f"{variance_label} shoreline variance (pixels^2)")
        ax.set_title(f"{title_prefix}: onshore and offshore sea-state correlation")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return group_stats


def generate_synthetic_onshore_points(
    points: pd.DataFrame,
    target_count: int,
    seed: int,
) -> pd.DataFrame:
    """Create clearly labeled illustrative points following an increasing trend."""
    observed = points[points["final_plot_group"] == "onshore"].dropna(
        subset=["sea_state_for_plot", "shoreline_variance"]
    )
    n_to_add = max(0, target_count - len(observed))
    if n_to_add == 0 or len(observed) < 2:
        return pd.DataFrame()

    x_observed = observed["sea_state_for_plot"].to_numpy(dtype=float)
    y_observed = observed["shoreline_variance"].to_numpy(dtype=float)
    x_min = float(np.min(x_observed))
    x_max = float(np.max(x_observed))
    if x_max <= x_min:
        return pd.DataFrame()

    # Percentile anchors reduce the influence of isolated segmentation errors.
    y_low, y_high = np.percentile(y_observed, [20, 80])
    slope = max(0.0, float((y_high - y_low) / (x_max - x_min)))
    x_synthetic = np.linspace(x_min, x_max, n_to_add + 2)[1:-1]
    rng = np.random.default_rng(seed)
    expected_observed = y_low + slope * (x_observed - x_min)
    observed_residual_sigma = float(np.std(y_observed - expected_observed))
    noise_sigma = max(
        1.0,
        0.18 * float(y_high - y_low),
        0.45 * observed_residual_sigma,
    )
    y_synthetic = y_low + slope * (x_synthetic - x_min)
    y_synthetic += rng.normal(0.0, noise_sigma, size=n_to_add)
    y_synthetic = np.maximum(0.0, y_synthetic)

    return pd.DataFrame(
        {
            "synthetic_id": [f"synthetic_onshore_{i:02d}" for i in range(1, n_to_add + 1)],
            "final_plot_group": "onshore",
            "direction_group": "synthetic_expected_onshore",
            "sea_state_for_plot": x_synthetic,
            "shoreline_variance": y_synthetic,
            "included_in_correlation": True,
            "data_source": "synthetic_expected_not_observed",
            "synthetic_seed": seed,
            "generation_method": "positive percentile-anchor trend plus observed-residual-scale noise",
        }
    )


def plot_expected_scenario(
    observed_points: pd.DataFrame,
    synthetic_onshore: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    window_position: str,
    variance_label: str,
) -> None:
    """Plot observed groups with a separately marked synthetic onshore scenario."""
    observed = observed_points.dropna(subset=["sea_state_for_plot", "shoreline_variance"]).copy()
    onshore_observed = observed[observed["final_plot_group"] == "onshore"]
    offshore_observed = observed[observed["final_plot_group"] == "offshore"]
    onshore_augmented = pd.concat([onshore_observed, synthetic_onshore], ignore_index=True, sort=False)

    onshore_stats = correlation_stats(onshore_observed)
    augmented_stats = correlation_stats(onshore_augmented)
    offshore_stats = correlation_stats(offshore_observed)

    orange = "#D55E00"
    blue = "#0072B2"
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        onshore_observed["sea_state_for_plot"],
        onshore_observed["shoreline_variance"],
        s=76,
        color=orange,
        alpha=0.85,
        label="Onshore observed",
    )
    ax.scatter(
        synthetic_onshore["sea_state_for_plot"],
        synthetic_onshore["shoreline_variance"],
        s=82,
        marker="o",
        facecolors="none",
        edgecolors=orange,
        linewidths=1.8,
        label="Onshore synthetic expected",
    )
    ax.scatter(
        offshore_observed["sea_state_for_plot"],
        offshore_observed["shoreline_variance"],
        s=76,
        color=blue,
        alpha=0.85,
        label="Offshore observed",
    )

    for frame, stats, color, label, linestyle in [
        (onshore_augmented, augmented_stats, orange, "Onshore illustrative trend", "--"),
        (offshore_observed, offshore_stats, blue, "Offshore observed trend", "-"),
    ]:
        included = frame[frame["included_in_correlation"]]
        if len(included) >= 2 and np.isfinite(stats["slope"]):
            x = included["sea_state_for_plot"].to_numpy(dtype=float)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(
                x_line,
                stats["slope"] * x_line + stats["intercept"],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )

    all_x = pd.concat([observed["sea_state_for_plot"], synthetic_onshore["sea_state_for_plot"]])
    all_y = pd.concat([observed["shoreline_variance"], synthetic_onshore["shoreline_variance"]])
    ax.set_xlim(0, max(1.0, float(all_x.max()) * 1.05))
    ax.set_ylim(0, max(1.0, float(all_y.max()) * 1.10))

    stats_text = (
        f"Onshore observed: n={int(onshore_stats['n'])}, Pearson r={onshore_stats['pearson_r']:.2f}\n"
        f"Onshore illustrative: n={int(augmented_stats['n'])}, Pearson r={augmented_stats['pearson_r']:.2f}\n"
        f"Offshore observed: n={int(offshore_stats['n'])}, Pearson r={offshore_stats['pearson_r']:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    ax.set_xlabel("Sea state = H^2 x T")
    ax.set_ylabel(f"{variance_label} shoreline variance (pixels^2)")
    ax.set_title(f"{title_prefix}: illustrative expected onshore correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_stats(stats_rows: list[dict[str, Any]], stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)


def run(args: argparse.Namespace) -> int:
    boxplot_root = resolve_path(args.boxplot_root, DEFAULT_BOXPLOT_ROOT)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    summary_path = resolve_path(args.summary_csv, boxplot_root / "curve_boxplot_summary.csv")

    if not summary_path.exists():
        print(f"Missing summary CSV: {summary_path}")
        return 1

    summary = pd.read_csv(summary_path)
    required = {"direction_group", "sea_state_bin"}
    missing = required - set(summary.columns)
    if missing:
        print(f"Summary CSV is missing columns: {sorted(missing)}")
        return 1

    points = add_scalar_rows(
        summary,
        boxplot_root=boxplot_root,
        middle_fraction=args.middle_fraction,
        min_sample_valid_fraction=args.min_sample_valid_fraction,
        window_position=args.window_position,
        target_coordinate=args.target_coordinate,
    )
    points = points.dropna(subset=["sea_state_for_plot", "shoreline_variance"])
    points = mark_variance_outliers(
        points,
        mad_scale=args.outlier_mad_scale,
        enabled=not args.disable_outlier_cleanup,
    )
    points = mark_target_coordinate_quality(
        points,
        target_coordinate=args.target_coordinate,
        coordinate_tolerance=args.target_coordinate_tolerance,
        min_finite_curves=args.min_target_finite_curves,
    )
    manual_exclusions_path = (
        resolve_path(args.manual_exclusions_csv, output_root / "manual_qc_exclusions.csv")
        if args.manual_exclusions_csv
        else None
    )
    points = apply_manual_qc_exclusions(points, manual_exclusions_path)

    output_root.mkdir(parents=True, exist_ok=True)
    points_path = output_root / "final_correlation_points.csv"
    points.to_csv(points_path, index=False)

    title_prefix = args.location_label or boxplot_root.name.replace("_", " ").title()
    variance_position_label = (
        f"x={args.target_coordinate:g}" if args.target_coordinate is not None else f"{args.window_position.capitalize()}-window"
    )
    stats_rows: list[dict[str, Any]] = []
    for group in ["onshore", "offshore"]:
        plot_path = output_root / f"{safe_name(group)}_sea_state_vs_shoreline_variance.png"
        stats = plot_group(
            points,
            group,
            plot_path,
            title_prefix=title_prefix,
            window_position=args.window_position,
            variance_label=variance_position_label,
        )
        stats_rows.append(
            {
                "final_plot_group": group,
                "variance_window_position": "coordinate" if args.target_coordinate is not None else args.window_position,
                "variance_target_coordinate": args.target_coordinate if args.target_coordinate is not None else "",
                "plot_path": str(plot_path),
                **stats,
            }
        )
        print(
            f"[ok] {group}: n={int(stats['n'])}, "
            f"pearson_r={stats['pearson_r']:.3f}, spearman_r={stats['spearman_r']:.3f}"
        )

    combined_plot_path = output_root / "combined_onshore_offshore_sea_state_vs_shoreline_variance.png"
    plot_combined_groups(
        points,
        combined_plot_path,
        title_prefix=title_prefix,
        window_position=args.window_position,
        variance_label=variance_position_label,
    )
    print(f"[ok] combined onshore/offshore plot: {combined_plot_path}")

    if args.synthetic_onshore_target > 0:
        synthetic = generate_synthetic_onshore_points(
            points,
            target_count=args.synthetic_onshore_target,
            seed=args.synthetic_seed,
        )
        if synthetic.empty:
            print("[skip] no synthetic points needed or insufficient observed onshore data")
        else:
            synthetic_path = output_root / "synthetic_expected_onshore_points.csv"
            synthetic.to_csv(synthetic_path, index=False)
            expected_plot_path = output_root / "illustrative_expected_onshore_correlation.png"
            plot_expected_scenario(
                points,
                synthetic,
                expected_plot_path,
                title_prefix=title_prefix,
                window_position=args.window_position,
                variance_label=variance_position_label,
            )
            print(
                f"[ok] illustrative expected plot: {expected_plot_path} "
                f"({len(synthetic)} synthetic points; {args.synthetic_onshore_target} total onshore)"
            )
            print(f"[ok] synthetic points: {synthetic_path}")

    stats_path = output_root / "final_correlation_stats.csv"
    write_stats(stats_rows, stats_path)

    print(f"Wrote points: {points_path}")
    print(f"Wrote stats: {stats_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxplot-root", default=None, help="Root containing per-bin curve boxplot outputs.")
    parser.add_argument("--summary-csv", default=None, help="Curve boxplot summary CSV.")
    parser.add_argument("--output-root", default=None, help="Output folder for final points, stats, and plots.")
    parser.add_argument(
        "--middle-fraction",
        type=float,
        default=0.25,
        help="Fraction of the baseline span included in the selected variance window.",
    )
    parser.add_argument(
        "--window-position",
        choices=["left", "middle", "right"],
        default="middle",
        help="Position of the baseline window used to calculate shoreline variance.",
    )
    parser.add_argument(
        "--target-coordinate",
        type=float,
        default=None,
        help="Use the nearest valid baseline coordinate to this value instead of a window.",
    )
    parser.add_argument(
        "--target-coordinate-tolerance",
        type=float,
        default=None,
        help="Exclude target-coordinate points whose nearest valid coordinate is farther than this many pixels.",
    )
    parser.add_argument(
        "--min-target-finite-curves",
        type=int,
        default=1,
        help="Exclude target-coordinate points with fewer finite curves than this at the selected coordinate.",
    )
    parser.add_argument(
        "--min-sample-valid-fraction",
        type=float,
        default=0.10,
        help="Minimum fraction of curves with data at a baseline sample before it can define the middle window.",
    )
    parser.add_argument("--location-label", default=None, help="Optional location label for plot titles.")
    parser.add_argument(
        "--outlier-mad-scale",
        type=float,
        default=4.0,
        help="Robust MAD scale for excluding extreme shoreline-variance outliers from trend/correlation.",
    )
    parser.add_argument(
        "--disable-outlier-cleanup",
        action="store_true",
        help="Keep every point in trend/correlation calculations.",
    )
    parser.add_argument(
        "--manual-exclusions-csv",
        default=None,
        help=(
            "Optional CSV of reviewed points to exclude from trend/correlation while keeping "
            "them visible as X markers. Required column: video_stem. Optional columns: "
            "direction_group, sea_state_bin, reason."
        ),
    )
    parser.add_argument(
        "--synthetic-onshore-target",
        type=int,
        default=0,
        help=(
            "Create a separately labeled illustrative plot by adding synthetic expected onshore "
            "points until this total is reached; zero disables synthetic generation."
        ),
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Random seed used only for illustrative synthetic onshore points.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
