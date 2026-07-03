"""Generate final sea-state versus shoreline-variance correlation plots.

Input is the per-bin curve boxplot output from generate_curve_boxplots_by_bin.py.
For each direction/bin curve boxplot, this script extracts one scalar shoreline
variance from the middle valid baseline region, pairs it with buoy sea_state,
and writes two final plots:

    onshore  = combined onshore direction groups
    offshore = offshore direction group

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


def add_scalar_rows(
    summary: pd.DataFrame,
    boxplot_root: Path,
    middle_fraction: float,
    min_sample_valid_fraction: float,
    window_position: str,
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
        plt.ylabel(f"{window_position.capitalize()}-window shoreline variance (pixels^2)")
        plt.title(f"{title_prefix} {group.capitalize()}: sea state vs shoreline variance")
        plt.grid(True, alpha=0.35)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return stats


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
    )
    points = points.dropna(subset=["sea_state_for_plot", "shoreline_variance"])
    points = mark_variance_outliers(
        points,
        mad_scale=args.outlier_mad_scale,
        enabled=not args.disable_outlier_cleanup,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    points_path = output_root / "final_correlation_points.csv"
    points.to_csv(points_path, index=False)

    title_prefix = args.location_label or boxplot_root.name.replace("_", " ").title()
    stats_rows: list[dict[str, Any]] = []
    for group in ["onshore", "offshore"]:
        plot_path = output_root / f"{safe_name(group)}_sea_state_vs_shoreline_variance.png"
        stats = plot_group(
            points,
            group,
            plot_path,
            title_prefix=title_prefix,
            window_position=args.window_position,
        )
        stats_rows.append(
            {
                "final_plot_group": group,
                "variance_window_position": args.window_position,
                "plot_path": str(plot_path),
                **stats,
            }
        )
        print(
            f"[ok] {group}: n={int(stats['n'])}, "
            f"pearson_r={stats['pearson_r']:.3f}, spearman_r={stats['spearman_r']:.3f}"
        )

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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
