
"""
curve_boxplot.py

Generate a curve "boxplot" visualization for shoreline polylines within a single clip folder.

A "clip folder" is expected to contain multiple CSVs, each with columns:
feature_id,label,vertex_index,x,y,group_id
and rows representing vertices of a detected shoreline polyline (in pixel coordinates).

This script:
- Loads all shoreline CSVs in the folder
- Sorts vertices by x for each curve
- Builds a common x-grid over the region covered by most curves (configurable coverage threshold)
- Interpolates each curve onto that grid
- Computes per-x median, quartiles (Q1, Q3), IQR, and Tukey-style whiskers
- Saves a PNG plot and a CSV of the band statistics
- Emits a JSON summary with scalar variability metrics

Usage (CLI):
    python curve_boxplot.py --clip_dir "C:/.../jennette_north/calm/9" --out_prefix "/path/to/output/calm_9"

Outputs:
    {out_prefix}_curve_boxplot.png
    {out_prefix}_band_stats.csv
    {out_prefix}_summary.json

Author: ChatGPT
"""
import argparse
import os
import glob
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_curves(clip_dir: str, label_name: str = "shoreline") -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load shoreline curves from all CSV files in clip_dir.
    Returns a list of (x_sorted, y_sorted, filename_stem).
    Skips files with < 10 points or missing columns.
    """
    curves = []
    csv_paths = sorted(glob.glob(os.path.join(clip_dir, "*.csv")))
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Basic validation
        if not set(["label","x","y"]).issubset(df.columns):
            continue
        if label_name in df["label"].values:
            sdf = df[df["label"] == label_name][["x","y"]].dropna()
        else:
            # If label column isn't filled consistently, try all rows
            sdf = df[["x","y"]].dropna()
        if len(sdf) < 10:
            continue
        # sort by x
        sdf = sdf.sort_values("x")
        x = sdf["x"].to_numpy()
        y = sdf["y"].to_numpy()
        # guard against non-finite
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 10:
            continue
        curves.append((x, y, os.path.splitext(os.path.basename(p))[0]))
    return curves

def choose_common_grid(curves: List[Tuple[np.ndarray, np.ndarray, str]], coverage: float = 0.8, n_grid: int = 500) -> np.ndarray:
    """
    Choose a common x-grid where at least `coverage` fraction of curves have support.
    We do this by computing the distribution of per-curve min/max x and taking
    the percentile range [pct_min, pct_max] where pct_min = 100*coverage for mins
    and pct_max = 100*(1-coverage) for maxes, then flipping as needed.

    Example: coverage=0.8 -> x_min = percentile(min_x, 80), x_max = percentile(max_x, 20).
    """
    if not curves:
        raise ValueError("No curves to build grid from.")
    mins = np.array([c[0].min() for c in curves])
    maxs = np.array([c[0].max() for c in curves])
    pct_min = 100 * coverage
    pct_max = 100 * (1 - coverage)
    x_min = np.percentile(mins, pct_min)
    x_max = np.percentile(maxs, pct_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        # fallback to global min/max
        x_min = np.max(mins)  # max of mins (guarantees all curves cover this start)
        x_max = np.min(maxs)  # min of maxs
        if x_max <= x_min:
            # final fallback: use global range and accept nans later
            x_min = np.min(mins)
            x_max = np.max(maxs)
    grid = np.linspace(x_min, x_max, n_grid)
    return grid

def interpolate_curves_to_grid(curves: List[Tuple[np.ndarray, np.ndarray, str]], x_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate each curve onto x_grid using linear interpolation.
    Values outside a curve's [min_x, max_x] are set to NaN.
    Returns array of shape (n_curves, n_grid).
    """
    n_grid = len(x_grid)
    interp_stack = []
    for (x, y, _) in curves:
        if len(x) < 2:
            continue
        # Ensure strict monotonic x for np.interp
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        # mask for in-range grid points
        in_range = (x_grid >= x_sorted.min()) & (x_grid <= x_sorted.max())
        yi = np.full(n_grid, np.nan, dtype=float)
        if np.any(in_range):
            yi[in_range] = np.interp(x_grid[in_range], x_sorted, y_sorted)
        interp_stack.append(yi)
    if not interp_stack:
        raise ValueError("No curves could be interpolated onto grid.")
    return np.vstack(interp_stack)

def compute_band_stats(Y: np.ndarray) -> dict:
    """
    Given Y of shape (n_curves, n_grid) with NaNs for missing,
    compute per-x statistics: median, q1, q3, whiskers, iqr, count, std.
    Returns a dict of arrays.
    """
    median = np.nanmedian(Y, axis=0)
    q1 = np.nanpercentile(Y, 25, axis=0)
    q3 = np.nanpercentile(Y, 75, axis=0)
    iqr = q3 - q1
    # Tukey whiskers
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # clip whiskers to min/max of available data at each x
    ymin = np.nanmin(Y, axis=0)
    ymax = np.nanmax(Y, axis=0)
    whisker_low = np.maximum(lower, ymin)
    whisker_high = np.minimum(upper, ymax)
    # counts and std
    count = np.sum(np.isfinite(Y), axis=0)
    std = np.nanstd(Y, axis=0)

    return dict(
        median=median, q1=q1, q3=q3, iqr=iqr,
        whisker_low=whisker_low, whisker_high=whisker_high,
        ymin=ymin, ymax=ymax, count=count, std=std
    )

def plot_curve_boxplot(x_grid: np.ndarray, stats: dict, out_png: str, title: str = "", figsize=(10,6)):
    plt.figure(figsize=figsize)
    # IQR band
    plt.fill_between(x_grid, stats["q1"], stats["q3"], alpha=0.35, label="IQR (Q1–Q3)")
    # Whiskers
    plt.plot(x_grid, stats["whisker_low"], linewidth=1, linestyle="--", label="Whiskers")
    plt.plot(x_grid, stats["whisker_high"], linewidth=1, linestyle="--")
    # Median
    plt.plot(x_grid, stats["median"], linewidth=2, label="Median")
    plt.title(title or "Shoreline Curve Boxplot")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_band_csv(x_grid: np.ndarray, stats: dict, out_csv: str):
    df = pd.DataFrame({
        "x": x_grid,
        "median": stats["median"],
        "q1": stats["q1"],
        "q3": stats["q3"],
        "whisker_low": stats["whisker_low"],
        "whisker_high": stats["whisker_high"],
        "iqr": stats["iqr"],
        "std": stats["std"],
        "count": stats["count"]
    })
    df.to_csv(out_csv, index=False)

def summarize_scalar(stats: dict) -> dict:
    """
    Produce scalar variability metrics useful for clip-level summaries.
    """
    # Use only x-positions with at least, say, 50% of curves present for robustness
    count = stats["count"]
    max_count = np.nanmax(count) if np.size(count) else 0
    mask = count >= 0.5 * max_count if max_count > 0 else np.ones_like(count, dtype=bool)

    mean_iqr = float(np.nanmean(stats["iqr"][mask]))
    median_iqr = float(np.nanmedian(stats["iqr"][mask]))
    mean_std = float(np.nanmean(stats["std"][mask]))
    median_std = float(np.nanmedian(stats["std"][mask]))

    return {
        "mean_iqr_pixels": round(mean_iqr, 3),
        "median_iqr_pixels": round(median_iqr, 3),
        "mean_std_pixels": round(mean_std, 3),
        "median_std_pixels": round(median_std, 3),
        "coverage_points": int(np.sum(mask)),
        "total_points": int(len(mask))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="./jennette_north/active/13/", help="Path to a single clip folder containing shoreline CSVs (e.g., .../calm/9)")
    ap.add_argument("--out_prefix", default="./vis/", help="Output file prefix (without extension) for PNG/CSV/JSON")
    ap.add_argument("--coverage", type=float, default=0.8, help="Fraction of curves that must overlap the x-range to define the grid (0.5–0.95 typical)")
    ap.add_argument("--n_grid", type=int, default=500, help="Number of x-grid samples for resampling the curves")
    args = ap.parse_args()

    curves = load_curves(args.clip_dir, label_name="shoreline")
    if not curves:
        raise SystemExit(f"No valid shoreline curves found in: {args.clip_dir}")

    x_grid = choose_common_grid(curves, coverage=args.coverage, n_grid=args.n_grid)
    Y = interpolate_curves_to_grid(curves, x_grid)

    stats = compute_band_stats(Y)
    title = f"Curve Boxplot: {os.path.basename(os.path.normpath(args.clip_dir))}  (n_curves={Y.shape[0]})"
    out_png = f"{args.out_prefix}_curve_boxplot.png"
    out_csv = f"{args.out_prefix}_band_stats.csv"
    out_json = f"{args.out_prefix}_summary.json"

    plot_curve_boxplot(x_grid, stats, out_png, title=title)
    save_band_csv(x_grid, stats, out_csv)

    summary = summarize_scalar(stats)
    # add some context
    summary.update({
        "clip_dir": args.clip_dir,
        "n_curves": int(Y.shape[0]),
        "x_min": float(x_grid.min()),
        "x_max": float(x_grid.max())
    })
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print("  PNG :", out_png)
    print("  CSV :", out_csv)
    print("  JSON:", out_json)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
