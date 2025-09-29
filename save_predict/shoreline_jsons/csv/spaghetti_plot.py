"""
spaghetti_plot.py

Generate a spaghetti plot for shoreline polylines in a folder: one line per CSV curve.

Inputs (CSV per curve) expected columns (at minimum):
  - x, y, label
Optional:
  - vertex_index (used to preserve original polyline order if present)

Usage examples (PowerShell):
  # Single folder
  python .\save_predict\shoreline_jsons\csv\spaghetti_plot.py --clip_dir ".\jennette_north\calm\7" --out_prefix ".\vis_spaghetti\calm_7\spaghetti"

  # Batch: recursively find folders with CSVs under a root and render per-folder
  python .\save_predict\shoreline_jsons\csv\spaghetti_plot.py --root_dir ".\jennette_north" --out_root ".\vis_spaghetti"
"""
import argparse
import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def load_curves(clip_dir: str, label_name: str = "shoreline", require_vertex_index: bool = False,
                split_on_feature: bool = True) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Load curves from CSVs in clip_dir. Returns list of (x, y, stem).
    - If vertex_index exists, preserve that order (do NOT sort by x).
    - If multiple features/segments exist in one CSV (feature_id/group_id), split into separate curves.
    - If vertex_index is missing and require_vertex_index=True, skip the file to avoid spurious connectors.
      Otherwise, preserve original row order as a best-effort fallback (no x-sort to avoid artificial lines).
    - Skips curves with < 10 valid points.
    """
    curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
    csv_paths = sorted(glob.glob(os.path.join(clip_dir, "*.csv")))
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Filter by label when present
        if "label" in df.columns:
            if label_name in df["label"].values:
                df = df[df["label"] == label_name]
            # else keep all rows (fallback)

        # Must have coordinates
        if not set(["x", "y"]).issubset(df.columns):
            continue

        # Split by features if requested
        if split_on_feature and ("feature_id" in df.columns or "group_id" in df.columns):
            group_cols = [c for c in ["feature_id", "group_id"] if c in df.columns]
            grouped = df.groupby(group_cols, dropna=False)
            parts = [g for _, g in grouped]
        else:
            parts = [df]

        file_stem = os.path.splitext(os.path.basename(p))[0]
        for idx, part in enumerate(parts):
            sdf = part.dropna(subset=["x", "y"]).copy()
            if sdf.empty:
                continue
            # Preserve logical path order
            if "vertex_index" in sdf.columns:
                sdf = sdf.sort_values("vertex_index")
            elif require_vertex_index:
                # Skip to avoid artificial connectors
                continue
            else:
                # Fallback: keep original row order; DO NOT sort by x
                sdf = sdf.reset_index(drop=True)

            x = sdf["x"].to_numpy()
            y = sdf["y"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 10:
                continue
            stem = f"{file_stem}_part{idx+1}" if len(parts) > 1 else file_stem
            curves.append((x, y, stem))
    return curves


def find_csv_leaf_folders(root_dir: str) -> List[str]:
    leafs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.lower().endswith(".csv") for f in filenames):
            leafs.append(dirpath)
    leafs.sort()
    return leafs


def _slugify_path(rel_path: str) -> str:
    # Flatten path: replace separators with '__' and strip invalid chars
    s = rel_path.replace("\\", "__").replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "folder"


# ==== Functional/contour boxplot helpers ====
def _choose_common_grid_from_curves(curves: List[Tuple[np.ndarray, np.ndarray, str]], coverage: float = 0.8, n_grid: int = 500) -> np.ndarray:
    if not curves:
        raise ValueError("No curves to build grid from.")
    mins = np.array([c[0].min() for c in curves if c[0].size > 0])
    maxs = np.array([c[0].max() for c in curves if c[0].size > 0])
    pct_min = 100 * coverage
    pct_max = 100 * (1 - coverage)
    x_min = np.percentile(mins, pct_min)
    x_max = np.percentile(maxs, pct_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min = np.max(mins)
        x_max = np.min(maxs)
        if x_max <= x_min:
            x_min = float(np.min(mins))
            x_max = float(np.max(maxs))
    return np.linspace(x_min, x_max, n_grid)


def _interp_curves_to_grid(curves: List[Tuple[np.ndarray, np.ndarray, str]], x_grid: np.ndarray) -> np.ndarray:
    n_grid = len(x_grid)
    stack = []
    for (x, y, _) in curves:
        if x.size < 2:
            continue
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        yi = np.full(n_grid, np.nan, dtype=float)
        in_range = (x_grid >= xs.min()) & (x_grid <= xs.max())
        if np.any(in_range):
            yi[in_range] = np.interp(x_grid[in_range], xs, ys)
        stack.append(yi)
    if not stack:
        raise ValueError("No curves could be interpolated onto grid.")
    return np.vstack(stack)


def _compute_band_stats(Y: np.ndarray) -> dict:
    median = np.nanmedian(Y, axis=0)
    mean = np.nanmean(Y, axis=0)
    q1 = np.nanpercentile(Y, 25, axis=0)
    q3 = np.nanpercentile(Y, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    ymin = np.nanmin(Y, axis=0)
    ymax = np.nanmax(Y, axis=0)
    whisker_low = np.maximum(lower, ymin)
    whisker_high = np.minimum(upper, ymax)
    count = np.sum(np.isfinite(Y), axis=0)
    return dict(xcount=count, mean=mean, median=median, q1=q1, q3=q3, whisker_low=whisker_low, whisker_high=whisker_high, ymin=ymin, ymax=ymax)


def _detect_outliers(Y: np.ndarray, stats: dict, frac_threshold: float = 0.02, min_support_frac: float = 0.6) -> np.ndarray:
    """Support-aware outlier detection.

    Flags a curve as outlier if the fraction of well-supported x positions where it lies
    outside the whisker band exceeds frac_threshold. "Well-supported" means at least
    min_support_frac of curves have valid data at that position.
    """
    wl = stats["whisker_low"]
    wh = stats["whisker_high"]
    xcount = stats.get("xcount")
    n_curves, n_cols = Y.shape
    min_support = max(1, int(np.ceil(min_support_frac * n_curves)))
    support_mask = np.ones(n_cols, dtype=bool)
    if xcount is not None:
        support_mask = np.asarray(xcount) >= min_support

    outside = (Y < wl) | (Y > wh)
    outside &= np.isfinite(Y)
    if outside.ndim == 2 and support_mask.ndim == 1:
        outside[:, ~support_mask] = False

    denom = np.sum((np.isfinite(Y) & support_mask), axis=1)
    denom = np.maximum(denom, 1)
    frac = np.sum(outside, axis=1) / denom
    return frac > float(frac_threshold)


def _cap_outliers(Y: np.ndarray, stats: dict, mask: np.ndarray, cap_ratio: float = 0.10, min_support_frac: float = 0.6) -> np.ndarray:
    """Limit the number of outliers to the top-k by outside distance, where k is cap_ratio of curves."""
    if mask is None or not np.any(mask):
        return mask
    n = Y.shape[0]
    k = max(1, int(np.ceil(cap_ratio * n)))
    if np.count_nonzero(mask) <= k:
        return mask
    wl = stats["whisker_low"]
    wh = stats["whisker_high"]
    xcount = stats.get("xcount")
    min_support = max(1, int(np.ceil(min_support_frac * n)))
    support_mask = np.ones(Y.shape[1], dtype=bool)
    if xcount is not None:
        support_mask = np.asarray(xcount) >= min_support
    # outside distance (0 inside whiskers), only on supported positions
    below = np.maximum(wl - Y, 0.0)
    above = np.maximum(Y - wh, 0.0)
    delta = (below + above)
    delta[:, ~support_mask] = 0.0
    scores = np.nansum(delta, axis=1)
    idx = np.where(mask)[0]
    # pick top-k indices among current outliers
    top_idx = idx[np.argsort(scores[idx])[::-1][:k]]
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[top_idx] = True
    return new_mask


def _split_by_jump(x: np.ndarray, y: np.ndarray, max_jump_dist: float | None) -> List[Tuple[np.ndarray, np.ndarray]]:
    if max_jump_dist is None or max_jump_dist <= 0:
        return [(x, y)]
    dx = np.diff(x)
    dy = np.diff(y)
    d = np.hypot(dx, dy)
    # indices where jump is too large; split after these indices
    split_idx = np.where(d > max_jump_dist)[0]
    if split_idx.size == 0:
        return [(x, y)]
    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in split_idx:
        end = i + 1
        if end - start >= 2:
            segs.append((x[start:end], y[start:end]))
        start = end
    if len(x) - start >= 2:
        segs.append((x[start:], y[start:]))
    return segs


def plot_spaghetti(curves: List[Tuple[np.ndarray, np.ndarray, str]], out_png: str, title: str = "",
                   linewidth: float = 1.0, alpha: float = 0.35, cmap_name: str = "tab20",
                   xlim: Tuple[float, float] | None = None, ylim: Tuple[float, float] | None = None,
                   max_jump_dist: float | None = None,
                   overlay_stats: dict | None = None,
                   x_grid: np.ndarray | None = None,
                   outlier_mask: np.ndarray | None = None,
                   show_legend: bool = True,
                   draw_spaghetti_lines: bool = False):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap(cmap_name)
    n_colors = max(1, getattr(cmap, 'N', 20))
    for i, (x, y, _) in enumerate(curves):
        is_out = bool(outlier_mask[i]) if (outlier_mask is not None and i < len(outlier_mask)) else False
        # Only draw outliers unless draw_spaghetti_lines=True
        if not (is_out or draw_spaghetti_lines):
            continue
        segments = _split_by_jump(x, y, max_jump_dist=max_jump_dist)
        for xs, ys in segments:
            if is_out:
                plt.plot(xs, ys, color='r', linestyle='--', linewidth=max(1.2, linewidth), alpha=0.9)
            else:
                plt.plot(xs, ys, color=cmap(i % n_colors), linewidth=linewidth, alpha=alpha)

    handles = []
    labels = []
    if overlay_stats is not None and x_grid is not None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        # 100% non-outlier band (whisker range) - light blue
        plt.fill_between(x_grid, overlay_stats["whisker_low"], overlay_stats["whisker_high"],
                         color=(0.5, 0.6, 0.95, 0.20), zorder=1)
        handles.append(Patch(facecolor=(0.5, 0.6, 0.95, 0.20), edgecolor='none'))
        labels.append("100% (non-outliers)")
        # Central 50% band - darker blue
        plt.fill_between(x_grid, overlay_stats["q1"], overlay_stats["q3"],
                         color=(0.3, 0.4, 0.9, 0.35), zorder=2)
        handles.append(Patch(facecolor=(0.3, 0.4, 0.9, 0.35), edgecolor='none'))
        labels.append("50% of data")
        # Mean and Median lines only
        mean_line, = plt.plot(x_grid, overlay_stats["mean"], color=(0.2, 0.8, 0.2, 1.0), linewidth=1.8)
        med_line, = plt.plot(x_grid, overlay_stats["median"], color=(0.85, 0.3, 0.05, 1.0), linewidth=2.0)
        handles.extend([mean_line, med_line])
        labels.extend(["Mean", "Median"])
        if outlier_mask is not None and np.any(outlier_mask):
            handles.append(Line2D([0], [0], color='r', lw=1.2, ls='--'))
            labels.append("Outlier")
    plt.title(title or "Shoreline Spaghetti Plot")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.grid(True, alpha=0.25)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if show_legend and overlay_stats is not None and labels:
        plt.legend(handles, labels, loc="upper right", framealpha=0.9)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_legend_panel(out_png: str):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(2.4, 3.0))
    ax.axis('off')
    handles = [
        Patch(facecolor=(0.5, 0.6, 0.95, 0.20), edgecolor='none', label='100% (non-outliers)'),
        Patch(facecolor=(0.3, 0.4, 0.9, 0.35), edgecolor='none', label='50% of data'),
        Line2D([0], [0], color=(0.2, 0.8, 0.2, 1.0), lw=2.0, label='Mean'),
        Line2D([0], [0], color=(0.85, 0.3, 0.05, 1.0), lw=2.5, label='Median'),
        Line2D([0], [0], color='r', lw=1.2, ls='--', label='Outlier')
    ]
    ax.legend(handles=handles, loc='center', framealpha=0.95)
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def process_folder(clip_dir: str, out_prefix: str, label_name: str = "shoreline",
                   linewidth: float = 1.0, alpha: float = 0.35, cmap_name: str = "tab20",
                   xlim: Tuple[float, float] | None = None, ylim: Tuple[float, float] | None = None,
                   require_vertex_index: bool = False,
                   max_jump_dist: float | None = None,
                   report_top_jumps: int = 0,
                   overlay_boxplot: bool = True,
                   coverage: float = 0.8,
                   n_grid: int = 500,
                   outlier_factor: float = 2.0,
                   outlier_frac: float = 0.03,
                   min_support_frac: float = 0.6,
                   cap_outliers_ratio: float = 0.10,
                   save_legend_panel: bool = True) -> bool:
    curves = load_curves(clip_dir, label_name=label_name, require_vertex_index=require_vertex_index, split_on_feature=True)
    if not curves:
        print(f"Skipping (no curves): {clip_dir}")
        return False
    title = f"Spaghetti: {os.path.basename(os.path.normpath(clip_dir))} (n={len(curves)})"
    out_png = f"{out_prefix}_spaghetti.png"
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # Log diagnostic extents
    xs = np.concatenate([c[0] for c in curves])
    ys = np.concatenate([c[1] for c in curves])
    print(f"Folder: {clip_dir}\n  curves: {len(curves)}\n  x_range: [{float(np.nanmin(xs)):.3f}, {float(np.nanmax(xs)):.3f}]\n  y_range: [{float(np.nanmin(ys)):.3f}, {float(np.nanmax(ys)):.3f}]")
    # Optional diagnostics: report curves with largest single-vertex jump
    if report_top_jumps and report_top_jumps > 0:
        jump_info = []
        for (x, y, stem) in curves:
            if x.size < 2:
                continue
            d = np.hypot(np.diff(x), np.diff(y))
            jump_info.append((float(np.nanmax(d)) if d.size else 0.0, stem))
        jump_info.sort(reverse=True)
        top = jump_info[:report_top_jumps]
        if top:
            print("Top jumps (pixels):")
            for val, stem in top:
                print(f"  {val:.1f}  {stem}")

    # Compute overlay stats if requested
    overlay_stats = None
    x_grid = None
    outlier_mask = None
    if overlay_boxplot:
        try:
            x_grid = _choose_common_grid_from_curves(curves, coverage=coverage, n_grid=n_grid)
            Y = _interp_curves_to_grid(curves, x_grid)
            overlay_stats = _compute_band_stats(Y)
            # Adjust whiskers by outlier_factor
            q1 = overlay_stats["q1"]; q3 = overlay_stats["q3"]; iqr = q3 - q1
            wl = np.maximum(q1 - outlier_factor * iqr, overlay_stats["ymin"])
            wh = np.minimum(q3 + outlier_factor * iqr, overlay_stats["ymax"]) 
            overlay_stats["whisker_low"] = wl
            overlay_stats["whisker_high"] = wh
            outlier_mask = _detect_outliers(Y, overlay_stats, frac_threshold=outlier_frac, min_support_frac=min_support_frac)
            # Cap outliers to at most a fraction of curves
            outlier_mask = _cap_outliers(Y, overlay_stats, outlier_mask, cap_ratio=cap_outliers_ratio, min_support_frac=min_support_frac)
        except Exception as e:
            print(f"Overlay stats failed: {e}")

    plot_spaghetti(curves, out_png, title=title, linewidth=linewidth, alpha=alpha, cmap_name=cmap_name,
                   xlim=xlim, ylim=ylim, max_jump_dist=max_jump_dist,
                   overlay_stats=overlay_stats, x_grid=x_grid, outlier_mask=outlier_mask, show_legend=overlay_boxplot,
                   draw_spaghetti_lines=False)
    if overlay_boxplot and save_legend_panel:
        _save_legend_panel(out_png.replace("_spaghetti.png", "_legend.png"))
    print(f"Saved: {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="./jennette_north/calm/7/", help="Path to a single clip folder containing shoreline CSVs (e.g., .../calm/9)")
    ap.add_argument("--out_prefix", default=None, help="Output file prefix (without extension) for the PNG in single-folder mode")
    ap.add_argument("--root_dir", default="./jennette_north/", help="If provided, recursively find folders with CSVs under this root and render a spaghetti plot per folder.")
    ap.add_argument("--out_root", default="./vis_box/", help="Output root directory for batch mode. Defaults to ./vis_spaghetti/")
    ap.add_argument("--flat_out", action="store_true", default=True, help="Save all outputs directly under out_root with flattened file names (no subfolders).")
    ap.add_argument("--label_name", default="shoreline", help="Label name to filter within CSVs (falls back to all rows if not found)")
    ap.add_argument("--linewidth", type=float, default=1.0, help="Line width for individual curves")
    ap.add_argument("--alpha", type=float, default=0.35, help="Alpha transparency for curves")
    ap.add_argument("--cmap", default="tab20", help="Matplotlib colormap name to cycle through for curves")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Optional x-limits: xmin xmax")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Optional y-limits: ymin ymax")
    ap.add_argument("--require_vertex_index", action="store_true", help="Skip files lacking vertex_index to avoid spurious connectors.")
    ap.add_argument("--max_jump_dist", type=float, default=25, help="If set, split a curve when adjacent vertices are farther than this (pixels).")
    ap.add_argument("--report_top_jumps", type=int, default=10, help="Print the top N curves by largest adjacent-vertex jump distance.")
    # Overlay options
    ap.add_argument("--overlay_boxplot", action="store_true", default=True, help="Overlay functional/contour boxplot stats on top of spaghetti lines.")
    ap.add_argument("--coverage", type=float, default=0.8, help="Coverage fraction for grid building (0.5–0.95 typical).")
    ap.add_argument("--n_grid", type=int, default=500, help="Number of samples in the overlay grid.")
    ap.add_argument("--outlier_factor", type=float, default=2.0, help="Whisker factor (Tukey).")
    ap.add_argument("--outlier_frac", type=float, default=0.03, help="Fraction of x outside whiskers to flag a curve outlier.")
    ap.add_argument("--min_support_frac", type=float, default=0.6, help="Only evaluate outliers where >= this fraction of curves have support.")
    ap.add_argument("--cap_outliers_ratio", type=float, default=0.10, help="Cap outliers to at most this fraction of curves (0.0–1.0).")
    ap.add_argument("--no_legend_panel", action="store_true", help="Do not save a separate legend panel image.")
    args = ap.parse_args()

    # Batch mode
    if args.root_dir:
        folders = find_csv_leaf_folders(args.root_dir)
        if not folders:
            raise SystemExit(f"No CSV-containing folders found under: {args.root_dir}")
        os.makedirs(args.out_root or "./vis_spaghetti/", exist_ok=True)
        processed = 0
        for d in folders:
            rel = os.path.relpath(d, start=args.root_dir)
            if args.flat_out:
                slug = _slugify_path(rel)
                out_prefix = os.path.join(args.out_root, slug)
                os.makedirs(args.out_root, exist_ok=True)
            else:
                out_prefix = os.path.join(args.out_root, rel, "spaghetti")
                os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            ok = process_folder(d, out_prefix, label_name=args.label_name,
                                linewidth=args.linewidth, alpha=args.alpha, cmap_name=args.cmap,
                                xlim=tuple(args.xlim) if args.xlim else None,
                                ylim=tuple(args.ylim) if args.ylim else None,
                                require_vertex_index=args.require_vertex_index,
                                max_jump_dist=args.max_jump_dist,
                                report_top_jumps=args.report_top_jumps,
                                overlay_boxplot=args.overlay_boxplot,
                                coverage=args.coverage,
                                n_grid=args.n_grid,
                                outlier_factor=args.outlier_factor,
                                outlier_frac=args.outlier_frac,
                                min_support_frac=args.min_support_frac,
                                cap_outliers_ratio=args.cap_outliers_ratio,
                                save_legend_panel=(not args.no_legend_panel))
            if ok:
                processed += 1
        print(f"\n✅ Batch complete. Folders processed: {processed}/{len(folders)}.")
        return

    # Single-folder mode
    if not args.clip_dir:
        raise SystemExit("Provide either --clip_dir (single) or --root_dir (batch).")
    if args.out_prefix is None:
        # Default into vis_spaghetti under the clip folder name
        base = os.path.basename(os.path.normpath(args.clip_dir))
        args.out_prefix = os.path.join("./vis_spaghetti/", base, "spaghetti")
    process_folder(args.clip_dir, args.out_prefix, label_name=args.label_name,
                   linewidth=args.linewidth, alpha=args.alpha, cmap_name=args.cmap,
                   xlim=tuple(args.xlim) if args.xlim else None,
                   ylim=tuple(args.ylim) if args.ylim else None,
                   require_vertex_index=args.require_vertex_index,
                   max_jump_dist=args.max_jump_dist,
                   report_top_jumps=args.report_top_jumps,
                   overlay_boxplot=args.overlay_boxplot,
                   coverage=args.coverage,
                   n_grid=args.n_grid,
                   outlier_factor=args.outlier_factor,
                   outlier_frac=args.outlier_frac,
                   min_support_frac=args.min_support_frac,
                   cap_outliers_ratio=args.cap_outliers_ratio,
                   save_legend_panel=(not args.no_legend_panel))


if __name__ == "__main__":
    main()
