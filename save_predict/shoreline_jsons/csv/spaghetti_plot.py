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
                   max_jump_dist: float | None = None):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap(cmap_name)
    n_colors = max(1, getattr(cmap, 'N', 20))
    for i, (x, y, _) in enumerate(curves):
        color = cmap(i % n_colors)
        segments = _split_by_jump(x, y, max_jump_dist=max_jump_dist)
        for xs, ys in segments:
            plt.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    plt.title(title or "Shoreline Spaghetti Plot")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.grid(True, alpha=0.25)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def process_folder(clip_dir: str, out_prefix: str, label_name: str = "shoreline",
                   linewidth: float = 1.0, alpha: float = 0.35, cmap_name: str = "tab20",
                   xlim: Tuple[float, float] | None = None, ylim: Tuple[float, float] | None = None,
                   require_vertex_index: bool = False,
                   max_jump_dist: float | None = None,
                   report_top_jumps: int = 0) -> bool:
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

    plot_spaghetti(curves, out_png, title=title, linewidth=linewidth, alpha=alpha, cmap_name=cmap_name,
                   xlim=xlim, ylim=ylim, max_jump_dist=max_jump_dist)
    print(f"Saved: {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="./jennette_north/calm/7/", help="Path to a single clip folder containing shoreline CSVs (e.g., .../calm/9)")
    ap.add_argument("--out_prefix", default=None, help="Output file prefix (without extension) for the PNG in single-folder mode")
    ap.add_argument("--root_dir", default="./jennette_north/", help="If provided, recursively find folders with CSVs under this root and render a spaghetti plot per folder.")
    ap.add_argument("--out_root", default="./vis_spaghetti/", help="Output root directory for batch mode. Defaults to ./vis_spaghetti/")
    ap.add_argument("--flat_out", action="store_true", help="Save all outputs directly under out_root with flattened file names (no subfolders).")
    ap.add_argument("--label_name", default="shoreline", help="Label name to filter within CSVs (falls back to all rows if not found)")
    ap.add_argument("--linewidth", type=float, default=1.0, help="Line width for individual curves")
    ap.add_argument("--alpha", type=float, default=0.35, help="Alpha transparency for curves")
    ap.add_argument("--cmap", default="tab20", help="Matplotlib colormap name to cycle through for curves")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Optional x-limits: xmin xmax")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Optional y-limits: ymin ymax")
    ap.add_argument("--require_vertex_index", action="store_true", help="Skip files lacking vertex_index to avoid spurious connectors.")
    ap.add_argument("--max_jump_dist", type=float, default=25, help="If set, split a curve when adjacent vertices are farther than this (pixels).")
    ap.add_argument("--report_top_jumps", type=int, default=10, help="Print the top N curves by largest adjacent-vertex jump distance.")
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
                                report_top_jumps=args.report_top_jumps)
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
                   report_top_jumps=args.report_top_jumps)


if __name__ == "__main__":
    main()
