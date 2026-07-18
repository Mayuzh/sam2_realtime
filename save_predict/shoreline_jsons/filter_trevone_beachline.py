#!/usr/bin/env python3
"""Keep the Trevone beach shoreline branch from rectified CSVs.

The Trevone camera predictions often contain several nearly parallel branches:
the true wet/dry beach shoreline near the upper beach, plus lower foreground
or right-wall returns. This filter keeps the camera-image row band that maps to
the sandy beach shoreline before downstream representative-shoreline steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def process_csv(src: Path, dst: Path, args: argparse.Namespace) -> bool:
    df = pd.read_csv(src)
    lower = {c.lower(): c for c in df.columns}
    if "x_warped" not in lower or "y_warped" not in lower:
        return False

    if "baseline" in src.stem.lower():
        out = df.copy()
    else:
        keep = pd.Series(True, index=df.index)
        if "x" in lower:
            x = pd.to_numeric(df[lower["x"]], errors="coerce")
            keep &= x.between(args.raw_x_min, args.raw_x_max)
        if "y" in lower:
            y = pd.to_numeric(df[lower["y"]], errors="coerce")
            keep &= y.between(args.raw_y_min, args.raw_y_max)

        xw = pd.to_numeric(df[lower["x_warped"]], errors="coerce")
        yw = pd.to_numeric(df[lower["y_warped"]], errors="coerce")
        keep &= xw.notna() & yw.notna()
        if args.x_min is not None:
            keep &= xw >= args.x_min
        if args.x_max is not None:
            keep &= xw <= args.x_max
        if args.y_min is not None:
            keep &= yw >= args.y_min
        if args.y_max is not None:
            keep &= yw <= args.y_max

        out = df.loc[keep].copy()
        if len(out) < args.min_points:
            return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Input rectified Trevone CSV root.")
    parser.add_argument("--out", required=True, help="Output filtered CSV root.")
    parser.add_argument("--raw-x-min", type=float, default=60.0, help="Minimum original image x to keep.")
    parser.add_argument("--raw-x-max", type=float, default=820.0, help="Maximum original image x to keep.")
    parser.add_argument(
        "--raw-y-min",
        type=float,
        default=-750.0,
        help="Minimum original image y to keep. Trevone CSV y values are negative image rows.",
    )
    parser.add_argument(
        "--raw-y-max",
        type=float,
        default=-675.0,
        help="Maximum original image y to keep. Trevone CSV y values are negative image rows.",
    )
    parser.add_argument("--x-min", type=float, default=None, help="Optional minimum X_warped to keep.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional maximum X_warped to keep.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional minimum Y_warped to keep.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional maximum Y_warped to keep.")
    parser.add_argument("--min-points", type=int, default=8, help="Minimum kept points for shoreline CSVs.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    written = 0
    skipped = 0
    for src in sorted(root.rglob("*.csv")):
        rel = src.relative_to(root)
        if process_csv(src, out_root / rel, args):
            written += 1
        else:
            skipped += 1
    print(f"Wrote {written} CSV(s) under {out_root}; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
