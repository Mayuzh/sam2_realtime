#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a corrected rectified shoreline CSV root by clipping to a map ROI and
optionally applying a smooth east/right-end Y lift.

This is intended for cases where the georectified shorelines include coastline
outside the site of interest, or where a small local map-space correction is
needed after the main control-point warp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def process_csv(src: Path, dst: Path, args: argparse.Namespace) -> bool:
    df = pd.read_csv(src)
    lower = {c.lower(): c for c in df.columns}
    x_col = lower.get("x_warped")
    y_col = lower.get("y_warped")
    if x_col is None or y_col is None:
        return False

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    keep = x.notna() & y.notna()
    if args.x_min is not None:
        keep &= x >= args.x_min
    if args.x_max is not None:
        keep &= x <= args.x_max
    if args.y_min is not None:
        keep &= y >= args.y_min
    if args.y_max is not None:
        keep &= y <= args.y_max

    out = df.loc[keep].copy()
    if len(out) < args.min_points and "baseline" not in src.stem.lower():
        return False

    if args.lift_end_m != 0.0 or args.shift_end_x_m != 0.0:
        x_out = pd.to_numeric(out[x_col], errors="coerce").to_numpy(float)
        if args.lift_start_x is None:
            if args.x_min is None or args.x_max is None:
                raise ValueError("--lift-start-x is required unless --x-min and --x-max are set")
            lift_start = args.x_min + 0.65 * (args.x_max - args.x_min)
        else:
            lift_start = args.lift_start_x
        lift_end = args.lift_end_x if args.lift_end_x is not None else float(np.nanmax(x_out))
        denom = max(lift_end - lift_start, 1e-9)
        weights = _smoothstep((x_out - lift_start) / denom)
        out[x_col] = x_out + weights * args.shift_end_x_m
        out[y_col] = pd.to_numeric(out[y_col], errors="coerce").to_numpy(float) + weights * args.lift_end_m

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Input rectified CSV root.")
    parser.add_argument("--out", required=True, help="Output corrected CSV root.")
    parser.add_argument("--x-min", type=float, default=None, help="Minimum X_warped to keep.")
    parser.add_argument("--x-max", type=float, default=None, help="Maximum X_warped to keep.")
    parser.add_argument("--y-min", type=float, default=None, help="Minimum Y_warped to keep.")
    parser.add_argument("--y-max", type=float, default=None, help="Maximum Y_warped to keep.")
    parser.add_argument("--lift-start-x", type=float, default=None, help="X_warped where right-end Y lift starts.")
    parser.add_argument("--lift-end-x", type=float, default=None, help="X_warped where right-end Y lift reaches full value.")
    parser.add_argument("--lift-end-m", type=float, default=0.0, help="Meters added to Y_warped at/right of lift end.")
    parser.add_argument("--shift-end-x-m", type=float, default=0.0, help="Meters added to X_warped at/right of lift end.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum kept points for non-baseline CSVs.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    written = 0
    skipped = 0
    for src in sorted(root.rglob("*.csv")):
        rel = src.relative_to(root)
        dst = out_root / rel
        if process_csv(src, dst, args):
            written += 1
        else:
            skipped += 1
    print(f"Wrote {written} CSV(s) under {out_root}; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
