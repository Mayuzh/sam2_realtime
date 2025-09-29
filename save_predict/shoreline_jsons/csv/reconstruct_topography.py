#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beach_profile_reconstruct.py

Reconstruct a cross-shore beach profile z(t) from shoreline polylines and NOAA water levels.

Inputs
------
--shoreline_root : folder containing shoreline CSVs (recursively).
                   Each CSV has at least x,y OR X_warped,Y_warped and a filename with UTC timestamp,
                   e.g. walton_lighthouse-2024-11-16-161653Z_000105_warped.csv
--water_csv      : NOAA station 9413450 water levels with columns like:
                   "Date","Time (GMT)","Verified (ft)"
--out_root       : output folder

Outputs
-------
- profile_points.csv : one row per shoreline file with [file, clip_dt_utc, noaa_dt_utc, dt_diff_sec, s0, t_m, z_m, z_ft]
- profile_curve.csv  : monotone (non-increasing) z vs t curve
- profile_plot.png   : scatter + fitted monotone curve
- contours_map.png   : shoreline polylines colored by tide level (centered XY for readability)
- st_scatter.png     : all points in (s,t) after PCA, colored by tide (debug)

Notes
-----
- Uses a single alongshore transect s = s0. For each shoreline, we intersect the polyline
  with s=s0 (or use a small |s - s0| bin) to get t(s0). Pair with water level z for that clip.
- Automatically orients cross-shore t to point seaward (so higher tide -> smaller/landward t).
- No SciPy dependency. Pure numpy/pandas/matplotlib.
"""

from __future__ import annotations
import argparse, os, re, sys, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------ CLI ------------------

def get_args():
    ap = argparse.ArgumentParser(description="Reconstruct cross-shore beach profile from shorelines + NOAA water levels.")
    ap.add_argument('--shoreline_root', default='./seabright_rec',
                    help='Root folder containing shoreline CSVs (recursively).')
    ap.add_argument('--water_csv', default='./waterlevel/CO-OPS_9413450_wl.csv',
                    help='NOAA water level CSV with Date, Time (GMT), Verified (ft).')
    ap.add_argument('--out_root', default='./vis_topography', help='Output folder.')
    # Expert knobs (safe defaults)
    ap.add_argument('--tol_minutes', type=int, default=15, help='Max |Δt| to nearest NOAA record (minutes).')
    ap.add_argument('--interpolate', action='store_true', help='Linearly interpolate NOAA z between 6-min records.')
    ap.add_argument('--min_points', type=int, default=20, help='Min vertices needed to use a shoreline file.')
    ap.add_argument('--ds_bin', type=float, default=2.0, help='Half-width alongshore bin for transect intersection (meters).')
    ap.add_argument('--s0', type=float, default=None, help='Alongshore transect location in s. If None, auto-detect (median).')
    ap.add_argument('--plot_subsample', type=int, default=1, help='Plot every Nth point in debug scatter to keep files light.')
    return ap.parse_args()


# ------------------ Utilities ------------------

TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}-\d{6})Z')

def parse_time_from_filename(path: str) -> Optional[pd.Timestamp]:
    """Parse UTC timestamp of form YYYY-MM-DD-HHMMSSZ from filename."""
    name = os.path.basename(path)
    m = TS_RE.search(name)
    if not m:
        return None
    ts = m.group(1)
    try:
        return pd.to_datetime(ts, format="%Y-%m-%d-%H%M%S", utc=True)
    except Exception:
        return None


def find_csv_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for dp, _, fns in os.walk(root_dir):
        # Skip known non-shoreline folders
        dpl = dp.lower()
        if ('waterlevel' in dpl) or ('vis_topography' in dpl):
            continue
        for f in fns:
            if f.lower().endswith('.csv'):
                out.append(os.path.join(dp, f))
    out.sort()
    return out

def load_noaa_water_levels(csv_path: str) -> pd.DataFrame:
    """
    Robustly load NOAA water levels with columns like:
    "Date","Time (GMT)","Verified (ft)".
    Returns a DataFrame indexed by UTC timestamp with columns ['z_ft','z_m'].
    """
    import pandas as pd
    import numpy as np
    import unicodedata

    # 1) Read CSV (handle BOM/odd encodings).
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1")

    # 2) Normalize headers and pull needed columns (very permissive).
    df.columns = (df.columns.astype(str)
                  .str.strip().str.strip('"').str.lower())
    def col(name_opts):
        for n in name_opts:
            for c in df.columns:
                if n in c:
                    return c
        return None

    c_date = col(['date'])
    c_time = col(['time (gmt)', 'time (utc)', 'time'])
    c_ver  = col(['verified (ft)', 'verified'])
    if not all([c_date, c_time, c_ver]):
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    # 3) Clean strings (remove BOM/ZWSP/NBSP), normalize unicode.
    def clean_series(s):
        s = s.astype(str)
        s = (s.str.replace('"', '', regex=False)
               .str.replace('\ufeff', '', regex=False)
               .str.replace('\u200b', '', regex=False)   # zero-width space
               .str.replace('\xa0', ' ', regex=False))   # NBSP -> space
        return s.map(lambda x: unicodedata.normalize('NFKC', x).strip())

    date_s = clean_series(df[c_date])
    time_s = clean_series(df[c_time]).str.replace(r'[^0-9:]', '', regex=True)

    # 4) Build UTC timestamps with a strict format matching your file.
    ts = pd.to_datetime(date_s + ' ' + time_s,
                        format="%Y/%m/%d %H:%M", utc=True, errors='coerce')

    # 5) Clean z; accept only digits/dot/minus. Map lone '-' to NaN.
    z_raw = clean_series(df[c_ver])
    z_raw = z_raw.str.replace('[\u2012\u2013\u2014\u2212]', '-', regex=True)
    z_raw = z_raw.replace({'-': np.nan, '': np.nan})
    z_raw = z_raw.str.replace(r'[^0-9\.\-+]', '', regex=True)
    z_ft = pd.to_numeric(z_raw, errors='coerce')

    # 6) Build keep mask directly from original Series to avoid any index-boolean quirks.
    keep = (~ts.isna()) & (~z_ft.isna())
    if keep.sum() == 0:
        out = pd.DataFrame(columns=['z_ft', 'z_m'])
    else:
        ts_kept = ts[keep]
        z_kept = z_ft[keep]
        out = pd.DataFrame({'z_ft': z_kept.to_numpy(),
                            'z_m': (z_kept * 0.3048).to_numpy()}, index=ts_kept)
        out = out.sort_index()

    # 7) Diagnostics
    if out.empty:
        print("[water] Still empty after minimal loader. Showing first 5 parsed rows:")
        dbg = pd.DataFrame({
            'date': date_s.head(5),
            'time': time_s.head(5),
            'ts_is_na': ts.head(5).isna(),
            'verified_raw': df[c_ver].head(5).astype(str),
            'verified_clean': z_raw.head(5),
            'z_ft_is_na': z_ft.head(5).isna(),
        })
        print(dbg)
    else:
        print(f"[water] Loaded {len(out)} water rows "
              f"({out.index.min()} .. {out.index.max()})")

    return out




def load_shoreline_xy(path: str) -> Optional[np.ndarray]:
    """Load shoreline XY (rectified if available else pixels)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # Filter only when 'shoreline' actually appears
    if 'label' in df.columns and (df['label'].astype(str) == 'shoreline').any():
        df = df[df['label'].astype(str) == 'shoreline']
    # Prefer rectified
    if {'X_warped', 'Y_warped'}.issubset(df.columns):
        arr = df[['X_warped', 'Y_warped']].to_numpy(dtype=float)
    elif {'x', 'y'}.issubset(df.columns):
        arr = df[['x', 'y']].to_numpy(dtype=float)
    else:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return None
    return arr


def compute_pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return origin and 2x2 rotation R where rows are basis vectors [pc1; pc2]."""
    origin = np.nanmean(points, axis=0)
    X = points - origin
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)       # ascending eigenvalues
    pc1 = V[:, 1]                  # alongshore
    pc2 = V[:, 0]                  # cross-shore
    R = np.vstack([pc1, pc2])      # rows are basis vectors
    return origin, R


def to_st(xy: np.ndarray, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ (xy - origin).T).T  # (N,2): [s, t]


def nearest_or_interp(wdf: pd.DataFrame, dt: pd.Timestamp, tol_min: int = 15, interpolate: bool = True
                      ) -> Optional[Tuple[pd.Timestamp, float, float]]:
    """Find nearest NOAA record within tol; if interpolate=True and dt inside gap, linearly interpolate."""
    if dt.tzinfo is None:
        dt = dt.tz_localize('UTC')
    # out of range?
    if dt < wdf.index[0] or dt > wdf.index[-1]:
        return None

    # exact hit?
    if dt in wdf.index:
        row = wdf.loc[dt]
        return dt, float(row['z_ft']), float(row['z_m'])

    # locate bracketing
    loc = wdf.index.searchsorted(dt)
    i0 = loc - 1
    i1 = loc
    if i0 < 0 or i1 >= len(wdf):
        return None

    t0, t1 = wdf.index[i0], wdf.index[i1]
    z0, z1 = wdf.iloc[i0]['z_m'], wdf.iloc[i1]['z_m']
    # too far?
    if min(abs((dt - t0).total_seconds()), abs((t1 - dt).total_seconds())) > tol_min * 60:
        return None

    if interpolate:
        # linear interpolation in z_m; also return z_ft from z_m / 0.3048 for convenience
        alpha = (dt - t0) / (t1 - t0)
        zm = float((1 - alpha) * z0 + alpha * z1)
        zf = zm / 0.3048
        # for reporting, choose the closer timestamp
        rep_t = t0 if abs(dt - t0) <= abs(t1 - dt) else t1
        return rep_t, float(zf), float(zm)
    else:
        # nearest neighbor
        if abs(dt - t0) <= abs(t1 - dt):
            row = wdf.iloc[i0]
            return t0, float(row['z_ft']), float(row['z_m'])
        else:
            row = wdf.iloc[i1]
            return t1, float(row['z_ft']), float(row['z_m'])


def t_at_transect(ST: np.ndarray, s0: float, ds_bin: float = 2.0) -> Optional[float]:
    """
    Return cross-shore t where the shoreline crosses s = s0.

    Strategy hierarchy:
    1) Bin points with |s - s0| <= ds_bin and take median t (robust).
    2) Sort by s and linearly interpolate t at crossings of s0 across segments.
    3) Fallback to nearest point in |s - s0|.
    """
    s, t = ST[:, 0], ST[:, 1]
    # (1) bin
    mask = np.abs(s - s0) <= ds_bin
    if np.any(mask):
        return float(np.nanmedian(t[mask]))

    # (2) interpolate across crossings
    order = np.argsort(s)
    s_sorted = s[order]
    t_sorted = t[order]
    hits: List[float] = []
    for i in range(len(s_sorted) - 1):
        a, b = s_sorted[i], s_sorted[i + 1]
        if (a - s0) == 0:
            hits.append(float(t_sorted[i]))
        if (a - s0) * (b - s0) < 0:  # crosses
            frac = (s0 - a) / (b - a)
            tt = t_sorted[i] + frac * (t_sorted[i + 1] - t_sorted[i])
            hits.append(float(tt))
    if hits:
        return float(np.nanmedian(hits))

    # (3) nearest
    j = int(np.nanargmin(np.abs(s - s0)))
    return float(t[j]) if len(t) else None


def pava_monotone_decreasing(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pool-Adjacent-Violators to enforce y is non-increasing in x.
    Returns (x_avg, y_monotone). x must be ascending.
    """
    # group as blocks
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return x, y
    blocks = [{'sumx': x[0], 'sumy': y[0], 'n': 1}]
    for i in range(1, len(x)):
        blocks.append({'sumx': x[i], 'sumy': y[i], 'n': 1})
        # while violation (last-1 y < last y) for decreasing sequence
        while len(blocks) >= 2 and (blocks[-2]['sumy']/blocks[-2]['n']) < (blocks[-1]['sumy']/blocks[-1]['n']):
            # merge last two
            b2 = blocks.pop()
            b1 = blocks.pop()
            merged = {'sumx': b1['sumx'] + b2['sumx'],
                      'sumy': b1['sumy'] + b2['sumy'],
                      'n': b1['n'] + b2['n']}
            blocks.append(merged)
    # unpack
    x_out, y_out = [], []
    for b in blocks:
        xv = (b['sumx'] / b['n'])
        yv = (b['sumy'] / b['n'])
        x_out.extend([xv] * b['n'])
        y_out.extend([yv] * b['n'])
    return np.asarray(x_out), np.asarray(y_out)


# ------------------ Main logic ------------------

@dataclass
class Record:
    file: str
    clip_dt: pd.Timestamp
    noaa_dt: pd.Timestamp
    dt_diff_sec: float
    s0: float
    t_m: float
    z_m: float
    z_ft: float


def main():
    args = get_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Load NOAA water levels
    wdf = load_noaa_water_levels(args.water_csv)
    if wdf.empty:
        # Provide diagnostics to help users fix CSV path/format issues
        try:
            preview = pd.read_csv(args.water_csv, nrows=5)
            print(f"ERROR: water level table is empty. File: {args.water_csv}\nPreview head:\n{preview}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: water level table is empty and CSV preview failed: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[water] Loaded {len(wdf)} rows from '{args.water_csv}'. Range: {wdf.index.min()} .. {wdf.index.max()}")

    # Gather shoreline points to define PCA frame
    files = find_csv_files(args.shoreline_root)
    all_xy: List[np.ndarray] = []
    valid_files: List[str] = []
    for p in files:
        xy = load_shoreline_xy(p)
        if xy is None or len(xy) < args.min_points:
            continue
        all_xy.append(xy)
        valid_files.append(p)

    if not all_xy:
        print("ERROR: Found no usable shoreline CSVs.", file=sys.stderr)
        sys.exit(1)

    all_points = np.vstack(all_xy)
    origin, R = compute_pca_axes(all_points)

    # Choose transect s0
    ST_all = to_st(all_points, origin, R)
    s0 = float(np.nanmedian(ST_all[:, 0])) if args.s0 is None else float(args.s0)

    # First pass extraction to test orientation
    tmp_ts, tmp_z = [], []
    tmp_rows: List[Record] = []
    dropped = 0

    for p, xy in zip(valid_files, all_xy):
        dt = parse_time_from_filename(p)
        if dt is None:
            dropped += 1
            continue
        hit = nearest_or_interp(wdf, dt, tol_min=args.tol_minutes, interpolate=args.interpolate)
        if hit is None:
            dropped += 1
            continue
        noaa_dt, z_ft, z_m = hit
        ST = to_st(xy, origin, R)
        t_here = t_at_transect(ST, s0=s0, ds_bin=args.ds_bin)
        if t_here is None or not np.isfinite(t_here):
            dropped += 1
            continue
        dsec = abs((noaa_dt - dt).total_seconds())
        tmp_rows.append(Record(file=os.path.relpath(p, start=args.shoreline_root),
                               clip_dt=dt, noaa_dt=noaa_dt, dt_diff_sec=dsec,
                               s0=s0, t_m=float(t_here), z_m=float(z_m), z_ft=float(z_ft)))
        tmp_ts.append(t_here)
        tmp_z.append(z_m)

    if len(tmp_rows) < 5:
        print("ERROR: Too few matched shoreline+water points. "
              f"Matched={len(tmp_rows)}, dropped={dropped}", file=sys.stderr)
        sys.exit(1)

    # Ensure cross-shore points offshore correspond to *lower* elevations (negative correlation).
    corr = np.corrcoef(np.asarray(tmp_ts), np.asarray(tmp_z))[0, 1]
    if corr > 0:
        # flip cross-shore direction
        R[1, :] *= -1
        # recompute t values
        tmp_rows2: List[Record] = []
        for r in tmp_rows:
            # re-load xy for this file (we need xy again)
            p_full = os.path.join(args.shoreline_root, r.file)
            xy = load_shoreline_xy(p_full)
            ST = to_st(xy, origin, R)
            t_here = t_at_transect(ST, s0=s0, ds_bin=args.ds_bin)
            r.t_m = float(t_here)
            tmp_rows2.append(r)
        tmp_rows = tmp_rows2

    # Save profile points
    prof_df = pd.DataFrame([{
        'file': r.file,
        'clip_dt_utc': r.clip_dt.isoformat(),
        'noaa_dt_utc': r.noaa_dt.isoformat(),
        'dt_diff_sec': r.dt_diff_sec,
        's0': r.s0,
        't_m': r.t_m,
        'z_m': r.z_m,
        'z_ft': r.z_ft
    } for r in tmp_rows]).sort_values('t_m').reset_index(drop=True)

    prof_csv = os.path.join(args.out_root, 'profile_points.csv')
    prof_df.to_csv(prof_csv, index=False)

    # Fit monotone (non-increasing) z(t)
    t_sorted = prof_df['t_m'].to_numpy()
    z_sorted = prof_df['z_m'].to_numpy()
    # make t ascending
    order = np.argsort(t_sorted)
    t_sorted = t_sorted[order]
    z_sorted = z_sorted[order]
    t_fit, z_fit = pava_monotone_decreasing(t_sorted, z_sorted)

    curve_df = pd.DataFrame({'t_m': t_fit, 'z_m': z_fit})
    curve_csv = os.path.join(args.out_root, 'profile_curve.csv')
    curve_df.to_csv(curve_csv, index=False)

    # -------- Plots --------

    # 1) Profile plot
    plt.figure(figsize=(7.6, 5.2))
    plt.scatter(t_sorted, z_sorted, s=24, alpha=0.8, label='Transect samples (tide levels)')
    plt.plot(t_fit, z_fit, linewidth=2.25, label='Monotone fit (z decreasing seaward)')
    plt.xlabel('Cross-shore distance t (meters or pixels)')
    plt.ylabel('Elevation z (m, MLLW)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, 'profile_plot.png'), dpi=220)
    plt.close()

    # 2) Contours map (centered for readability)
    #    use same vmin/vmax from all used z_m
    vmin, vmax = float(np.min(z_sorted)), float(np.max(z_sorted))
    cmap = plt.cm.viridis
    plt.figure(figsize=(7.5, 6.0))
    for r in tmp_rows:
        p_full = os.path.join(args.shoreline_root, r.file)
        xy = load_shoreline_xy(p_full)
        # center for plot so axes aren't ~1e7
        xy_c = xy - origin
        cval = (r.z_m - vmin) / max(vmax - vmin, 1e-9)
        col = cmap(cval)
        # sort by alongshore for a cleaner line
        ST = to_st(xy, origin, R)
        order = np.argsort(ST[:, 0])
        plt.plot(xy_c[order, 0], xy_c[order, 1], color=col, alpha=0.8, linewidth=1.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Tide level z (m, MLLW)')
    plt.xlabel('X (rectified, centered)')
    plt.ylabel('Y (rectified, centered)')
    plt.title('Shoreline contours colored by tide level')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, 'contours_map.png'), dpi=220)
    plt.close()

    # 3) Debug scatter in (s,t)
    all_rows_st = []
    all_rows_z = []
    for r in tmp_rows:
        p_full = os.path.join(args.shoreline_root, r.file)
        xy = load_shoreline_xy(p_full)
        ST = to_st(xy, origin, R)
        if args.plot_subsample > 1:
            ST = ST[::args.plot_subsample]
        all_rows_st.append(ST)
        all_rows_z.append(np.full(len(ST), r.z_m))
    ST_cat = np.vstack(all_rows_st)
    Z_cat = np.concatenate(all_rows_z)
    plt.figure(figsize=(8.3, 5.2))
    im = plt.scatter(ST_cat[:, 0], ST_cat[:, 1], c=Z_cat, s=3, cmap='viridis')
    plt.axvline(s0, color='k', linestyle='--', linewidth=1.1, alpha=0.7, label='transect s0')
    plt.colorbar(im, label='Tide level z (m, MLLW)')
    plt.xlabel('Alongshore s (PCA)')
    plt.ylabel('Cross-shore t (PCA)')
    plt.title('All shoreline points in local (s,t)')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, 'st_scatter.png'), dpi=220)
    plt.close()

    # Summary
    print(f"[OK] Wrote:\n  - {prof_csv}\n  - {curve_csv}\n  - profile_plot.png\n  - contours_map.png\n  - st_scatter.png")
    print(f"Matched shoreline files: {len(tmp_rows)}  (dropped: {dropped})")
    print(f"Transect s0: {s0:.3f} (units of alongshore PCA). "
          f"Correlation sign after orientation fix should be <= 0.")


if __name__ == '__main__':
    main()
