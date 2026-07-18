#!/usr/bin/env python3
"""Convert rectified Web Mercator shoreline CSV coordinates to British National Grid.

The Trevone control-point files currently contain destination coordinates that
look like EPSG:3857 Web Mercator. This script preserves the source columns and
adds/replaces X_warped/Y_warped with EPSG:27700 British National Grid eastings
and northings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


WEB_MERCATOR_RADIUS = 6378137.0


def webmercator_to_wgs84(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(x / WEB_MERCATOR_RADIUS)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS)) - math.pi / 2.0)
    return lon, lat


def wgs84_to_osgb36_helmert(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate WGS84 lon/lat to OSGB36 lon/lat using Helmert transform."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    h = np.zeros_like(lon)

    # WGS84 ellipsoid.
    a = 6378137.0
    b = 6356752.314245
    e2 = 1.0 - (b * b) / (a * a)
    nu = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x1 = (nu + h) * np.cos(lat) * np.cos(lon)
    y1 = (nu + h) * np.cos(lat) * np.sin(lon)
    z1 = ((1.0 - e2) * nu + h) * np.sin(lat)

    # Inverse of OSGB36->WGS84 transform, commonly used for WGS84->OSGB36.
    tx, ty, tz = -446.448, 125.157, -542.060
    rx = math.radians(-0.1502 / 3600.0)
    ry = math.radians(-0.2470 / 3600.0)
    rz = math.radians(-0.8421 / 3600.0)
    s = -20.4894e-6

    x2 = tx + (1.0 + s) * x1 + (-rz) * y1 + (ry) * z1
    y2 = ty + (rz) * x1 + (1.0 + s) * y1 + (-rx) * z1
    z2 = tz + (-ry) * x1 + (rx) * y1 + (1.0 + s) * z1

    # Airy 1830 ellipsoid.
    a2 = 6377563.396
    b2 = 6356256.909
    e22 = 1.0 - (b2 * b2) / (a2 * a2)
    p = np.sqrt(x2 * x2 + y2 * y2)
    lat2 = np.arctan2(z2, p * (1.0 - e22))
    for _ in range(8):
        nu2 = a2 / np.sqrt(1.0 - e22 * np.sin(lat2) ** 2)
        lat2 = np.arctan2(z2 + e22 * nu2 * np.sin(lat2), p)
    lon2 = np.arctan2(y2, x2)
    return lon2, lat2


def osgb36_to_bng(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project OSGB36 lon/lat radians to EPSG:27700 eastings/northings."""
    a = 6377563.396
    b = 6356256.909
    f0 = 0.9996012717
    lat0 = math.radians(49.0)
    lon0 = math.radians(-2.0)
    n0 = -100000.0
    e0 = 400000.0
    e2 = 1.0 - (b * b) / (a * a)
    n = (a - b) / (a + b)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    tan_lat = np.tan(lat)
    nu = a * f0 / np.sqrt(1.0 - e2 * sin_lat ** 2)
    rho = a * f0 * (1.0 - e2) / (1.0 - e2 * sin_lat ** 2) ** 1.5
    eta2 = nu / rho - 1.0

    ma = (1.0 + n + 5.0 / 4.0 * n**2 + 5.0 / 4.0 * n**3) * (lat - lat0)
    mb = (3.0 * n + 3.0 * n**2 + 21.0 / 8.0 * n**3) * np.sin(lat - lat0) * np.cos(lat + lat0)
    mc = (15.0 / 8.0 * n**2 + 15.0 / 8.0 * n**3) * np.sin(2.0 * (lat - lat0)) * np.cos(2.0 * (lat + lat0))
    md = 35.0 / 24.0 * n**3 * np.sin(3.0 * (lat - lat0)) * np.cos(3.0 * (lat + lat0))
    meridional = b * f0 * (ma - mb + mc - md)

    dlon = lon - lon0
    i = meridional + n0
    ii = nu / 2.0 * sin_lat * cos_lat
    iii = nu / 24.0 * sin_lat * cos_lat**3 * (5.0 - tan_lat**2 + 9.0 * eta2)
    iiia = nu / 720.0 * sin_lat * cos_lat**5 * (61.0 - 58.0 * tan_lat**2 + tan_lat**4)
    iv = nu * cos_lat
    v = nu / 6.0 * cos_lat**3 * (nu / rho - tan_lat**2)
    vi = nu / 120.0 * cos_lat**5 * (5.0 - 18.0 * tan_lat**2 + tan_lat**4 + 14.0 * eta2 - 58.0 * tan_lat**2 * eta2)

    northing = i + ii * dlon**2 + iii * dlon**4 + iiia * dlon**6
    easting = e0 + iv * dlon + v * dlon**3 + vi * dlon**5
    return easting, northing


def webmercator_to_bng(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_deg, lat_deg = webmercator_to_wgs84(x, y)
    lon_osgb, lat_osgb = wgs84_to_osgb36_helmert(lon_deg, lat_deg)
    return osgb36_to_bng(lon_osgb, lat_osgb)


def process_csv(src: Path, dst: Path) -> bool:
    df = pd.read_csv(src)
    lower = {c.lower(): c for c in df.columns}
    x_col = lower.get("x_warped")
    y_col = lower.get("y_warped")
    if x_col is None or y_col is None:
        return False

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    easting, northing = webmercator_to_bng(x, y)
    df[x_col] = easting
    df[y_col] = northing
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Input CSV root with EPSG:3857 X_warped/Y_warped.")
    parser.add_argument("--out", required=True, help="Output CSV root with EPSG:27700 X_warped/Y_warped.")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    written = 0
    skipped = 0
    for src in sorted(root.rglob("*.csv")):
        rel = src.relative_to(root)
        if process_csv(src, out_root / rel):
            written += 1
        else:
            skipped += 1
    print(f"Wrote {written} BNG CSV(s) under {out_root}; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
