"""Download nearest WebCOOS 10-minute archive clips for candidate buoy rows.

Example:
    python download_webcoos_clips.py --dry-run
    python download_webcoos_clips.py

The input CSV is expected to come from sea_state_heatmap_clip_candidates.ipynb.
It should include at least: location, timestamp, direction_group, sea_state_bin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://app.webcoos.org/webcoos/api/v1"
API_TOKEN = os.environ.get("WEBCOOS_API_TOKEN", "4af70560667182ab2a37fbca973cfdde4ea058ce")
SCRIPT_DIR = Path(__file__).resolve().parent

CAMERAS = {
    "jennettes_pier": {
        "camera_slug": "jennette_north",
        "service_uuid": "0c431456-41e0-4e99-82f3-8c2457b4985d",
        "archive_url": "https://webcoos.org/cameras/jennette_north/?gallery=jennette_north-video-archive-s3",
    },
    "jennette_north": {
        "camera_slug": "jennette_north",
        "service_uuid": "0c431456-41e0-4e99-82f3-8c2457b4985d",
        "archive_url": "https://webcoos.org/cameras/jennette_north/?gallery=jennette_north-video-archive-s3",
    },
    "seabright": {
        "camera_slug": "walton_lighthouse",
        "service_uuid": "f33cd48f-c9b8-450a-82f2-0a2c25a8fcce",
        "archive_url": "https://webcoos.org/cameras/walton_lighthouse/?gallery=walton_lighthouse-video-archive-s3",
    },
    "walton_lighthouse": {
        "camera_slug": "walton_lighthouse",
        "service_uuid": "f33cd48f-c9b8-450a-82f2-0a2c25a8fcce",
        "archive_url": "https://webcoos.org/cameras/walton_lighthouse/?gallery=walton_lighthouse-video-archive-s3",
    },
}


def parse_timestamp(value: str) -> datetime:
    """Parse a candidate timestamp and treat timezone-naive values as UTC."""
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            raise

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def request_json(url: str, retries: int = 3, sleep_seconds: float = 1.0) -> dict[str, Any]:
    headers = {"Authorization": f"Token {API_TOKEN}", "Accept": "application/json"}
    request = Request(url, headers=headers)

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"WebCOOS request failed after {retries} tries: {url}") from exc
            time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"WebCOOS request failed: {url}")


def request_file(url: str, output_path: Path, retries: int = 3, sleep_seconds: float = 1.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url), timeout=120) as response, temp_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if temp_path.stat().st_size == 0:
                raise OSError(f"Downloaded an empty file: {url}")
            temp_path.replace(output_path)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} tries: {url}") from exc
            time.sleep(sleep_seconds * attempt)


def fetch_elements(service_uuid: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch all archive elements for one camera between start and end UTC."""
    elements: list[dict[str, Any]] = []
    page = 1
    total_pages = None

    while total_pages is None or page <= total_pages:
        params = {
            "service": service_uuid,
            "page": page,
            "starting_after": iso_z(start),
            "starting_before": iso_z(end),
        }
        url = f"{API_BASE_URL}/elements/?{urlencode(params)}"
        payload = request_json(url)
        elements.extend(payload.get("results", []))
        pagination = payload.get("pagination", {})
        total_pages = int(pagination.get("total_pages") or 1)
        page += 1

    return elements


def element_timestamp(element: dict[str, Any]) -> datetime:
    value = element["data"]["extents"]["temporal"]["min"]
    return parse_timestamp(value)


def nearest_element(
    service_uuid: str,
    target: datetime,
    search_window_minutes: int,
    min_archive_size_bytes: int = 1,
) -> tuple[dict[str, Any] | None, int | None]:
    start = target - timedelta(minutes=search_window_minutes)
    end = target + timedelta(minutes=search_window_minutes)
    elements = fetch_elements(service_uuid, start, end)
    downloadable_elements = []
    preferred_elements = []
    for element in elements:
        properties = element.get("data", {}).get("properties", {})
        try:
            size_bytes = int(properties.get("size") or 0)
        except (TypeError, ValueError):
            size_bytes = 0
        if properties.get("url") and size_bytes > 0:
            downloadable_elements.append(element)
            if size_bytes >= min_archive_size_bytes:
                preferred_elements.append(element)

    if not downloadable_elements:
        return None, None

    usable_elements = preferred_elements or downloadable_elements
    best = min(
        usable_elements,
        key=lambda item: abs((element_timestamp(item) - target).total_seconds()),
    )
    delta_seconds = int((element_timestamp(best) - target).total_seconds())
    return best, delta_seconds


def output_path_for(row: dict[str, str], camera_slug: str, matched_dt: datetime, output_dir: Path) -> Path:
    location = safe_name(row.get("location", camera_slug))
    direction_group = safe_name(row.get("direction_group", "unknown_direction"))
    sea_state_bin = safe_name(row.get("sea_state_bin", "unknown_bin"))
    timestamp_name = matched_dt.strftime("%Y-%m-%d_%H%M%SZ")
    filename = f"{camera_slug}-{timestamp_name}.mp4"
    return output_dir / location / direction_group / f"bin_{sea_state_bin}" / filename


def build_manifest_row(
    row: dict[str, str],
    camera_slug: str,
    element: dict[str, Any] | None,
    delta_seconds: int | None,
    output_path: Path | None,
    status: str,
) -> dict[str, Any]:
    data = element.get("data", {}) if element else {}
    properties = data.get("properties", {}) if element else {}
    matched_dt = element_timestamp(element) if element else None

    manifest = dict(row)
    manifest.update(
        {
            "camera_slug": camera_slug,
            "requested_timestamp_utc": iso_z(parse_timestamp(row["timestamp"])),
            "matched_timestamp_utc": iso_z(matched_dt) if matched_dt else "",
            "match_delta_seconds": delta_seconds if delta_seconds is not None else "",
            "match_delta_minutes": round(delta_seconds / 60, 2) if delta_seconds is not None else "",
            "webcoos_element_uuid": element.get("uuid", "") if element else "",
            "webcoos_label": data.get("common", {}).get("label", ""),
            "webcoos_url": properties.get("url", ""),
            "webcoos_size_bytes": properties.get("size", ""),
            "clip_path": str(output_path) if output_path else "",
            "download_status": status,
        }
    )
    return manifest


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    candidate_csv = resolve_path(args.candidate_csv)
    output_dir = resolve_path(args.output_dir)
    manifest_path = resolve_path(args.manifest)
    candidates = read_candidates(candidate_csv)
    if args.location:
        requested_locations = {safe_name(value) for value in args.location}
        candidates = [
            row
            for row in candidates
            if safe_name(row.get("location", "")) in requested_locations
        ]
        if not candidates:
            available = sorted(
                {safe_name(row.get("location", "")) for row in read_candidates(candidate_csv)}
            )
            parser_locations = ", ".join(available) or "none"
            raise ValueError(
                f"No candidates matched --location {sorted(requested_locations)}. "
                f"Available locations: {parser_locations}"
            )
    if args.limit is not None:
        candidates = candidates[: args.limit]
    manifest_rows: list[dict[str, Any]] = []

    for index, row in enumerate(candidates, start=1):
        location = safe_name(row.get("location", ""))
        camera = CAMERAS.get(location)
        if not camera:
            print(f"[{index}/{len(candidates)}] skipped unknown location: {row.get('location')}")
            manifest_rows.append(build_manifest_row(row, "", None, None, None, "unknown_location"))
            continue

        target = parse_timestamp(row["timestamp"])
        element, delta_seconds = nearest_element(
            camera["service_uuid"],
            target,
            search_window_minutes=args.search_window_minutes,
            min_archive_size_bytes=max(1, round(args.min_archive_size_mb * 1_000_000)),
        )

        if not element:
            print(f"[{index}/{len(candidates)}] no archive clip within +/- {args.search_window_minutes} min for {row['timestamp']}")
            manifest_rows.append(build_manifest_row(row, camera["camera_slug"], None, None, None, "no_match"))
            continue

        matched_dt = element_timestamp(element)
        clip_path = output_path_for(row, camera["camera_slug"], matched_dt, output_dir)
        url = element["data"]["properties"]["url"]
        delta_min = round((delta_seconds or 0) / 60, 2)

        if args.dry_run:
            status = "dry_run"
        elif clip_path.exists() and not args.overwrite:
            status = "exists"
        else:
            print(f"[{index}/{len(candidates)}] downloading {camera['camera_slug']} {iso_z(matched_dt)} ({delta_min:+} min)")
            request_file(url, clip_path)
            status = "downloaded"

        if args.dry_run:
            print(f"[{index}/{len(candidates)}] match {camera['camera_slug']} {row['timestamp']} -> {iso_z(matched_dt)} ({delta_min:+} min)")

        manifest_rows.append(build_manifest_row(row, camera["camera_slug"], element, delta_seconds, clip_path, status))

    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        default="./candidate_clip_outputs/candidate_clip_table.csv",
        help="CSV made by sea_state_heatmap_clip_candidates.ipynb.",
    )
    parser.add_argument(
        "--output-dir",
        default="./downloaded_webcoos_clips",
        help="Directory where MP4 clips will be saved.",
    )
    parser.add_argument(
        "--manifest",
        default="./downloaded_webcoos_clips/download_manifest.csv",
        help="CSV manifest with requested timestamps, matched timestamps, URLs, and paths.",
    )
    parser.add_argument(
        "--search-window-minutes",
        type=int,
        default=24 * 60,
        help="Nearest-match search window before/after each requested timestamp.",
    )
    parser.add_argument(
        "--min-archive-size-mb",
        type=float,
        default=0.0,
        help=(
            "Ignore smaller archive objects when matching. Walton Lighthouse "
            "uses about 75 MB as a practical floor for full-length clips."
        ),
    )
    parser.add_argument(
        "--location",
        action="append",
        help=(
            "Only process this candidate location (for example, seabright). "
            "Repeat the option to include more than one location."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Find nearest clips and write manifest without downloading.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download clips even if the output file already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N candidate rows.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
