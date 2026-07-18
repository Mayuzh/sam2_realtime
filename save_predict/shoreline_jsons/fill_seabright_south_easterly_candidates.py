"""Build a full 10-bin x 2-clip candidate table for Seabright SE onshore.

The 2025 Santa Cruz buoy file only has 3 records in the strict Seabright
south-easterly onshore range (104-194 degrees). This script keeps those strict
records, then fills the missing candidates from the nearest wave directions to
that range. Fallback rows are explicitly marked so they can be reviewed later.

Outputs:
    candidate_clip_outputs/seabright_south_easterly_onshore_filled_candidates.csv
    candidate_clip_outputs/candidate_clip_table_filled_seabright_se.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUOY_FILE = SCRIPT_DIR / "buoy" / "santacruz2025.txt"
DEFAULT_CANDIDATE_CSV = SCRIPT_DIR / "candidate_clip_outputs" / "candidate_clip_table.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "candidate_clip_outputs"

NDBC_COLUMNS = [
    "YY",
    "MM",
    "DD",
    "hh",
    "mm",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "TIDE",
]

LOCATION = "seabright"
DIRECTION_GROUP = "south_easterly_onshore"
FINAL_PLOT_GROUP = "onshore"
STRICT_START = 104.0
STRICT_END = 194.0
TARGET_BINS = 10
CLIPS_PER_BIN = 2


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def in_range(direction: pd.Series, start: float, end: float) -> pd.Series:
    direction = direction.astype(float) % 360.0
    if start <= end:
        return (direction >= start) & (direction <= end)
    return (direction >= start) | (direction <= end)


def distance_to_interval(direction: pd.Series, start: float, end: float) -> pd.Series:
    direction = direction.astype(float) % 360.0
    inside = in_range(direction, start, end)
    clockwise_to_start = (start - direction) % 360.0
    counter_to_start = (direction - start) % 360.0
    clockwise_to_end = (end - direction) % 360.0
    counter_to_end = (direction - end) % 360.0
    distance = pd.concat(
        [
            clockwise_to_start,
            counter_to_start,
            clockwise_to_end,
            counter_to_end,
        ],
        axis=1,
    ).min(axis=1)
    return distance.where(~inside, 0.0)


def load_buoy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", comment="#", names=NDBC_COLUMNS)
    for column in ["YY", "MM", "DD", "hh", "mm", "WVHT", "DPD", "MWD"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["timestamp"] = pd.to_datetime(
        dict(
            year=frame["YY"],
            month=frame["MM"],
            day=frame["DD"],
            hour=frame["hh"],
            minute=frame["mm"],
        ),
        errors="coerce",
    )
    valid = (
        frame["timestamp"].notna()
        & frame["WVHT"].between(0.01, 30)
        & frame["DPD"].between(0.01, 40)
        & frame["MWD"].between(0, 359)
    )
    frame = frame.loc[valid].copy()
    frame["sea_state"] = frame["WVHT"].pow(2) * frame["DPD"]
    frame = frame.rename(columns={"WVHT": "H", "DPD": "T", "MWD": "direction_degrees"})
    frame["distance_from_strict_se_range_degrees"] = distance_to_interval(
        frame["direction_degrees"], STRICT_START, STRICT_END
    )
    frame["strict_direction_match"] = frame["distance_from_strict_se_range_degrees"].eq(0.0)
    return frame.sort_values("timestamp").reset_index(drop=True)


def assign_bins(frame: pd.DataFrame, target_bins: int) -> pd.DataFrame:
    frame = frame.sort_values(["sea_state", "timestamp"]).reset_index(drop=True).copy()
    ranked = frame["sea_state"].rank(method="first")
    frame["sea_state_bin"] = pd.qcut(
        ranked,
        q=target_bins,
        labels=range(1, target_bins + 1),
    ).astype(int)
    stats = (
        frame.groupby("sea_state_bin", as_index=False)["sea_state"]
        .agg(bin_min="min", bin_max="max", bin_count="size")
    )
    return frame.merge(stats, on="sea_state_bin", how="left")


def pick_two_per_bin(frame: pd.DataFrame) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for bin_number, bin_df in frame.groupby("sea_state_bin", sort=True):
        ordered = bin_df.sort_values(
            [
                "distance_from_strict_se_range_degrees",
                "strict_direction_match",
                "sea_state",
                "timestamp",
            ],
            ascending=[True, False, True, True],
        ).copy()

        if len(ordered) < CLIPS_PER_BIN:
            raise ValueError(f"Bin {bin_number} only has {len(ordered)} candidate rows.")

        # Keep direction-nearest rows, but spread them inside the bin if many ties
        # have the same direction distance.
        best_distance = ordered["distance_from_strict_se_range_degrees"].min()
        nearest = ordered[ordered["distance_from_strict_se_range_degrees"].eq(best_distance)]
        if len(nearest) >= CLIPS_PER_BIN:
            pool = nearest.sort_values(["sea_state", "timestamp"]).reset_index(drop=True)
            positions = np.linspace(0, len(pool) - 1, CLIPS_PER_BIN + 2)[1:-1]
            chosen = pool.iloc[np.unique(np.round(positions).astype(int))].copy()
        else:
            fallback = ordered[~ordered.index.isin(nearest.index)].sort_values(
                ["distance_from_strict_se_range_degrees", "sea_state", "timestamp"]
            )
            chosen = pd.concat(
                [nearest.sort_values(["sea_state", "timestamp"]), fallback.head(CLIPS_PER_BIN - len(nearest))],
                ignore_index=False,
            ).copy()
        if len(chosen) < CLIPS_PER_BIN:
            chosen = ordered.head(CLIPS_PER_BIN).copy()
        chosen = chosen.head(CLIPS_PER_BIN).copy()
        chosen["candidate_rank"] = range(1, len(chosen) + 1)
        selected_parts.append(chosen)

    return pd.concat(selected_parts, ignore_index=True)


def parse_naive_utc(value: pd.Timestamp):
    from datetime import timezone

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime().astimezone(timezone.utc)


def pick_two_per_bin_with_archive(
    frame: pd.DataFrame,
    search_window_minutes: int,
    min_archive_size_mb: float,
    max_candidates_per_bin: int,
) -> pd.DataFrame:
    """Pick two rows per bin that have nearby Walton Lighthouse archive clips."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from download_webcoos_clips import (  # pylint: disable=import-outside-toplevel
        CAMERAS,
        element_timestamp,
        iso_z,
        nearest_element,
    )

    selected_parts: list[pd.DataFrame] = []
    used_element_uuids: set[str] = set()
    camera = CAMERAS[LOCATION]
    min_archive_size_bytes = max(1, round(min_archive_size_mb * 1_000_000))

    for bin_number, bin_df in frame.groupby("sea_state_bin", sort=True):
        ordered = bin_df.sort_values(
            [
                "strict_direction_match",
                "distance_from_strict_se_range_degrees",
                "sea_state",
                "timestamp",
            ],
            ascending=[False, True, True, True],
        ).head(max_candidates_per_bin)

        chosen_rows = []
        for _, row in ordered.iterrows():
            target = parse_naive_utc(row["timestamp"])
            element, delta_seconds = nearest_element(
                camera["service_uuid"],
                target,
                search_window_minutes=search_window_minutes,
                min_archive_size_bytes=min_archive_size_bytes,
            )
            if element is None:
                continue

            try:
                size_bytes = int(element.get("data", {}).get("properties", {}).get("size") or 0)
            except (TypeError, ValueError):
                size_bytes = 0
            if size_bytes < min_archive_size_bytes:
                continue

            element_uuid = str(element.get("uuid", ""))
            if element_uuid and element_uuid in used_element_uuids:
                continue

            chosen = row.copy()
            matched_dt = element_timestamp(element)
            chosen["matched_timestamp_utc"] = iso_z(matched_dt)
            chosen["match_delta_minutes"] = round(float(delta_seconds or 0) / 60.0, 2)
            chosen["webcoos_element_uuid"] = element_uuid
            chosen["webcoos_size_bytes"] = size_bytes
            chosen_rows.append(chosen)
            if element_uuid:
                used_element_uuids.add(element_uuid)
            print(
                f"[archive ok] bin {bin_number} rank {len(chosen_rows)} "
                f"{row['timestamp']} -> {chosen['matched_timestamp_utc']} "
                f"({chosen['match_delta_minutes']:+.2f} min)"
            )
            if len(chosen_rows) >= CLIPS_PER_BIN:
                break

        if len(chosen_rows) < CLIPS_PER_BIN:
            raise ValueError(
                f"Bin {bin_number} only found {len(chosen_rows)} archive-backed rows "
                f"within +/- {search_window_minutes} minutes. Increase "
                "--max-video-match-minutes or --archive-candidates-per-bin."
            )

        chosen_df = pd.DataFrame(chosen_rows)
        chosen_df["candidate_rank"] = range(1, len(chosen_df) + 1)
        selected_parts.append(chosen_df)

    return pd.concat(selected_parts, ignore_index=True)


def build_filled_candidates(
    buoy: pd.DataFrame,
    max_relaxation_degrees: float,
    target_bins: int,
    archive_aware: bool = False,
    max_video_match_minutes: int = 10,
    min_archive_size_mb: float = 75.0,
    archive_candidates_per_bin: int = 50,
) -> pd.DataFrame:
    pool = buoy[buoy["distance_from_strict_se_range_degrees"] <= max_relaxation_degrees].copy()
    if len(pool) < target_bins * CLIPS_PER_BIN:
        raise ValueError(
            f"Only {len(pool)} records are within {max_relaxation_degrees} degrees of "
            f"the strict SE range; need at least {target_bins * CLIPS_PER_BIN}."
        )

    binned = assign_bins(pool, target_bins)
    if archive_aware:
        selected = pick_two_per_bin_with_archive(
            binned,
            search_window_minutes=max_video_match_minutes,
            min_archive_size_mb=min_archive_size_mb,
            max_candidates_per_bin=archive_candidates_per_bin,
        )
    else:
        selected = pick_two_per_bin(binned)
    selected["location"] = LOCATION
    selected["direction_group"] = DIRECTION_GROUP
    selected["final_plot_group"] = FINAL_PLOT_GROUP
    selected["selection_rule"] = np.where(
        selected["strict_direction_match"],
        "strict_104_194",
        f"nearest_direction_fallback_within_{max_relaxation_degrees:g}_deg",
    )
    selected["requested_direction_start"] = STRICT_START
    selected["requested_direction_end"] = STRICT_END

    columns = [
        "location",
        "timestamp",
        "direction_degrees",
        "direction_group",
        "final_plot_group",
        "H",
        "T",
        "sea_state",
        "sea_state_bin",
        "bin_min",
        "bin_max",
        "bin_count",
        "candidate_rank",
        "selection_rule",
        "strict_direction_match",
        "distance_from_strict_se_range_degrees",
        "requested_direction_start",
        "requested_direction_end",
    ]
    for optional_column in [
        "matched_timestamp_utc",
        "match_delta_minutes",
        "webcoos_element_uuid",
        "webcoos_size_bytes",
    ]:
        if optional_column in selected.columns:
            columns.append(optional_column)
    return selected[columns].sort_values(["sea_state_bin", "candidate_rank"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> int:
    buoy_file = resolve_path(args.buoy_file, DEFAULT_BUOY_FILE)
    candidate_csv = resolve_path(args.candidate_csv, DEFAULT_CANDIDATE_CSV)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    buoy = load_buoy(buoy_file)
    filled = build_filled_candidates(
        buoy,
        max_relaxation_degrees=args.max_relaxation_degrees,
        target_bins=args.target_bins,
        archive_aware=args.archive_aware,
        max_video_match_minutes=args.max_video_match_minutes,
        min_archive_size_mb=args.min_archive_size_mb,
        archive_candidates_per_bin=args.archive_candidates_per_bin,
    )

    filled_path = output_dir / "seabright_south_easterly_onshore_filled_candidates.csv"
    filled.to_csv(filled_path, index=False)

    if candidate_csv.exists():
        existing = pd.read_csv(candidate_csv)
        keep = ~(
            existing["location"].eq(LOCATION)
            & existing["direction_group"].eq(DIRECTION_GROUP)
        )
        combined = pd.concat([existing.loc[keep], filled], ignore_index=True, sort=False)
        combined = combined.sort_values(
            ["location", "direction_group", "sea_state_bin", "candidate_rank"]
        ).reset_index(drop=True)
        combined_path = output_dir / "candidate_clip_table_filled_seabright_se.csv"
        combined.to_csv(combined_path, index=False)
    else:
        combined_path = None

    print(f"Wrote filled SE candidates: {filled_path}")
    if combined_path:
        print(f"Wrote combined candidate table: {combined_path}")
    print(
        filled.groupby(["sea_state_bin", "selection_rule"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .to_string(index=False)
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buoy-file", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-bins", type=int, default=TARGET_BINS)
    parser.add_argument("--max-relaxation-degrees", type=float, default=20.0)
    parser.add_argument(
        "--archive-aware",
        action="store_true",
        help="Require selected rows to have nearby Walton Lighthouse archive clips.",
    )
    parser.add_argument("--max-video-match-minutes", type=int, default=10)
    parser.add_argument("--min-archive-size-mb", type=float, default=75.0)
    parser.add_argument("--archive-candidates-per-bin", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
