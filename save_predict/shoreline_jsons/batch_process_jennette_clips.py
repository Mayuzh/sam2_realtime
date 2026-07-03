"""Batch-run sam2_realtime.py on downloaded Jennette WebCOOS clips.

Outputs mirror the video folder structure and put each video's shoreline PNG/JSON
files in its own folder.

Example:
    python batch_process_jennette_clips.py --limit 2
    python batch_process_jennette_clips.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_PREDICT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAVE_PREDICT_DIR.parent


DEFAULT_INPUT_ROOT = SCRIPT_DIR / "downloaded_webcoos_clips" / "jennettes_pier"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "shoreline_outputs" / "jennettes_pier"
DEFAULT_SAM_CONFIG = REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_b+.yaml"
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def find_videos(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*.mp4"))


def output_dir_for(video_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_parent = video_path.parent.relative_to(input_root)
    return output_root / relative_parent / video_path.stem


def has_existing_json(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.glob("*.json"))


def run(args: argparse.Namespace) -> int:
    input_root = resolve_path(args.input_root, DEFAULT_INPUT_ROOT)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    sam_config = resolve_path(args.sam_config, DEFAULT_SAM_CONFIG)
    sam_checkpoint = resolve_path(args.sam_checkpoint, DEFAULT_SAM_CHECKPOINT)

    videos = find_videos(input_root)
    if args.limit is not None:
        videos = videos[: args.limit]

    if not videos:
        print(f"No .mp4 files found under {input_root}")
        return 1

    print(f"Found {len(videos)} clip(s) under {input_root}")
    print(f"Writing shoreline outputs under {output_root}")

    failures: list[tuple[Path, int]] = []
    for index, video_path in enumerate(videos, start=1):
        save_dir = output_dir_for(video_path, input_root, output_root)

        if has_existing_json(save_dir) and not args.overwrite:
            print(f"[{index}/{len(videos)}] skip existing: {save_dir}")
            continue

        cmd = [
            sys.executable,
            "sam2_realtime.py",
            "--video-path",
            str(video_path),
            "--desired-fps",
            str(args.fps),
            "--save-shorelines",
            "--save-dir",
            str(save_dir),
            "--sam-config",
            str(sam_config),
            "--sam-checkpoint",
            str(sam_checkpoint),
            "--no-display",
        ]

        if args.ignore_json:
            cmd.extend(["--ignore-json", str(resolve_path(args.ignore_json, Path(args.ignore_json)))])
        if args.ignore_blackout:
            cmd.append("--ignore-blackout")
        if args.sdp_math_only:
            cmd.append("--sdp-math-only")
        if args.perf_log:
            cmd.append("--perf-log")

        print(f"[{index}/{len(videos)}] processing: {video_path.name}")
        print(f"    output: {save_dir}")
        completed = subprocess.run(cmd, cwd=SAVE_PREDICT_DIR)
        if completed.returncode != 0:
            failures.append((video_path, completed.returncode))
            print(f"    failed with exit code {completed.returncode}")
            if args.stop_on_error:
                break

    if failures:
        print("\nFailures:")
        for video_path, code in failures:
            print(f"  {code}: {video_path}")
        return 1

    print("Batch processing complete.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None, help="Root containing downloaded Jennette .mp4 clips.")
    parser.add_argument("--output-root", default=None, help="Root for mirrored shoreline PNG/JSON outputs.")
    parser.add_argument("--fps", type=float, default=5.0, help="Frame sampling rate for shoreline detection.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess clips even if JSONs already exist.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the batch on the first failed clip.")
    parser.add_argument("--ignore-json", default=None, help="Optional ignore-region LabelMe JSON passed through to sam2_realtime.py.")
    parser.add_argument("--ignore-blackout", action="store_true", help="Enable ignore-region blackout in sam2_realtime.py.")
    parser.add_argument("--sdp-math-only", action="store_true", help="Force math attention backend.")
    parser.add_argument("--perf-log", action="store_true", help="Print performance logs from sam2_realtime.py.")
    parser.add_argument("--sam-config", default=None, help="Override SAM config path.")
    parser.add_argument("--sam-checkpoint", default=None, help="Override SAM checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
