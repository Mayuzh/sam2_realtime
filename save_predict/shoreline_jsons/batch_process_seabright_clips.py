"""Batch-run SAM2 only on stable Seabright portions of Walton clips.

The camera alternates between Seabright and Twin Lakes. A reference-image
pre-pass identifies stable Seabright raw-frame ranges before shoreline tracking.

Examples:
    python batch_process_seabright_clips.py --scan-only
    python batch_process_seabright_clips.py --limit 1
    python batch_process_seabright_clips.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_PREDICT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAVE_PREDICT_DIR.parent
sys.path.insert(0, str(SAVE_PREDICT_DIR))

from utils.view_filter import (  # noqa: E402
    ReferenceViewClassifier,
    format_frame_ranges,
    scan_video_views,
    stable_target_ranges,
)


DEFAULT_INPUT_ROOT = SCRIPT_DIR / "downloaded_webcoos_clips" / "seabright"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "shoreline_outputs" / "seabright"
DEFAULT_SEABRIGHT_REFERENCE = (
    SAVE_PREDICT_DIR / "masks" / "walton_lighthouse-2025-05-13-231928Z.jpg"
)
DEFAULT_TWIN_LAKES_REFERENCE = (
    SAVE_PREDICT_DIR
    / "masks"
    / "walton_lighthouse-2025-05-18-172217Z_001325.png"
)
DEFAULT_SAM_CONFIG = REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_b+.yaml"
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def output_dir_for(video_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_parent = video_path.parent.relative_to(input_root)
    return output_root / relative_parent / video_path.stem


def has_existing_json(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.glob("*.json"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scan_clips(args: argparse.Namespace, videos: list[Path], input_root: Path):
    classifier = ReferenceViewClassifier(
        resolve_path(args.seabright_reference, DEFAULT_SEABRIGHT_REFERENCE),
        resolve_path(args.twin_lakes_reference, DEFAULT_TWIN_LAKES_REFERENCE),
        min_inliers=args.min_inliers,
        dominance_ratio=args.dominance_ratio,
    )
    plans: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for index, video_path in enumerate(videos, start=1):
        print(f"[scan {index}/{len(videos)}] {video_path.name}")
        samples, source_fps, frame_count = scan_video_views(
            video_path,
            classifier,
            sample_seconds=args.view_sample_seconds,
        )
        ranges = stable_target_ranges(
            samples,
            fps=source_fps,
            frame_count=frame_count,
            sample_seconds=args.view_sample_seconds,
            bridge_unknown_seconds=args.bridge_unknown_seconds,
            boundary_margin_seconds=args.boundary_margin_seconds,
            min_run_seconds=args.min_run_seconds,
        )
        relative_path = video_path.relative_to(input_root)
        label_counts = Counter(sample.label for sample in samples)
        included_seconds = sum(end - start + 1 for start, end in ranges) / source_fps
        range_text = format_frame_ranges(ranges)
        plan = {
            "video_path": str(video_path.resolve()),
            "relative_path": str(relative_path),
            "direction_group": relative_path.parts[0] if len(relative_path.parts) > 1 else "",
            "sea_state_bin": relative_path.parts[1] if len(relative_path.parts) > 2 else "",
            "source_fps": round(source_fps, 6),
            "frame_count": frame_count,
            "clip_duration_seconds": round(frame_count / source_fps, 2),
            "seabright_samples": label_counts.get("seabright", 0),
            "twin_lakes_samples": label_counts.get("twin_lakes", 0),
            "unknown_samples": label_counts.get("unknown", 0),
            "included_frame_ranges": range_text,
            "included_duration_seconds": round(included_seconds, 2),
            "processing_status": "ready" if ranges else "no_stable_seabright",
            "ranges": ranges,
        }
        plans.append(plan)
        for sample in samples:
            sample_rows.append(
                {
                    "relative_path": str(relative_path),
                    "frame_index": sample.frame_index,
                    "time_seconds": round(sample.time_seconds, 3),
                    "view_label": sample.label,
                    "seabright_inliers": sample.seabright_inliers,
                    "twin_lakes_inliers": sample.twin_lakes_inliers,
                }
            )
        print(
            f"    samples S/T/?={label_counts.get('seabright', 0)}/"
            f"{label_counts.get('twin_lakes', 0)}/{label_counts.get('unknown', 0)}; "
            f"included={included_seconds:.1f}s ranges={range_text or 'none'}"
        )
    return plans, sample_rows


def manifest_rows(plans: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in plan.items() if key != "ranges"} for plan in plans]


def run(args: argparse.Namespace) -> int:
    input_root = resolve_path(args.input_root, DEFAULT_INPUT_ROOT)
    output_root = resolve_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    sam_config = resolve_path(args.sam_config, DEFAULT_SAM_CONFIG)
    sam_checkpoint = resolve_path(args.sam_checkpoint, DEFAULT_SAM_CHECKPOINT)
    view_manifest = resolve_path(
        args.view_manifest,
        output_root / "seabright_view_manifest.csv",
    )
    sample_manifest = resolve_path(
        args.sample_manifest,
        output_root / "seabright_view_samples.csv",
    )

    videos = sorted(input_root.rglob("*.mp4"))
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print(f"No .mp4 files found under {input_root}")
        return 1

    print(f"Found {len(videos)} clip(s) under {input_root}")
    plans, sample_rows = scan_clips(args, videos, input_root)
    write_csv(view_manifest, manifest_rows(plans))
    write_csv(sample_manifest, sample_rows)
    print(f"Wrote view manifest: {view_manifest}")
    print(f"Wrote sample audit: {sample_manifest}")

    if args.scan_only:
        return 0

    failures: list[tuple[Path, int]] = []
    for index, plan in enumerate(plans, start=1):
        video_path = Path(str(plan["video_path"]))
        ranges = plan["ranges"]
        if not ranges:
            print(f"[{index}/{len(plans)}] skip no stable Seabright: {video_path.name}")
            continue

        save_dir = output_dir_for(video_path, input_root, output_root)
        if has_existing_json(save_dir) and not args.overwrite:
            plan["processing_status"] = "existing"
            print(f"[{index}/{len(plans)}] skip existing: {save_dir}")
            continue

        cmd = [
            sys.executable,
            "sam2_realtime.py",
            "--video-path",
            str(video_path),
            "--desired-fps",
            str(args.fps),
            "--include-frame-ranges",
            format_frame_ranges(ranges),
            "--save-shorelines",
            "--save-dir",
            str(save_dir),
            "--sam-config",
            str(sam_config),
            "--sam-checkpoint",
            str(sam_checkpoint),
            "--prompt-image",
            str(resolve_path(args.prompt_image, Path(args.prompt_image))),
            "--prompt-json",
            str(resolve_path(args.prompt_json, Path(args.prompt_json))),
            "--min-shoreline-contour-points",
            str(args.min_shoreline_contour_points),
            "--restart-interval",
            str(args.restart_interval),
            "--infer-width",
            str(args.infer_width),
            "--infer-height",
            str(args.infer_height),
            "--no-display",
        ]
        if not args.save_images:
            cmd.append("--json-only")
        if args.ignore_json:
            cmd.extend(["--ignore-json", str(resolve_path(args.ignore_json, Path(args.ignore_json)))])
        if args.ignore_blackout:
            cmd.append("--ignore-blackout")
        if args.sdp_math_only:
            cmd.append("--sdp-math-only")
        if args.perf_log:
            cmd.append("--perf-log")

        print(f"[{index}/{len(plans)}] processing: {video_path.name}")
        print(f"    Seabright ranges: {format_frame_ranges(ranges)}")
        print(f"    output: {save_dir}")
        completed = subprocess.run(cmd, cwd=SAVE_PREDICT_DIR)
        plan["processing_status"] = (
            "processed" if completed.returncode == 0 else f"failed_{completed.returncode}"
        )
        write_csv(view_manifest, manifest_rows(plans))
        if completed.returncode != 0:
            failures.append((video_path, completed.returncode))
            if args.stop_on_error:
                break

    write_csv(view_manifest, manifest_rows(plans))
    if failures:
        print("\nFailures:")
        for video_path, code in failures:
            print(f"  {code}: {video_path}")
        return 1

    print("Seabright batch processing complete.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None, help="Root containing Seabright MP4 clips.")
    parser.add_argument("--output-root", default=None, help="Root for mirrored shoreline outputs.")
    parser.add_argument("--fps", type=float, default=5.0, help="SAM2 frame processing rate.")
    parser.add_argument("--limit", type=int, default=None, help="Scan/process only the first N clips.")
    parser.add_argument("--scan-only", action="store_true", help="Write view audits without running SAM2.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess clips with existing JSONs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on the first SAM2 failure.")
    parser.add_argument("--view-sample-seconds", type=float, default=5.0)
    parser.add_argument(
        "--bridge-unknown-seconds",
        type=float,
        default=0.0,
        help="Bridge brief unknown gaps between Seabright samples; zero is safest.",
    )
    parser.add_argument("--boundary-margin-seconds", type=float, default=5.0)
    parser.add_argument("--min-run-seconds", type=float, default=20.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--dominance-ratio", type=float, default=1.25)
    parser.add_argument("--min-shoreline-contour-points", type=int, default=100)
    parser.add_argument(
        "--restart-interval",
        type=int,
        default=200,
        help="Processed frames between scheduled SAM2 rebuilds; zero disables them.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save full-resolution PNG frames; JSON-only is faster and is the default.",
    )
    parser.add_argument("--infer-width", type=int, default=960)
    parser.add_argument("--infer-height", type=int, default=720)
    parser.add_argument("--seabright-reference", default=None)
    parser.add_argument("--twin-lakes-reference", default=None)
    parser.add_argument("--view-manifest", default=None)
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--ignore-json", default=None)
    parser.add_argument(
        "--prompt-image",
        default=str(SAVE_PREDICT_DIR / "masks" / "walton_lighthouse-2025-05-13-231928Z.jpg"),
        help="Seabright prompt image passed to sam2_realtime.py.",
    )
    parser.add_argument(
        "--prompt-json",
        default=str(SAVE_PREDICT_DIR / "masks" / "walton_lighthouse-2025-05-13-231928Z.json"),
        help="Seabright prompt mask JSON passed to sam2_realtime.py.",
    )
    parser.add_argument("--ignore-blackout", action="store_true")
    parser.add_argument("--sdp-math-only", action="store_true")
    parser.add_argument("--perf-log", action="store_true")
    parser.add_argument("--sam-config", default=None)
    parser.add_argument("--sam-checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
