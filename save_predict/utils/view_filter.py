"""Reference-image filtering for fixed-position coastal camera presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SEABRIGHT = "seabright"
TWIN_LAKES = "twin_lakes"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ViewSample:
    frame_index: int
    time_seconds: float
    label: str
    seabright_inliers: int
    twin_lakes_inliers: int


class ReferenceViewClassifier:
    """Classify a frame using geometric matches to two fixed camera presets."""

    def __init__(
        self,
        seabright_reference: str | Path,
        twin_lakes_reference: str | Path,
        *,
        min_inliers: int = 8,
        dominance_ratio: float = 1.25,
        max_width: int = 960,
    ) -> None:
        self.min_inliers = max(4, int(min_inliers))
        self.dominance_ratio = max(1.0, float(dominance_ratio))
        self.max_width = max(320, int(max_width))
        self.detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references = {
            SEABRIGHT: self._load_features(seabright_reference),
            TWIN_LAKES: self._load_features(twin_lakes_reference),
        }

    def _load_features(self, path: str | Path):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read view reference: {path}")
        keypoints, descriptors = self._features(image)
        if descriptors is None or len(keypoints) < self.min_inliers:
            raise ValueError(f"Too few usable features in view reference: {path}")
        return keypoints, descriptors

    def _features(self, image: np.ndarray):
        height, width = image.shape[:2]
        scale = min(1.0, self.max_width / float(width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_mask = np.full(gray.shape, 255, dtype=np.uint8)
        feature_mask[: round(gray.shape[0] * 0.08), :] = 0
        return self.detector.detectAndCompute(gray, feature_mask)

    def _inlier_count(self, frame_features, reference_label: str) -> int:
        frame_keypoints, frame_descriptors = frame_features
        reference_keypoints, reference_descriptors = self.references[reference_label]
        if frame_descriptors is None or len(frame_keypoints) < 4:
            return 0

        pairs = self.matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)
        good = [
            first
            for first, second in pairs
            if first.distance < 0.75 * second.distance
        ]
        if len(good) < 5:
            return 0

        reference_points = np.float32(
            [reference_keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        frame_points = np.float32(
            [frame_keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        _, inlier_mask = cv2.findHomography(
            reference_points,
            frame_points,
            cv2.RANSAC,
            5.0,
        )
        return int(inlier_mask.sum()) if inlier_mask is not None else 0

    def classify(self, frame: np.ndarray) -> tuple[str, int, int]:
        frame_features = self._features(frame)
        seabright_score = self._inlier_count(frame_features, SEABRIGHT)
        twin_lakes_score = self._inlier_count(frame_features, TWIN_LAKES)

        if (
            seabright_score >= self.min_inliers
            and seabright_score >= twin_lakes_score * self.dominance_ratio
        ):
            label = SEABRIGHT
        elif (
            twin_lakes_score >= self.min_inliers
            and twin_lakes_score >= seabright_score * self.dominance_ratio
        ):
            label = TWIN_LAKES
        else:
            label = UNKNOWN
        return label, seabright_score, twin_lakes_score


def scan_video_views(
    video_path: str | Path,
    classifier: ReferenceViewClassifier,
    *,
    sample_seconds: float = 5.0,
) -> tuple[list[ViewSample], float, int]:
    """Sample a video and classify camera view without decoding every frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS/frame count for video: {video_path}")

    stride = max(1, round(fps * max(0.25, sample_seconds)))
    frame_indexes = list(range(0, frame_count, stride))
    if frame_indexes[-1] != frame_count - 1:
        frame_indexes.append(frame_count - 1)

    samples: list[ViewSample] = []
    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        label, seabright_score, twin_lakes_score = classifier.classify(frame)
        samples.append(
            ViewSample(
                frame_index=frame_index,
                time_seconds=frame_index / fps,
                label=label,
                seabright_inliers=seabright_score,
                twin_lakes_inliers=twin_lakes_score,
            )
        )
    capture.release()
    return samples, fps, frame_count


def stable_target_ranges(
    samples: list[ViewSample],
    *,
    fps: float,
    frame_count: int,
    sample_seconds: float = 5.0,
    bridge_unknown_seconds: float = 0.0,
    boundary_margin_seconds: float = 5.0,
    min_run_seconds: float = 20.0,
) -> list[tuple[int, int]]:
    """Return inclusive raw-frame ranges containing only stable Seabright runs."""
    if not samples:
        return []

    labels = [sample.label for sample in samples]
    max_unknown_samples = max(0, int(bridge_unknown_seconds / sample_seconds))
    index = 0
    while index < len(labels):
        if labels[index] != UNKNOWN:
            index += 1
            continue
        gap_start = index
        while index < len(labels) and labels[index] == UNKNOWN:
            index += 1
        gap_end = index
        left_is_target = gap_start > 0 and labels[gap_start - 1] == SEABRIGHT
        right_is_target = gap_end < len(labels) and labels[gap_end] == SEABRIGHT
        if left_is_target and right_is_target and gap_end - gap_start <= max_unknown_samples:
            labels[gap_start:gap_end] = [SEABRIGHT] * (gap_end - gap_start)

    ranges: list[tuple[int, int]] = []
    index = 0
    margin_frames = round(max(0.0, boundary_margin_seconds) * fps)
    min_frames = round(max(0.0, min_run_seconds) * fps)
    while index < len(labels):
        if labels[index] != SEABRIGHT:
            index += 1
            continue
        run_start = index
        while index < len(labels) and labels[index] == SEABRIGHT:
            index += 1
        run_end = index - 1

        start_frame = samples[run_start].frame_index
        end_frame = samples[run_end].frame_index
        if run_start > 0:
            start_frame += margin_frames
        if run_end < len(samples) - 1:
            end_frame -= margin_frames
        else:
            end_frame = frame_count - 1

        start_frame = max(0, start_frame)
        end_frame = min(frame_count - 1, end_frame)
        if end_frame >= start_frame and end_frame - start_frame + 1 >= min_frames:
            ranges.append((start_frame, end_frame))
    return ranges


def format_frame_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{start}:{end}" for start, end in ranges)
