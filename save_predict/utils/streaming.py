import cv2
import os
import re
import threading  # kept for legacy commented code
import time
from datetime import datetime
from utils.config import VIDEO_PATH

# =====================
# Legacy threaded single-frame capture (commented out to prevent frame skipping with file inputs)
# =====================
# latest_frame = None
# latest_frame_time = None
# capture_running = True
# lock = threading.Lock()
# retry_counter = 0
# max_retries = 5
#
# def frame_capture():
#     global latest_frame, latest_frame_time, capture_running, retry_counter
#     stream_url = VIDEO_PATH
#     cap = cv2.VideoCapture(stream_url)
#     while capture_running:
#         now = datetime.now()
#         # if now.hour < 7 or now.hour >= 19:
#         #     time.sleep(300)
#         #     continue
#         ret, frame = cap.read()
#         if ret:
#             with lock:
#                 latest_frame = frame
#                 latest_frame_time = time.time()
#             retry_counter = 0
#         else:
#             latest_frame = None
#             break
#             retry_counter += 1
#             print(f"[Capture Thread] Frame read failed ({retry_counter}/{max_retries})")
#             time.sleep(1)
#             if retry_counter >= max_retries:
#                 print("[Capture Thread] Reinitializing stream...")
#                 cap.release()
#                 cap = cv2.VideoCapture(stream_url)
#                 retry_counter = 0
#         time.sleep(0.01)
#     cap.release()

# =====================
# New: Sequential frame generator for file-based videos (no skipping)
# =====================

def _natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def iter_video_frames(video_path: str = None):
    """Yield (frame_index, frame, timestamp_seconds) for every frame in the file.

    This is pull-based: a new frame is only decoded when the caller iterates,
    so no frames are lost even if processing is slow.
    """
    if video_path is None:
        video_path = VIDEO_PATH

    # If a directory is provided, read images as frames in natural-sorted order
    if os.path.isdir(video_path):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        names = [n for n in os.listdir(video_path) if os.path.splitext(n)[1].lower() in exts]
        names.sort(key=_natural_key)
        if not names:
            raise RuntimeError(f"No image frames found in directory: {video_path}")
        for idx, name in enumerate(names):
            fp = os.path.join(video_path, name)
            frame = cv2.imread(fp)
            if frame is None:
                continue
            ts = time.time()
            yield idx, frame, ts
        return

    # Otherwise, assume a video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time()  # wall-clock time when read (not PTS)
        yield idx, frame, ts
        idx += 1
    cap.release()

# Convenience function if code previously expected latest_frame semantics.
# Not recommended for new code; provided for transitional use only.

def read_all_frames(video_path: str = None):
    """Return list of frames (memory heavy for large videos)."""
    return [f for _, f, _ in iter_video_frames(video_path)]

def get_video_fps(video_path: str = None) -> float:
    """Return the FPS reported by the video container, or None if unavailable."""
    if video_path is None:
        video_path = VIDEO_PATH
    # For a directory of frames, FPS is unknown
    if os.path.isdir(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        cap.release()
    except Exception:
        pass
    if fps and fps > 0:
        return float(fps)
    return None
