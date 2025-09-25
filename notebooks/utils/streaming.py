import cv2
import threading
import time
import os
from datetime import datetime
from utils.config import VIDEO_PATH, RETRY_FRAMES

latest_frame = None
latest_frame_time = None
capture_running = True
lock = threading.Lock()
retry_counter = 0
max_retries = 5

def frame_capture():
    """Capture frames.

    Behavior changes:
      - If VIDEO_PATH is a local file (existing path on disk), loop back to start when end-of-file is reached.
      - If VIDEO_PATH looks like a network/stream (http/rtsp or non-existent file), on failure we wait and retry.
    """
    global latest_frame, latest_frame_time, capture_running, retry_counter
    source = VIDEO_PATH
    is_file = os.path.isfile(source)
    cap = cv2.VideoCapture(source)
    eof_failures = 0
    last_off_msg_ts = 0.0
    while capture_running:
        # If streaming (not a local file), respect operational hours to avoid reconnect spam
        if not is_file:
            now = datetime.now()
            if now.hour < 7 or now.hour >= 19:
                # Print at most once per minute
                now_ts = time.time()
                if now_ts - last_off_msg_ts > 60:
                    print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
                    last_off_msg_ts = now_ts
                # Ensure capture is released while off-hours
                if cap is not None and cap.isOpened():
                    cap.release()
                latest_frame_ts = time.time()
                with lock:
                    # Clear frame so consumers know there's nothing new
                    globals()['latest_frame'] = None
                    globals()['latest_frame_time'] = latest_frame_ts
                time.sleep(60)  # sleep and re-check
                continue

        # Ensure capture is open before reading (helps after off-hours resume)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source)
            time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
                latest_frame_time = time.time()
            retry_counter = 0
            eof_failures = 0
        else:
            # End of file behavior
            if is_file:
                eof_failures += 1
                # Try to loop: seek to frame 0
                pos_set = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not pos_set:
                    cap.release()
                    cap = cv2.VideoCapture(source)
                time.sleep(0.01)
                continue
            # Streaming source behavior
            latest_frame = None
            retry_counter += 1
            print(f"[Capture Thread] Stream frame read failed ({retry_counter}/{max_retries})")
            time.sleep(1.0)
            if retry_counter >= max_retries:
                print("[Capture Thread] Reinitializing stream source...")
                cap.release()
                cap = cv2.VideoCapture(source)
                retry_counter = 0
            continue
        time.sleep(0.005)

    cap.release()
