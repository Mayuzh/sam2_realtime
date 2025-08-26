import cv2
import threading
import time
from datetime import datetime
from utils.config2 import VIDEO_PATH

latest_frame = None
latest_frame_time = None
capture_running = True
lock = threading.Lock()
retry_counter = 0
max_retries = 5

def frame_capture():
    global latest_frame, latest_frame_time, capture_running, retry_counter
    stream_url = VIDEO_PATH
    cap = cv2.VideoCapture(stream_url)
    while capture_running:
        now = datetime.now()
        # if now.hour < 7 or now.hour >= 19:
        #     time.sleep(300)
        #     continue
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
                latest_frame_time = time.time()
            retry_counter = 0
        else:
            retry_counter += 1
            print(f"[Capture Thread] Frame read failed ({retry_counter}/{max_retries})")
            time.sleep(1)
            if retry_counter >= max_retries:
                print("[Capture Thread] Reinitializing stream...")
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                retry_counter = 0
        time.sleep(0.01)
    cap.release()
