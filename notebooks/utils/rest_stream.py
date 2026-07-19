import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


_latest_frame = None
_latest_frame_time = None
_frame_counter = 0
_lock = threading.Lock()


def publish_frame(frame):
    global _latest_frame, _latest_frame_time, _frame_counter
    with _lock:
        _latest_frame = frame.copy()
        _latest_frame_time = time.time()
        _frame_counter += 1


def get_status():
    with _lock:
        has_frame = _latest_frame is not None
        frame_time = _latest_frame_time
        frame_counter = _frame_counter
    age_seconds = None if frame_time is None else max(0.0, time.time() - frame_time)
    return {
        "ok": has_frame,
        "has_frame": has_frame,
        "frame_counter": frame_counter,
        "frame_age_seconds": age_seconds,
    }


def _encode_latest_frame(jpeg_quality):
    with _lock:
        frame = None if _latest_frame is None else _latest_frame.copy()
    if frame is None:
        return None
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        return None
    return encoded.tobytes()


class StreamRequestHandler(BaseHTTPRequestHandler):
    jpeg_quality = 80
    stream_fps = 15
    cors_origin = "*"

    def log_message(self, format, *args):
        return

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(get_status())
            return
        if self.path == "/snapshot.jpg":
            self._send_snapshot()
            return
        if self.path == "/video":
            self._send_mjpeg_stream()
            return
        self.send_error(404, "Not found")

    def _send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_snapshot(self):
        jpeg = _encode_latest_frame(self.jpeg_quality)
        if jpeg is None:
            self.send_error(503, "No frame has been published yet")
            return
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _send_mjpeg_stream(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        delay = 1.0 / max(1, self.stream_fps)
        while True:
            jpeg = _encode_latest_frame(self.jpeg_quality)
            if jpeg is None:
                time.sleep(0.1)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(delay)


def start_stream_server(host="localhost", port=8000, jpeg_quality=80, stream_fps=15):
    handler = type(
        "ConfiguredStreamRequestHandler",
        (StreamRequestHandler,),
        {
            "jpeg_quality": jpeg_quality,
            "stream_fps": stream_fps,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Stream Server] Serving MJPEG at http://{host}:{port}/video")
    print(f"[Stream Server] Health check at http://{host}:{port}/health")
    return server
