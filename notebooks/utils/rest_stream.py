import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2


_latest_frame = None
_latest_frame_time = None
_frame_counter = 0
_lock = threading.Lock()
_hls_encoder = None


def publish_frame(frame):
    global _latest_frame, _latest_frame_time, _frame_counter
    with _lock:
        _latest_frame = frame.copy()
        _latest_frame_time = time.time()
        _frame_counter += 1
    if _hls_encoder is not None:
        _hls_encoder.publish(frame)


def _find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise RuntimeError(
            "HLS output requires FFmpeg. Install FFmpeg and put it on PATH, "
            "or install the Python fallback with: pip install imageio-ffmpeg"
        )


class HlsEncoder:
    """Encode the latest processed frame into a short, rolling HLS stream."""

    def __init__(self, output_dir, fps):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.ffmpeg = _find_ffmpeg()
        self._frame = None
        self._frame_lock = threading.Lock()
        self._process = None
        self._thread = None
        self._stopped = threading.Event()

        # Only remove files owned by this stream's dedicated output directory.
        for pattern in ("stream.m3u8", "segment_*.ts", "*.tmp"):
            for path in self.output_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    def publish(self, frame):
        with self._frame_lock:
            self._frame = frame.copy()
            should_start = self._thread is None
        if should_start:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def _run(self):
        with self._frame_lock:
            first_frame = self._frame.copy()
        height, width = first_frame.shape[:2]
        playlist = self.output_dir / "stream.m3u8"
        segment_pattern = self.output_dir / "segment_%06d.ts"
        keyframe_interval = self.fps

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-g", str(keyframe_interval),
            "-keyint_min", str(keyframe_interval),
            "-sc_threshold", "0",
            "-f", "hls",
            "-hls_time", "1",
            "-hls_list_size", "6",
            "-hls_delete_threshold", "2",
            "-hls_flags", "delete_segments+omit_endlist+independent_segments+program_date_time+temp_file",
            "-hls_segment_filename", str(segment_pattern),
            str(playlist),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            creationflags=creationflags,
        )
        print(f"[HLS Encoder] Writing {self.fps} FPS H.264 stream to {playlist}")

        deadline = time.monotonic()
        try:
            while not self._stopped.is_set() and self._process.poll() is None:
                deadline += 1.0 / self.fps
                with self._frame_lock:
                    frame = self._frame.copy()
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                self._process.stdin.write(frame.tobytes())
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._stopped.wait(remaining)
                else:
                    deadline = time.monotonic()
        except (BrokenPipeError, OSError) as error:
            if not self._stopped.is_set():
                print(f"[HLS Encoder] FFmpeg stopped unexpectedly: {error}")
        finally:
            if self._process.stdin:
                self._process.stdin.close()


def get_status():
    with _lock:
        has_frame = _latest_frame is not None
        frame_time = _latest_frame_time
        frame_counter = _frame_counter
    age_seconds = None if frame_time is None else max(0.0, time.time() - frame_time)
    hls_ready = bool(
        _hls_encoder is not None
        and (_hls_encoder.output_dir / "stream.m3u8").is_file()
        and _hls_encoder._process is not None
        and _hls_encoder._process.poll() is None
    )
    return {
        "ok": has_frame,
        "has_frame": has_frame,
        "frame_counter": frame_counter,
        "frame_age_seconds": age_seconds,
        "hls_ready": hls_ready,
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
        path = urlparse(self.path).path
        if path in ("/", "/health"):
            self._send_json(get_status())
            return
        if path == "/snapshot.jpg":
            self._send_snapshot()
            return
        if path == "/video":
            self._send_mjpeg_stream()
            return
        if path.startswith("/hls/"):
            self._send_hls_file(path)
            return
        self.send_error(404, "Not found")

    def _send_hls_file(self, request_path):
        if _hls_encoder is None:
            self.send_error(503, "HLS encoder is not running")
            return
        filename = request_path.removeprefix("/hls/")
        if filename != "stream.m3u8" and not (
            filename.startswith("segment_") and filename.endswith(".ts")
        ):
            self.send_error(404, "Not found")
            return
        path = _hls_encoder.output_dir / filename
        if not path.is_file():
            self.send_error(503, "Stream is starting")
            return
        data = path.read_bytes()
        self.send_response(200)
        self._send_cors_headers()
        if filename.endswith(".m3u8"):
            self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Content-Type", "video/mp2t")
            self.send_header("Cache-Control", "public, max-age=30, immutable")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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


class StreamHTTPServer(ThreadingHTTPServer):
    def shutdown(self):
        super().shutdown()
        if _hls_encoder is not None:
            _hls_encoder.stop()


def start_stream_server(host="localhost", port=8000, jpeg_quality=80, stream_fps=15):
    global _hls_encoder
    hls_dir = Path(__file__).resolve().parents[1] / "temp" / f"hls_{port}"
    _hls_encoder = HlsEncoder(hls_dir, stream_fps)
    handler = type(
        "ConfiguredStreamRequestHandler",
        (StreamRequestHandler,),
        {
            "jpeg_quality": jpeg_quality,
            "stream_fps": stream_fps,
        },
    )
    server = StreamHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Stream Server] Serving MJPEG at http://{host}:{port}/video")
    print(f"[Stream Server] Serving HLS at http://{host}:{port}/hls/stream.m3u8")
    print(f"[Stream Server] Health check at http://{host}:{port}/health")
    return server
