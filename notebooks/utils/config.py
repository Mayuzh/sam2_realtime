from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]

NUM_OBJECTS = 1
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
DEVICE = 'cuda:0'
#VIDEO_PATH = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"
VIDEO_PATH = str(NOTEBOOKS_DIR / "videos" / "walton_lighthouse-2025-10-07-215711Z.mp4")
DESIRED_FPS = 30
FRAME_INTERVAL = 1.0 / DESIRED_FPS
RESTART_INTERVAL = 500
RETRY_FRAMES = 80
