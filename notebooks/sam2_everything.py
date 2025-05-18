import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_object_tracker

# Config
MAX_OBJECTS = 10
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
DEVICE = 'cuda:0'
VIDEO_PATH = "./videos/camera_switch.mp4"

class Visualizer:
    def __init__(self, width, height):
        self.video_width = width
        self.video_height = height
        self.colors = np.random.randint(0, 255, (MAX_OBJECTS, 3), dtype=np.uint8)

    def overlay_masks(self, frame, masks):
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        for i in range(min(masks.shape[0], MAX_OBJECTS)):
            mask = (masks[i, 0] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, tuple(self.colors[i].tolist()), thickness=2)
        
        return frame

def generate_sparse_points(frame_shape, max_points=10):
    height, width = frame_shape[:2]
    points = []
    
    # Generate points in a grid pattern with proper spacing
    y_step = max(height // (int(np.sqrt(max_points)) + 1), 1)  # Fixed parenthesis
    x_step = max(width // (int(np.sqrt(max_points)) + 1), 1)   # Fixed parenthesis
    
    for y in range(0, height, y_step):
        for x in range(0, width, x_step):
            points.append([x, y])
            if len(points) >= max_points:
                return np.array(points)
    
    return np.array(points)

def main():
    video_stream = cv2.VideoCapture(VIDEO_PATH)
    if not video_stream.isOpened():
        print("Error: Unable to open video stream.")
        return

    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    visualizer = Visualizer(width, height)
    
    # Initialize SAM2 tracker
    sam = build_sam2_object_tracker(
        num_objects=MAX_OBJECTS,
        config_file=SAM_CONFIG_FILEPATH,
        ckpt_path=SAM_CHECKPOINT_FILEPATH,
        device=DEVICE,
        verbose=False
    )

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate exactly MAX_OBJECTS points
            points = generate_sparse_points(img.shape, max_points=MAX_OBJECTS)
            
            # Convert to normalized coordinates and proper shape
            norm_points = (points / np.array([img.shape[1], img.shape[0]])).reshape(-1, 1, 2)
            
            # Track objects
            sam_out = sam.track_new_object(
                img=img,
                points=norm_points
            )
            
            # Visualize
            frame_with_masks = visualizer.overlay_masks(frame, sam_out["pred_masks"])
            cv2.imshow("SAM2 Segment Everything", cv2.resize(frame_with_masks, (1280, 720)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()