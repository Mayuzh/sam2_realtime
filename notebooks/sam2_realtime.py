import os
import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_object_tracker

# =====================
# Config
# =====================
NUM_OBJECTS = 2
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"#
# SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_small.pt"
# SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_s.yaml"
DEVICE = 'cuda:0'
VIDEO_PATH = "./videos/seabright3.mp4"
#VIDEO_PATH = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"

# =====================
# Visualization Class
# =====================
class Visualizer:
    def __init__(self, width, height):
        self.video_width = width
        self.video_height = height

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False)
        return mask

    def overlay_mask(self, frame, mask):
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        mask = self.resize_mask(mask)
        mask = (mask > 0.0).numpy()
        for i in range(mask.shape[0]):
            #obj_mask = mask[i, 0, :, :]

            obj_mask = (mask[i, 0, :, :] * 255).astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw only the boundary
            cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)  # Green boundary

            #frame[obj_mask] = [255, 105, 180]  # Pink mask overlay
        return frame



# =====================
# Main Logic
# =====================
def main():
    video_stream = cv2.VideoCapture(VIDEO_PATH)
    if not video_stream.isOpened():
        print("Error: Unable to open video stream.")
        return

    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    visualizer = Visualizer(width, height)

    sam = build_sam2_object_tracker(
        num_objects=NUM_OBJECTS,
        config_file=SAM_CONFIG_FILEPATH,
        ckpt_path=SAM_CHECKPOINT_FILEPATH,
        device=DEVICE,
        verbose=False
    )

    first_frame = True

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        frame_idx = 0  # initialize frame counter
        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break
            # ✅ Skip every other frame
            frame_idx += 1
            # if frame_idx % 5 != 0:
            #     continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if first_frame:
                bbox = np.array([[[0, 580], [2560, 1920]]])  # Draw boundary for first frame

                sam_out = sam.track_new_object(img=img, box=bbox)

                for box in bbox:
                    top_left = tuple(box[0])
                    bottom_right = tuple(box[1])
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                first_frame = False
            else:
                sam_out = sam.track_all_objects(img=img)
            

            # Overlay segmentation mask
            frame_with_mask = visualizer.overlay_mask(frame, sam_out["pred_masks"])

            cv2.imshow("SAM2 Realtime Tracking", frame_with_mask)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()