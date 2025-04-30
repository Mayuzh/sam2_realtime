import os
import time
import cv2
import numpy as np
import torch
import json
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
VIDEO_PATH = "./videos/seabright2.mp4"
#VIDEO_PATH = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"
MASK_JSON_PATH = "./masks/frame_1745908973630.json"  # <-- path to your labeled mask

# =====================
# Helpers
# =====================
def json_to_mask(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask

def load_region_mask(json_path, image_shape):
    """
    Load polygon points from LabelMe-style JSON and convert to a binary mask.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask

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

    def split_contour_into_segments(self, contour, region_mask, distance_threshold=10):
        """
        Keep only points in the region, and split into segments to avoid long connecting lines.
        """
        filtered_points = []
        segments = []

        for pt in contour[:, 0, :]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < region_mask.shape[0] and 0 <= x < region_mask.shape[1] and region_mask[y, x]:
                filtered_points.append((x, y))
            else:
                # If we hit a point outside region and we have a current segment, finalize it
                if len(filtered_points) >= 2:
                    segments.append(np.array(filtered_points, dtype=np.int32).reshape(-1, 1, 2))
                    filtered_points = []

        # Final segment
        if len(filtered_points) >= 2:
            segments.append(np.array(filtered_points, dtype=np.int32).reshape(-1, 1, 2))

        return segments

    def overlay_mask(self, frame, mask):
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        mask = self.resize_mask(mask)
        mask = (mask > 0.0).numpy()
        #print("frame shape:", frame.shape)
        # Load region mask (once per frame, or cache it)
        region_mask = load_region_mask("./region/frame_1745908973630.json", frame.shape)


        for i in range(mask.shape[0]):
            #obj_mask = mask[i, 0, :, :]

            obj_mask = (mask[i, 0, :, :] * 255).astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw only the boundary
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)  # Green boundary
            for contour in contours:
                segments = self.split_contour_into_segments(contour, region_mask)
                for segment in segments:
                    cv2.polylines(frame, [segment], isClosed=False, color=(0, 255, 0), thickness=2)


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

            # if first_frame:
            #     bbox = np.array([[[0, 580], [2560, 1920]]])  # Draw boundary for first frame

            #     sam_out = sam.track_new_object(img=img, box=bbox)

            #     for box in bbox:
            #         top_left = tuple(box[0])
            #         bottom_right = tuple(box[1])
            #         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            #     first_frame = False
            if first_frame:
                binary_mask = json_to_mask(MASK_JSON_PATH, (img.shape[0], img.shape[1]))  # (H, W)

                # Convert to float32 and shape (1, 1, H, W)
                binary_mask = np.expand_dims(np.expand_dims(binary_mask.astype(np.float32), axis=0), axis=0)

                print("binary_mask.shape =", binary_mask.shape)  # Confirm (1, 1, H, W)
                print("binary_mask dtype =", binary_mask.dtype)  # Confirm float32

                sam_out = sam.track_new_object(img=img, mask=binary_mask)
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