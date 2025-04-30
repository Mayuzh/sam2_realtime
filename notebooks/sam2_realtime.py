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
    def crop_contour_to_shoreline(self, contour, width):
        """
        Keep only the points between the leftmost-highest and rightmost-highest points.
        Assumes contour is ordered.
        """
        left_index = None
        right_index = None
        min_left_y = float('inf')
        min_right_y = float('inf')

        for i, pt in enumerate(contour[:, 0, :]):
            x, y = pt
            if x < 0.15 * width and y < min_left_y:
                min_left_y = y
                left_index = i
            if x > 0.85 * width and y < min_right_y:
                min_right_y = y
                right_index = i

        if left_index is None or right_index is None:
            return None  # can't crop without endpoints

        # Slice between indices (handle circular cases)
        if left_index < right_index:
            cropped = contour[left_index:right_index + 1]
        else:
            # wraparound case
            cropped = np.concatenate((contour[left_index:], contour[:right_index + 1]), axis=0)

        return cropped


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