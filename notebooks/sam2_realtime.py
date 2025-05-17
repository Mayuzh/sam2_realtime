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
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
# SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_small.pt"
# SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_s.yaml"
DEVICE = 'cuda:0'
VIDEO_PATH = "./videos/camera_switch.mp4"
#VIDEO_PATH = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"

# =====================
# Helpers
# =====================
def is_mask_lost(mask_tensor, threshold=0.001):
    """
    Check if the predicted mask is mostly empty.
    """
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.tensor(mask_tensor)
    active_ratio = (mask_tensor > 0).sum().item() / mask_tensor.numel()
    return active_ratio < threshold

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
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.video_height, self.video_width),
            mode="bilinear",
            align_corners=False
        )
        return mask

    def overlay_mask(self, frame, pred_masks, rock_mask=None):
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        # Resize prediction masks
        pred_masks = self.resize_mask(pred_masks)
        pred_masks = (pred_masks > 0.0).numpy()

        # Resize rock mask if provided
        if rock_mask is not None:
            rock_mask = self.resize_mask(rock_mask)
            rock_mask = (rock_mask > 0.0).numpy()

        for i in range(pred_masks.shape[0]):
            # Draw prediction mask boundaries (green)
            obj_mask = (pred_masks[i, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)  # Green boundary

        # Draw rock mask boundaries (red) if provided
        if rock_mask is not None:
            rock_mask = (rock_mask[0, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=2)  # Red boundary

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
    object_lost = False
    frames_since_loss = 0
    RETRY_FRAMES = 100  # For 5 fps video, 5 seconds = 25 frames
    bbox = np.array([[[0, 580], [2560, 1920]]])  # Initial prompt region

    prompt_img_site_a = cv2.imread("./masks/walton_lighthouse-2025-05-13-231928Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)

    mask_json_site_a = "./masks/walton_lighthouse-2025-05-13-231928Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)  # Convert to float32 and shape (1, 1, H, W)

    rock_mask_json = "./region/walton_lighthouse-2025-05-13-231928Z.json"
    rock_mask =  json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)  # Convert to float32 and shape (1, 1, H, W)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        frame_idx = 0  # initialize frame counter
        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break

            # Skip every other frame
            frame_idx += 1
            # if frame_idx % 5 != 0:
            #     continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if first_frame:
                print("First frame: initializing with bounding box prompt.")
                sam_out = sam.track_new_object(img=img, box=bbox)
                #sam_out = sam.track_new_object(img=prompt_img_site_a, mask=mask_site_a)

                # # Resize rock mask to match predicted mask shape
                # pred_shape = sam_out["pred_masks"].shape[-2:]  # (H, W)
                # rock_mask_resized = resize_rock_mask_for_prediction(rock_mask, pred_shape, device=DEVICE)

                # # Remove detections in rock region
                # sam_out["pred_masks"] = sam_out["pred_masks"].clone()
                # sam_out["pred_masks"][:, :, rock_mask_resized.bool()[0, 0]] = 0
                first_frame = False
            else:
                if not object_lost:
                    sam_out = sam.track_all_objects(img=img)

                    # # Resize rock mask to match predicted mask shape
                    # pred_shape = sam_out["pred_masks"].shape[-2:]  # (H, W)
                    # rock_mask_resized = resize_rock_mask_for_prediction(rock_mask, pred_shape, device=DEVICE)

                    # # Remove detections in rock region
                    # sam_out["pred_masks"] = sam_out["pred_masks"].clone()
                    # sam_out["pred_masks"][:, :, rock_mask_resized.bool()[0, 0]] = 0

                    # Check if mask is lost
                    if is_mask_lost(sam_out["pred_masks"]):
                        print("Object lost — starting recovery countdown.")
                        object_lost = True
                        frames_since_loss = 0
                else:
                    frames_since_loss += 1
                    print(f"Waiting... {frames_since_loss}/{RETRY_FRAMES} frames since loss.")
                    
                    if frames_since_loss >= RETRY_FRAMES:
                        print("Reinitializing with bounding box prompt.")
                        torch.cuda.empty_cache()
                        sam = build_sam2_object_tracker(
                            num_objects=NUM_OBJECTS,
                            config_file=SAM_CONFIG_FILEPATH,
                            ckpt_path=SAM_CHECKPOINT_FILEPATH,
                            device=DEVICE,
                            verbose=False
                        )
                        sam_out = sam.track_new_object(img=img, box=bbox)

                        # # Resize rock mask to match predicted mask shape
                        # pred_shape = sam_out["pred_masks"].shape[-2:]  # (H, W)
                        # rock_mask_resized = resize_rock_mask_for_prediction(rock_mask, pred_shape, device=DEVICE)

                        # # Remove detections in rock region
                        # sam_out["pred_masks"] = sam_out["pred_masks"].clone()
                        # sam_out["pred_masks"][:, :, rock_mask_resized.bool()[0, 0]] = 0

                        object_lost = False
                        frames_since_loss = 0
                    else:
                        # Keep placeholder mask during cooldown
                        sam_out = {
                            "pred_masks": torch.zeros((1, 1, img.shape[0], img.shape[1]),
                                                    dtype=torch.bfloat16, device=DEVICE)
            }

            # Overlay segmentation mask and rock mask
            frame_with_mask = visualizer.overlay_mask(
                frame, 
                sam_out["pred_masks"], 
                rock_mask=rock_mask  # Pass the rock mask here
            )
            frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            cv2.namedWindow('SAM2 Realtime Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SAM2 Realtime Tracking', 1280, 960)
            cv2.imshow("SAM2 Realtime Tracking", frame_with_mask)

            # Overlay segmentation mask
            # frame_with_mask = visualizer.overlay_mask(frame, sam_out["pred_masks"])
            # frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            # cv2.namedWindow('SAM2 Realtime Tracking', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('SAM2 Realtime Tracking', 1280, 960)
            # cv2.imshow("SAM2 Realtime Tracking", frame_with_mask)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()