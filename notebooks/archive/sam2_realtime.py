import os
import time
import cv2
import numpy as np
import torch
import json
from datetime import datetime
import threading
from sam2.build_sam import build_sam2_object_tracker
import torch.nn.functional as F  

# =====================
# Config
# =====================
NUM_OBJECTS = 1
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
# SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_small.pt"
# SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_s.yaml"
DEVICE = 'cuda:0'
VIDEO_PATH = "./videos/walton_lighthouse-2025-06-02-232026Z.mp4"
#VIDEO_PATH = "http://stage-ams-nfs.srv.axds.co/stream/adaptive/ucsc/walton_lighthouse/hls.m3u8"

# =====================
# Streaming Globals
# =====================
latest_frame = None
latest_frame_time = None
capture_running = True
lock = threading.Lock()
retry_counter = 0
max_retries = 5

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
# Frame Capture Thread
# =====================
def frame_capture():
    global latest_frame, latest_frame_time, cap, capture_running, retry_counter
    stream_url = VIDEO_PATH
    cap = cv2.VideoCapture(stream_url)
    # Adding debugging for video capture issues
    print("[Debug] Attempting to open video source...")
    if not cap.isOpened():
        print("[Error] Video source could not be opened. Check VIDEO_PATH.")
        return
    print("[Debug] Video source opened successfully.")
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

# Helper: Write LabelMe JSON with edge-filtered polygon
def write_labelme_json(image_path, coords, image_shape, label="shoreline", margin=10):
    from __main__ import latest_frame  # pull the current frame from global
    h, w = image_shape
    filtered_coords = [
        [float(x), float(y)]
        for x, y in coords
        if margin < x < (w - margin) and margin < y < (h - margin)
    ]

    shapes = [{
        "label":      label,
        "points":     filtered_coords,
        "group_id":   None,
        "shape_type": "polygon",
        "flags":      {}
    }]

    data = {
        "version":     "0.3.3",
        "flags":       {},
        "shapes":      shapes,
        "imagePath":   os.path.basename(image_path),
        "imageData":   None,
        "imageHeight": h,
        "imageWidth":  w,
        "text":        ""
    }

    image_save_path = image_path
    saved_image = cv2.resize(latest_frame, (1280, 960))
    cv2.imwrite(image_save_path, saved_image)  # Save current frame image
    json_path = os.path.splitext(image_save_path)[0] + ".json"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved shoreline JSON: {json_path}")


# =====================
# Visualization Class
# =====================
class Visualizer:
    def __init__(self, width, height):
        self.video_width = width
        self.video_height = height
        self.saved_frame_count = 0

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.video_height, self.video_width),
            mode="bilinear",
            align_corners=False
        )
        return mask

    def overlay_mask(
        self,
        frame,
        pred_masks,
        rock_mask=None,
        save_shoreline_coords=False,
        save_path=None,
        max_save_frames=0,
        frame_index=None  # optional, to help filename
    ):
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

            if save_shoreline_coords and save_path and self.saved_frame_count < max_save_frames:
                for cnt in contours:
                    if len(cnt) > 2:
                        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]
                        frame_name = f"shoreline_frame_{frame_index or self.saved_frame_count:04d}.png"
                        json_image_path = os.path.join(save_path, frame_name)
                        write_labelme_json(
                            json_image_path,
                            coords=coords,
                            image_shape=(self.video_height, self.video_width)
                        )
                        self.saved_frame_count += 1
                        break  # only save one valid contour per frame

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
    global capture_running

    print("Initializing SAM2...")
    sam = build_sam2_object_tracker(
        num_objects=NUM_OBJECTS,
        config_file=SAM_CONFIG_FILEPATH,
        ckpt_path=SAM_CHECKPOINT_FILEPATH,
        device=DEVICE,
        verbose=False
    )

    fine_tuned_weights_path = "./training_output/tuned_shoreline_decoder.pth"
    sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    print("Loaded fine-tuned mask decoder weights.")

    capture_thread = threading.Thread(target=frame_capture)
    capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0
    RETRY_FRAMES = 80

    prompt_img_site_a = cv2.imread("./masks/walton_lighthouse-2025-05-13-231928Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/walton_lighthouse-2025-05-13-231928Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    prompt_img_site_b = cv2.imread("./masks/walton_lighthouse-2025-05-13-233327Z.jpg")
    prompt_img_site_b = cv2.cvtColor(prompt_img_site_b, cv2.COLOR_BGR2RGB)
    mask_json_site_b = "./masks/walton_lighthouse-2025-05-13-233327Z.json"
    mask_site_b = json_to_mask(mask_json_site_b, prompt_img_site_a.shape)
    mask_site_b = np.expand_dims(np.expand_dims(mask_site_b.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/walton_lighthouse-2025-05-13-231928Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    desired_fps = 10
    frame_interval = 1.0 / desired_fps
    last_processed_time = 0
    frame_counter = 0
    RESTART_INTERVAL = 500  # frames

    visualizer = Visualizer(1280, 960)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        while True:
            now = datetime.now()
            # if now.hour < 7 or now.hour >= 19:
            #     print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
            #     time.sleep(300)
            #     continue

            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
                frame_time = latest_frame_time if latest_frame_time is not None else 0

            if frame is None:
                print("No frame available, skipping...")
                continue

            if frame_time - last_processed_time < frame_interval:
                continue
            last_processed_time = frame_time
            frame_counter += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_for_detection = cv2.GaussianBlur(img, (5, 5), 0)
            #img_for_detection = img
            H, W = img.shape[:2]
            start_y = int(H / 3)
            bbox = np.array([[[0, start_y], [W, H]]])

            if first_frame:
                print("First frame: initializing with mask prompt.")
                #sam_out = sam.track_new_object(img=img, box=bbox)
                current_img = prompt_img_site_b
                current_mask = mask_site_b
                sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                first_frame = False
            else:
                if not object_lost:
                    #sam_out = sam.track_all_objects(img=img_for_detection)
                    # Restart detection using mask prompt every RESTART_INTERVAL frames
                    if frame_counter % RESTART_INTERVAL == 0:
                        print(f"Frame {frame_counter}: Periodic reinitialization with mask prompt.")
                        torch.cuda.empty_cache()
                        sam = build_sam2_object_tracker(
                            num_objects=NUM_OBJECTS,
                            config_file=SAM_CONFIG_FILEPATH,
                            ckpt_path=SAM_CHECKPOINT_FILEPATH,
                            device=DEVICE,
                            verbose=False
                        )
                        sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                        sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                    else:
                        sam_out = sam.track_all_objects(img=img_for_detection)

                    if is_mask_lost(sam_out["pred_masks"]):
                        print("Object lost — starting recovery countdown.")
                        object_lost = True
                        frames_since_loss = 0
                else:
                    frames_since_loss += 1
                    print(f"Waiting... {frames_since_loss}/{RETRY_FRAMES} frames since loss.")

                    if frames_since_loss >= RETRY_FRAMES:
                        print("Reinitializing with mask prompt.")
                        torch.cuda.empty_cache()
                        sam = build_sam2_object_tracker(
                            num_objects=NUM_OBJECTS,
                            config_file=SAM_CONFIG_FILEPATH,
                            ckpt_path=SAM_CHECKPOINT_FILEPATH,
                            device=DEVICE,
                            verbose=False
                        )
                        sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                        print("Re-loaded fine-tuned weights after reinitialization.")
                        #sam_out = sam.track_new_object(img=img, box=bbox)
                        current_img = prompt_img_site_a
                        current_mask = mask_site_a
                        sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                        object_lost = False
                        frames_since_loss = 0
                    else:
                        sam_out = {
                            "pred_masks": torch.zeros((1, 1, img.shape[0], img.shape[1]),
                                dtype=torch.bfloat16, device=DEVICE)
                        }

            rock_mask_tensor = torch.from_numpy(rock_mask).float().to(DEVICE)
            pred_mask_shape = sam_out["pred_masks"].shape[-2:]
            rock_mask_resized = F.interpolate(
                rock_mask_tensor,
                size=pred_mask_shape,
                mode='bilinear',
                align_corners=False
            )
            if rock_mask_resized.shape[0] != sam_out["pred_masks"].shape[0]:
                rock_mask_resized = rock_mask_resized.expand_as(sam_out["pred_masks"])

            sam_out["pred_masks"] = torch.where(
                rock_mask_resized > 0.5,
                torch.ones_like(sam_out["pred_masks"]),
                sam_out["pred_masks"]
            )

            frame_with_mask = visualizer.overlay_mask(
                frame,
                sam_out["pred_masks"],
                rock_mask=None,
                save_shoreline_coords=False,
                save_path="./shoreline_jsons/test1",
                max_save_frames=300,
                frame_index=None  
            )

            frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            cv2.namedWindow('SAM2 Realtime Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SAM2 Realtime Tracking', 1280, 960)
            cv2.imshow("SAM2 Realtime Tracking", frame_with_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture_running = False
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
