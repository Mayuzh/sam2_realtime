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

from utils.config import *
from utils.helpers import is_mask_lost, json_to_mask, write_labelme_json
import utils.streaming as streaming
from utils.visualizer import Visualizer

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

    fine_tuned_weights_path = "./finetuned_weights/tuned_shoreline_decoder.pth"
    sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    print("Loaded fine-tuned mask decoder weights.")

    capture_thread = threading.Thread(target=streaming.frame_capture)
    capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0

    prompt_img_site_a = cv2.imread("./masks/walton_lighthouse-2024-11-16-194259Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/walton_lighthouse-2024-11-16-194259Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    prompt_img_site_b = cv2.imread("./masks/walton_lighthouse-2024-11-16-195057Z.jpg")
    prompt_img_site_b = cv2.cvtColor(prompt_img_site_b, cv2.COLOR_BGR2RGB)
    mask_json_site_b = "./masks/walton_lighthouse-2024-11-16-195057Z.json"
    mask_site_b = json_to_mask(mask_json_site_b, prompt_img_site_b.shape)
    mask_site_b = np.expand_dims(np.expand_dims(mask_site_b.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/walton_lighthouse-2025-05-13-231928Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    last_processed_time = 0
    frame_counter = 0

    #visualizer = Visualizer(1280, 960)
    visualizer = Visualizer(1440, 1080)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        while True:
            now = datetime.now()
            # if now.hour < 7 or now.hour >= 19:
            #     print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
            #     time.sleep(300)
            #     continue

            with streaming.lock:
                frame = streaming.latest_frame.copy() if streaming.latest_frame is not None else None
                frame_time = streaming.latest_frame_time if streaming.latest_frame_time is not None else 0

            if frame is None:
                print("No frame available, skipping...")
                continue
                #break

            # if frame_time - last_processed_time < FRAME_INTERVAL:
            #     continue
            last_processed_time = frame_time
            frame_counter += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_for_detection = cv2.GaussianBlur(img, (15, 15), 0)

            H, W = img.shape[:2]
            start_x = int(2 * W / 5)
            start_y = int(3 * H / 5)

            if first_frame:
                print("First frame: initializing with mask prompt.")
                current_img = prompt_img_site_b
                current_mask = mask_site_b
                sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                #point_coords = np.array([[start_x, start_y]])  # Example point coordinates
                #sam_out = sam.track_new_object(img=current_img, points=point_coords)  # Pass only point_coords
                first_frame = False
            else:
                if not object_lost:
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
                        current_img = prompt_img_site_b
                        current_mask = mask_site_b
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
                #img_for_detection,
                frame,
                sam_out["pred_masks"],
                rock_mask=None,
                save_shoreline_coords=False,
                save_path="./shoreline_jsons/twinlakes/13",
                max_save_frames=None,
                frame_index=None
            )

            #frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            frame_with_mask = cv2.resize(frame_with_mask, (1440, 1080))
            cv2.namedWindow('Santa Cruz', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Santa Cruz', 1440, 1080)
            #cv2.resizeWindow('Santa Cruz', 1280, 960)
            cv2.imshow("Santa Cruz", frame_with_mask)

            # Calculate and print the frame rate
            if frame_time > last_processed_time:
                frame_rate = 1 / (frame_time - last_processed_time)
                print(f"Frame Rate: {frame_rate:.2f} FPS")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                streaming.capture_running = False  # Stop the capture thread
                break

    streaming.capture_running = False
    capture_thread.join()
    print("detroying windows...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
