import os
import time
import cv2
import numpy as np
import torch
import json
from datetime import datetime
import threading  # kept for historical reference
from sam2.build_sam import build_sam2_object_tracker
import torch.nn.functional as F
import argparse  # added for CLI flags

from utils.config import *
from utils.helpers import is_mask_lost, json_to_mask, write_labelme_json
import utils.streaming as streaming
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description='SAM2 shoreline tracking with optional ignore region blackout.')
    parser.add_argument('--ignore-json', default=None, help='Path to LabelMe JSON defining region to ignore (black out).')
    parser.add_argument('--ignore-blackout', action='store_true', help='Enable blacking out the ignore region (default off).')
    args = parser.parse_args()

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
    #sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    print("Loaded fine-tuned mask decoder weights.")

    # =====================
    # Legacy threaded capture (commented out after switching to file iteration)
    # =====================
    # capture_thread = threading.Thread(target=streaming.frame_capture)
    # capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0

    prompt_img_site_a = cv2.imread("./masks/jennette_north-2025-08-26-225322Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/jennette_north-2025-08-26-225322Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    prompt_img_site_b = cv2.imread("./masks/jennette_north-2025-08-26-225322Z.jpg")
    prompt_img_site_b = cv2.cvtColor(prompt_img_site_b, cv2.COLOR_BGR2RGB)
    mask_json_site_b = "./masks/jennette_north-2025-08-26-225322Z.json"
    mask_site_b = json_to_mask(mask_json_site_b, prompt_img_site_b.shape)
    mask_site_b = np.expand_dims(np.expand_dims(mask_site_b.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/walton_lighthouse-2024-11-16-194259Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    last_processed_time = 0  # kept for reference (not used to skip now)
    frame_counter = 0

    #visualizer = Visualizer(1440, 1080)
    visualizer = Visualizer(1280, 960)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        # Determine processing stride from desired vs actual FPS
        src_fps = streaming.get_video_fps()
        if DESIRED_FPS is None or (src_fps is not None and DESIRED_FPS >= src_fps):
            process_stride = 1
        elif src_fps is None:
            process_stride = 1  # unknown FPS, process all frames
        else:
            process_stride = max(1, int(round(src_fps / float(DESIRED_FPS))))

    # Inference resize target (kept consistent with Visualizer size)
    INFER_W, INFER_H = 1280, 960
    # Count frames actually processed (after skipping) for periodic reinit
    processed_frame_count = 0
    # =====================
    # New: iterate video frames sequentially (no skipping)
    # =====================
    for frame_idx, frame, frame_time in streaming.iter_video_frames():
            # Skip frames to match desired processing rate if needed
            if process_stride > 1 and (frame_idx % process_stride) != 0:
                continue
            processed_frame_count += 1
            frame_counter = frame_idx + 1
            #last_processed_time = frame_time

            frame_for_model = frame

            # Resize for inference to reduce compute
            if (frame_for_model.shape[1], frame_for_model.shape[0]) != (INFER_W, INFER_H):
                frame_for_model = cv2.resize(frame_for_model, (INFER_W, INFER_H))

            # ORIGINAL: img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(frame_for_model, cv2.COLOR_BGR2RGB)  # use possibly blacked-out, resized frame

            #img_for_detection = cv2.GaussianBlur(img, (3, 3), 0) 
            img_for_detection = img
            #img_for_detection = cv2.GaussianBlur(img, (151, 151), 0)

            H, W = img.shape[:2]
            start_x = int(1 * W / 3)
            start_y = int(4 * H / 5)
            point_coords = np.array([[start_x, start_y]])  # Example point coordinates
            if first_frame:
                print("First frame: initializing with mask prompt.")
                current_img = prompt_img_site_b
                current_mask = mask_site_b
                sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                #sam_out = sam.track_new_object(img=current_img, points=point_coords)  # Pass only point_coords
                first_frame = False
            else:
                if not object_lost:
                    # Reinitialize every RESTART_INTERVAL processed frames (not raw video frames)
                    if (isinstance(RESTART_INTERVAL, int) and RESTART_INTERVAL > 0 and (processed_frame_count % RESTART_INTERVAL == 0)):
                        print(f"Frame {frame_counter}: Periodic reinitialization with mask prompt.")
                        torch.cuda.empty_cache()
                        sam = build_sam2_object_tracker(
                            num_objects=NUM_OBJECTS,
                            config_file=SAM_CONFIG_FILEPATH,
                            ckpt_path=SAM_CHECKPOINT_FILEPATH,
                            device=DEVICE,
                            verbose=False
                        )
                        #sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                        sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                        #sam_out = sam.track_new_object(img=current_img, points=point_coords)  # Pass only point_coords

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
                        #sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                        print("Re-loaded fine-tuned weights after reinitialization.")
                        current_img = prompt_img_site_b
                        current_mask = mask_site_b
                        sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                        object_lost = False
                        frames_since_loss = 0
                    else:
                        sam_out = {
                            "pred_masks": torch.zeros((1, 1, img.shape[0], img.shape[1]), dtype=torch.bfloat16, device=DEVICE)
                        }

            # rock_mask_tensor = torch.from_numpy(rock_mask).float().to(DEVICE)
            # pred_mask_shape = sam_out["pred_masks"].shape[-2:]
            # rock_mask_resized = F.interpolate(
            #     rock_mask_tensor,
            #     size=pred_mask_shape,
            #     mode='bilinear',
            #     align_corners=False
            # )
            # if rock_mask_resized.shape[0] != sam_out["pred_masks"].shape[0]:
            #     rock_mask_resized = rock_mask_resized.expand_as(sam_out["pred_masks"])

            # sam_out["pred_masks"] = torch.where(
            #     rock_mask_resized > 0.5,
            #     torch.ones_like(sam_out["pred_masks"]),
            #     sam_out["pred_masks"]
            # )

            frame_with_mask = visualizer.overlay_mask(
                frame,
                sam_out["pred_masks"],
                rock_mask=None,
                save_shoreline_coords=True,
                save_path="./shoreline_jsons/jennette_north/active/13",
                max_save_frames=None,
                frame_index=frame_counter
            )

            frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            cv2.namedWindow('Santa Cruz', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Santa Cruz', 1280, 960)
            cv2.imshow("Santa Cruz", frame_with_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # streaming.capture_running = False  # legacy
    # capture_thread.join()  # legacy
    print("detroying windows...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()