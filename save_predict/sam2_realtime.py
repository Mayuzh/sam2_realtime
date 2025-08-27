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
    # sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    print("Loaded fine-tuned mask decoder weights.")

    # =====================
    # Legacy threaded capture (commented out after switching to file iteration)
    # =====================
    # capture_thread = threading.Thread(target=streaming.frame_capture)
    # capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0

    prompt_img_site_a = cv2.imread("./masks/jennette_north-2024-08-27-220509Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/jennette_north-2024-08-27-220509Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    prompt_img_site_b = cv2.imread("./masks/jennette_north-2024-08-27-220509Z.jpg")
    prompt_img_site_b = cv2.cvtColor(prompt_img_site_b, cv2.COLOR_BGR2RGB)
    mask_json_site_b = "./masks/jennette_north-2024-08-27-220509Z.json"
    mask_site_b = json_to_mask(mask_json_site_b, prompt_img_site_b.shape)
    mask_site_b = np.expand_dims(np.expand_dims(mask_site_b.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/walton_lighthouse-2024-11-16-194259Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    last_processed_time = 0  # kept for reference (not used to skip now)
    frame_counter = 0

    #visualizer = Visualizer(1440, 1080)
    visualizer = Visualizer(1280, 960)

    ignore_mask = None  # will load lazily to match actual frame size

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        # Determine save stride based on desired FPS vs source FPS
        src_fps = streaming.get_video_fps()
        if DESIRED_FPS is None or (src_fps is not None and DESIRED_FPS >= src_fps):
            save_stride = 1
        elif src_fps is None:
            # Unknown FPS; approximate by time using FRAME_INTERVAL
            save_stride = None  # use time-based gating
            last_saved_ts = None
        else:
            save_stride = max(1, int(round(src_fps / float(DESIRED_FPS))))
        # =====================
        # New: iterate video frames sequentially (no skipping)
        # =====================
        for frame_idx, frame, frame_time in streaming.iter_video_frames():
            frame_counter = frame_idx + 1
            last_processed_time = frame_time

            # Lazy load ignore mask with frame dimensions (H,W only)
            if ignore_mask is None and args.ignore_json and os.path.isfile(args.ignore_json):
                try:
                    ignore_mask_arr = json_to_mask(args.ignore_json, frame.shape[:2])  # FIX: pass only (H,W)
                    ignore_mask = (ignore_mask_arr > 0).astype(np.uint8)
                    print(f"Loaded ignore region mask from {args.ignore_json}")
                except Exception as e:
                    print(f"Failed to load ignore mask {args.ignore_json}: {e}")
                    ignore_mask = None

            if ignore_mask is not None and args.ignore_blackout:
                frame_for_model = frame.copy()
                frame_for_model[ignore_mask == 1] = 0  # black out region (FIX applied before cvtColor)
            else:
                frame_for_model = frame

            # ORIGINAL: img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(frame_for_model, cv2.COLOR_BGR2RGB)  # FIX: use possibly blacked-out frame

            img_for_detection = cv2.GaussianBlur(img, (3, 3), 0) 
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

            # Optional: enforce ignore region on predicted mask even if not blacked out (hard suppression)
            # ORIGINAL:
            # if ignore_mask is not None:
            #     ignore_t = torch.from_numpy(1 - ignore_mask).to(sam_out["pred_masks"].dtype).to(DEVICE).unsqueeze(0).unsqueeze(0)  # FIX dtype & shape
            #     sam_out["pred_masks"] = sam_out["pred_masks"] * ignore_t
            if ignore_mask is not None:
                pred_h, pred_w = sam_out["pred_masks"].shape[-2:]
                # Resize ignore_mask (frame size) to prediction size if needed
                if ignore_mask.shape != (pred_h, pred_w):
                    # Use nearest-neighbor to preserve binary nature
                    ignore_resized = cv2.resize(ignore_mask, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
                else:
                    ignore_resized = ignore_mask
                ignore_t = torch.from_numpy(1 - ignore_resized).to(sam_out["pred_masks"].dtype).to(DEVICE).unsqueeze(0).unsqueeze(0)
                sam_out["pred_masks"] = sam_out["pred_masks"] * ignore_t

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

            # sam_out["pred_masks"] = torch.where(
            #     rock_mask_resized > 0.5,
            #     torch.ones_like(sam_out["pred_masks"]),
            #     sam_out["pred_masks"]
            # )

            # Decide whether to save this frame based on DESIRED_FPS
            if 'save_stride' in locals() and save_stride is not None:
                should_save = (frame_idx % save_stride == 0)
            else:
                # time-based gating when FPS unknown
                if 'last_saved_ts' not in locals():
                    last_saved_ts = None
                if last_saved_ts is None or (frame_time - last_saved_ts) >= FRAME_INTERVAL:
                    should_save = True
                    last_saved_ts = frame_time
                else:
                    should_save = False

            # Render overlay and (optionally) save image/JSON when should_save
            frame_with_mask = visualizer.overlay_mask(
                frame,
                sam_out["pred_masks"],
                rock_mask=None,
                save_shoreline_coords=should_save,
                save_path="./shoreline_jsons/jennette_north/calm/11/" if should_save else None,
                max_save_frames=None,
                frame_index=frame_counter,
                video_filename=VIDEO_PATH,
                save_even_if_empty=True,
            )

            if should_save:
                print(f"Saved frame {frame_counter} (every {save_stride if ('save_stride' in locals() and save_stride) else 'time-gated'} frame)")

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
