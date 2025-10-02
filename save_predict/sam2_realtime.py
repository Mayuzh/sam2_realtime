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
import torch.backends.cuda as cuda_backends
from contextlib import nullcontext
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # new API replacing torch.backends.cuda.sdp_kernel
except Exception:  # Fallback if running on older PyTorch
    sdpa_kernel = None
    SDPBackend = None

from utils.config import *
from utils.helpers import is_mask_lost, json_to_mask, write_labelme_json
import utils.streaming as streaming
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description='SAM2 shoreline tracking with optional ignore region blackout.')
    parser.add_argument('--ignore-json', default=None, help='Path to LabelMe JSON defining region to ignore (black out).')
    parser.add_argument('--ignore-blackout', action='store_true', help='Enable blacking out the ignore region (default off).')
    parser.add_argument('--sdp-math-only', action='store_true', help='Force math attention backend to silence flash/memory-efficient warnings.')
    # New: saving shoreline frames/coords under a root per-clip subfolder
    parser.add_argument('--save-shorelines', action='store_true', help='Save shoreline frames and LabelMe JSONs for each frame.')
    parser.add_argument('--save-root', default='./shoreline_jsons/seabright_new/', help='Root folder under which a per-clip subfolder will be created.')
    parser.add_argument('--show-ignore-overlay', action='store_true', help='Draw a yellow translucent overlay for the ignore region (debug display only).')
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
    sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    print("Loaded fine-tuned mask decoder weights.")

    # =====================
    # Legacy threaded capture (commented out after switching to file iteration)
    # =====================
    # capture_thread = threading.Thread(target=streaming.frame_capture)
    # capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0

    prompt_img_site_a = cv2.imread("./masks/walton_lighthouse-2024-01-11-184016Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/walton_lighthouse-2024-01-11-184016Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    prompt_img_site_b = cv2.imread("./masks/walton_lighthouse-2024-01-11-184016Z.jpg")
    prompt_img_site_b = cv2.cvtColor(prompt_img_site_b, cv2.COLOR_BGR2RGB)
    mask_json_site_b = "./masks/walton_lighthouse-2024-01-11-184016Z.json"
    mask_site_b = json_to_mask(mask_json_site_b, prompt_img_site_b.shape)
    mask_site_b = np.expand_dims(np.expand_dims(mask_site_b.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/walton_lighthouse-2024-01-11-214420Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    last_processed_time = 0  # kept for reference (not used to skip now)
    frame_counter = 0

    #visualizer = Visualizer(1440, 1080)
    visualizer = Visualizer(1280, 960)

    # Prepare ignore-region handling
    # Fallback: if --ignore-blackout is set but --ignore-json is missing,
    # use the rock_mask_json path (if available) as the ignore region.
    try:
        fallback_ignore_json = rock_mask_json
    except NameError:
        fallback_ignore_json = None
    ignore_json_path = args.ignore_json or fallback_ignore_json
    use_ignore_blackout = bool(args.ignore_blackout and ignore_json_path)
    if args.ignore_blackout and args.ignore_json is None and ignore_json_path is not None:
        print(f"[ignore] Using rock_mask_json as ignore region: {ignore_json_path}")
    ignore_mask_full = None  # cached per input frame size
    ignore_contours = None   # cached contours for drawing

    # Build per-clip save directory under the provided root when saving is enabled
    if args.save_shorelines:
        try:
            # VIDEO_PATH is imported from utils.config via wildcard import
            vp = VIDEO_PATH
        except Exception:
            vp = None
        if vp and os.path.isdir(vp):
            clip_name = os.path.basename(os.path.normpath(vp))
        elif vp:
            clip_name = os.path.splitext(os.path.basename(vp))[0]
        else:
            # Fallback if VIDEO_PATH unavailable
            clip_name = 'clip'
        per_clip_save_dir = os.path.join(args.save_root, clip_name)
        os.makedirs(per_clip_save_dir, exist_ok=True)
    else:
        per_clip_save_dir = None

    # Optionally force math-only attention to avoid backend warnings (new PyTorch API)
    use_math_only_attn = bool(args.sdp_math_only and torch.cuda.is_available() and sdpa_kernel is not None and SDPBackend is not None)

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

            # If requested, blackout the ignore region before running the model
            if use_ignore_blackout and ignore_json_path:
                # Build/cached mask at the source frame resolution, then resize to inference size
                if (ignore_mask_full is None) or (ignore_mask_full.shape[:2] != frame.shape[:2]):
                    try:
                        ignore_mask_full = json_to_mask(ignore_json_path, frame.shape).astype(np.uint8)
                        # Precompute contours once for drawing (at source frame size)
                        try:
                            binary = (ignore_mask_full > 0).astype(np.uint8) * 255
                            cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # OpenCV 3/4 API compatibility
                            ignore_contours = cnts[0] if len(cnts) == 2 else cnts[1]
                        except Exception:
                            ignore_contours = None
                    except Exception as e:
                        print(f"Warning: Failed to load ignore JSON '{ignore_json_path}': {e}")
                        ignore_mask_full = None
                        ignore_contours = None
                if ignore_mask_full is not None:
                    ignore_mask_rs = cv2.resize(ignore_mask_full, (INFER_W, INFER_H), interpolation=cv2.INTER_NEAREST)
                    # Blackout pixels in the ignore region (set to 0)
                    frame_for_model[ignore_mask_rs > 0] = 0

            # ORIGINAL: img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(frame_for_model, cv2.COLOR_BGR2RGB)  # use possibly blacked-out, resized frame

            #img_for_detection = cv2.GaussianBlur(img, (3, 3), 0) 
            img_for_detection = cv2.GaussianBlur(img, (105, 105), 0)

            H, W = img.shape[:2]
            start_x = int(1 * W / 4)
            start_y = int(H)
            point_coords = np.array([[start_x, start_y]])  # Example point coordinates
            with torch.inference_mode():
                attn_cm = sdpa_kernel(SDPBackend.MATH) if use_math_only_attn else nullcontext()
                with attn_cm:
                    with torch.autocast(device_type=("cuda" if str(DEVICE).startswith("cuda") else "cpu"), dtype=torch.bfloat16):
                        if first_frame:
                            print("First frame: initializing with mask prompt.")
                            current_img = prompt_img_site_a
                            current_mask = mask_site_a
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
                                    sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                                    sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                                    #sam_out = sam.track_new_object(img=current_img, points=point_coords)  

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
                                    current_img = prompt_img_site_a
                                    current_mask = mask_site_a
                                    sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                                    #sam_out = sam.track_new_object(img=current_img, points=point_coords)  
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
                save_path=per_clip_save_dir,
                max_save_frames=None,
                frame_index=frame_counter,
                video_filename=VIDEO_PATH if 'VIDEO_PATH' in globals() else None,
            )

            # Draw the ignored region on the displayed frame for verification (outline + translucent fill)
            if use_ignore_blackout and ignore_contours is not None and args.show_ignore_overlay:
                try:
                    overlay = frame_with_mask.copy()
                    fill_color = (0, 255, 255)   # yellow fill
                    edge_color = (0, 200, 255)   # orange-yellow edge
                    # Filled region
                    cv2.drawContours(overlay, ignore_contours, -1, fill_color, thickness=cv2.FILLED)
                    # Alpha blend
                    frame_with_mask = cv2.addWeighted(overlay, 0.25, frame_with_mask, 0.75, 0)
                    # Edge outline on top
                    cv2.drawContours(frame_with_mask, ignore_contours, -1, edge_color, thickness=2)
                except Exception as e:
                    # Non-fatal: continue without overlay
                    print(f"[ignore-draw] Warning: failed to draw ignore overlay: {e}")

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