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
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

from utils.config2 import *
from utils.helpers import is_mask_lost, json_to_mask, write_labelme_json
import utils.streaming2 as streaming
from utils.visualizer import Visualizer

# Global frame sharing for web server
latest_processed_frame = None
frame_lock = threading.Lock()

# FastAPI app
app = FastAPI(title="Jennette North Stream")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running", "camera": "jennette_north", "port": 8001}

@app.get("/stream")
async def stream():
    async def generate():
        while True:
            try:
                with frame_lock:
                    frame = latest_processed_frame
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                await asyncio.sleep(0.1)  # ~10 FPS
            except Exception as e:
                print(f"Stream error: {e}")
                await asyncio.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info", access_log=True)
    server = uvicorn.Server(config)
    import asyncio
    asyncio.run(server.serve())

def overlay_mask_with_invisible_contour(frame, mask):
    """
    Makes contour lines within the mask area transparent by blending with background.
    """
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a contour mask (1px lines)
    contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

    # Ensure `contour_mask` matches the size and type of `frame`
    contour_mask = cv2.resize(contour_mask, (frame.shape[1], frame.shape[0]))
    if len(contour_mask.shape) == 2:
        contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)

    # Create a mask where we want to keep original pixels (not part of contours inside mask)
    keep_mask = cv2.bitwise_or(
        cv2.bitwise_not(contour_mask[:, :, 0]),  # Not part of any contour
        cv2.bitwise_and(contour_mask[:, :, 0], cv2.bitwise_not(binary_mask))  # Or contours outside main mask
    )

    # Create an image with only the contours we want to keep
    kept_contours = cv2.bitwise_and(frame, frame, mask=keep_mask)

    # Create background (original image without any contours)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(contour_mask[:, :, 0]))
    
    # Combine
    result = cv2.add(background, kept_contours)
    
    return result

def main():
    global capture_running, latest_processed_frame

    # Start web server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    print("Server start at http://localhost:8001/stream")

    print("Initializing SAM2...")
    sam = build_sam2_object_tracker(
        num_objects=NUM_OBJECTS,
        config_file=SAM_CONFIG_FILEPATH,
        ckpt_path=SAM_CHECKPOINT_FILEPATH,
        device=DEVICE,
        verbose=False
    )

    # Commenting out loading of fine-tuned weights to use the original SAM2 model
    # fine_tuned_weights_path = "./finetuned_weights/tuned_shoreline_decoder.pth"
    # sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
    # print("Loaded fine-tuned mask decoder weights.")

    capture_thread = threading.Thread(target=streaming.frame_capture)
    capture_thread.start()

    first_frame = True
    object_lost = False
    frames_since_loss = 0

    prompt_img_site_a = cv2.imread("./masks/jennette_north-2025-10-09-130944Z.jpg")
    prompt_img_site_a = cv2.cvtColor(prompt_img_site_a, cv2.COLOR_BGR2RGB)
    mask_json_site_a = "./masks/jennette_north-2025-10-09-130944Z.json"
    mask_site_a = json_to_mask(mask_json_site_a, prompt_img_site_a.shape)
    mask_site_a = np.expand_dims(np.expand_dims(mask_site_a.astype(np.float32), axis=0), axis=0)

    rock_mask_json = "./region/jennette_north-2025-07-21-235641Z.json"
    rock_mask = json_to_mask(rock_mask_json, prompt_img_site_a.shape)
    rock_mask = np.expand_dims(np.expand_dims(rock_mask.astype(np.float32), axis=0), axis=0)

    last_processed_time = 0
    frame_counter = 0

    visualizer = Visualizer(1280, 960)

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        while True:
            now = datetime.now()
            if now.hour < 7 or now.hour >= 19:
                print("STREAM OFF: Outside operational hours (7 AM to 7 PM).")
                time.sleep(300)
                continue

            with streaming.lock:
                frame = streaming.latest_frame.copy() if streaming.latest_frame is not None else None
                frame_time = streaming.latest_frame_time if streaming.latest_frame_time is not None else 0

            if frame is None:
                print("No frame available, skipping...")
                continue

            if frame_time - last_processed_time < FRAME_INTERVAL:
                continue
            last_processed_time = frame_time
            frame_counter += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #img_for_detection = cv2.GaussianBlur(img, (7, 7), 0)
            img_for_detection = img
            H, W = img.shape[:2]
            start_y = int(H / 3)
            start_x = int(3 * W / 4)
            #bbox = np.array([[[start_x, 0], [W, H]]])

            if first_frame:
                print("First frame: initializing with prompt points.")
                current_img = prompt_img_site_a
                #sam_out = sam.track_new_object(img=current_img, mask=current_mask)
                point_coords = np.array([[start_x, start_y]])  # Example point coordinates
                sam_out = sam.track_new_object(img=current_img, points=point_coords)  # Pass only point_coords

                # Draw the prompt points directly on the output stream
                for point in point_coords:
                    cv2.circle(frame, tuple(point), radius=5, color=(0, 255, 0), thickness=-1)
                first_frame = False
            else:
                if not object_lost:
                    if frame_counter % RESTART_INTERVAL == 0:
                        print(f"Frame {frame_counter}: Periodic reinitialization with prompt points.")
                        torch.cuda.empty_cache()
                        sam = build_sam2_object_tracker(
                            num_objects=NUM_OBJECTS,
                            config_file=SAM_CONFIG_FILEPATH,
                            ckpt_path=SAM_CHECKPOINT_FILEPATH,
                            device=DEVICE,
                            verbose=False
                        )
                        sam_out = sam.track_new_object(img=current_img, points=point_coords)

                        # Draw the prompt points directly on the output stream
                        for point in point_coords:
                            cv2.circle(frame, tuple(point), radius=5, color=(0, 255, 0), thickness=-1)
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
                        # sam.sam_mask_decoder.load_state_dict(torch.load(fine_tuned_weights_path, map_location=DEVICE))
                        print("Re-loaded fine-tuned weights after reinitialization.")
                        current_img = prompt_img_site_a
                        #current_mask = mask_site_a

                        sam_out = sam.track_new_object(img=current_img, points=point_coords)
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

            # sam_out["pred_masks"] = torch.where(
            #     rock_mask_resized > 0.5,
            #     torch.ones_like(sam_out["pred_masks"]),
            #     sam_out["pred_masks"]
            # )

            frame_with_mask = visualizer.overlay_mask(
                frame,
                sam_out["pred_masks"],
                rock_mask=None,
                save_shoreline_coords=False,
                save_path="./shoreline_jsons/test1",
                max_save_frames=300,
                frame_index=None
            )

            # frame_with_mask = overlay_mask_with_invisible_contour(
            #     frame,
            #     rock_mask_resized.cpu().numpy()[0, 0]
            # --- Blend original frame and overlay using region mask ---
            
            region_mask_json = "./region/jennette_north-2025-07-21-235641Z.json"  # Update this path as needed
            region_mask = json_to_mask(region_mask_json, frame.shape)
            # Resize mask to match display frame size (1280, 960)
            region_mask = cv2.resize(region_mask.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST)
            # Make sure mask is boolean
            mask_bool = region_mask > 0.5
            # Blend: inside mask shows original frame, outside shows overlay
            frame_resized = cv2.resize(frame, (1280, 960))
            blended = frame_with_mask.copy()
            blended[mask_bool] = frame_resized[mask_bool]         

            # Share frame with web server
            with frame_lock:
                latest_processed_frame = blended.copy()

            # Commented out GUI
            # frame_with_mask = cv2.resize(frame_with_mask, (1280, 960))
            # cv2.namedWindow('Jennette North', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Jennette North', 1280, 960)
            # cv2.imshow("Jennette North", blended)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Commented in for headless mode (No GUI)
            time.sleep(0.001)

    streaming.capture_running = False
    capture_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
