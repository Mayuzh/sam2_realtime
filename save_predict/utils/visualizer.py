import cv2
import torch
import numpy as np
from utils.helpers import write_labelme_json
import os

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
        max_save_frames=None,  # Use None to indicate no limit on saved frames
        frame_index=None,
        video_filename=None,  # NEW: original video filename for naming
        save_even_if_empty=False,  # NEW: save frame and empty JSON if no contours
    ):
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        original_frame = frame.copy()  # keep a copy of the resized original before drawing contours
        pred_masks = self.resize_mask(pred_masks)
        pred_masks = (pred_masks > 0.0).numpy()
        if rock_mask is not None:
            rock_mask = self.resize_mask(rock_mask)
            rock_mask = (rock_mask > 0.0).numpy()
        saved_any = False
        for i in range(pred_masks.shape[0]):
            obj_mask = (pred_masks[i, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)
            if save_shoreline_coords and save_path:
                if max_save_frames is None or self.saved_frame_count < max_save_frames:
                    os.makedirs(save_path, exist_ok=True)
                    for cnt in contours:
                        if len(cnt) > 2:
                            coords = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]
                            if video_filename is None:
                                try:
                                    from utils.config import VIDEO_PATH as _CFG_VIDEO_PATH
                                    video_filename = _CFG_VIDEO_PATH
                                except Exception:
                                    pass
                            if video_filename:
                                base = os.path.splitext(os.path.basename(video_filename))[0]
                            else:
                                base = "shoreline_frame"
                            idx_val = frame_index if frame_index is not None else self.saved_frame_count
                            frame_name = f"{base}_{idx_val:06d}.png"
                            image_output_path = os.path.join(save_path, frame_name)
                            # Save the original (no-contour) frame image first
                            cv2.imwrite(image_output_path, original_frame)
                            write_labelme_json(
                                image_output_path,
                                coords=coords,
                                image_shape=(self.video_height, self.video_width)
                            )
                            self.saved_frame_count += 1
                            saved_any = True
                            break
        # Fallback: save empty JSON if requested and nothing saved
        if not saved_any and save_shoreline_coords and save_path and save_even_if_empty:
            if video_filename is None:
                try:
                    from utils.config import VIDEO_PATH as _CFG_VIDEO_PATH
                    video_filename = _CFG_VIDEO_PATH
                except Exception:
                    pass
            if video_filename:
                base = os.path.splitext(os.path.basename(video_filename))[0]
            else:
                base = "shoreline_frame"
            idx_val = frame_index if frame_index is not None else self.saved_frame_count
            frame_name = f"{base}_{idx_val:06d}.png"
            image_output_path = os.path.join(save_path, frame_name)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(image_output_path, original_frame)
            # empty coords
            write_labelme_json(
                image_output_path,
                coords=[],
                image_shape=(self.video_height, self.video_width)
            )
            self.saved_frame_count += 1
        if rock_mask is not None:
            rock_mask = (rock_mask[0, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=2)
        return frame
