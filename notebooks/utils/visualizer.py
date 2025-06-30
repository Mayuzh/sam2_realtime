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
        max_save_frames=0,
        frame_index=None
    ):
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        pred_masks = self.resize_mask(pred_masks)
        pred_masks = (pred_masks > 0.0).numpy()
        if rock_mask is not None:
            rock_mask = self.resize_mask(rock_mask)
            rock_mask = (rock_mask > 0.0).numpy()
        for i in range(pred_masks.shape[0]):
            obj_mask = (pred_masks[i, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness=2)
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
                        break
        if rock_mask is not None:
            rock_mask = (rock_mask[0, 0, :, :] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=2)
        return frame
