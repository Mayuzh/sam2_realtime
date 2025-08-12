import cv2
import numpy as np
import json
import os
import torch

def is_mask_lost(mask_tensor, threshold=0.001):
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

def write_labelme_json(image_path, coords, image_shape, label="shoreline", margin=10):
    import utils.streaming as streaming
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
    #saved_image = cv2.resize(streaming.latest_frame, (1280, 960))
    saved_image = cv2.resize(streaming.latest_frame, (1440, 1080))
    cv2.imwrite(image_save_path, saved_image)
    json_path = os.path.splitext(image_save_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved shoreline JSON: {json_path}")
