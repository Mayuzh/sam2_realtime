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

    target_h, target_w = image_shape[:2]
    source_w = data.get('imageWidth')
    source_h = data.get('imageHeight')
    if not source_w or not source_h:
        raise ValueError(
            f"Mask JSON must contain imageWidth and imageHeight: {json_path}"
        )

    scale_x = target_w / source_w
    scale_y = target_h / source_h
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    for shape in data['shapes']:
        points = np.asarray(shape['points'], dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, target_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, target_h - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask

def write_labelme_json(image_path, coords, image_shape, image, label="shoreline", margin=10):
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
    output_dir = os.path.dirname(image_save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    saved_image = cv2.resize(image, (w, h))
    if not cv2.imwrite(image_save_path, saved_image):
        raise OSError(f"Failed to save shoreline image: {image_save_path}")
    json_path = os.path.splitext(image_save_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved shoreline JSON: {json_path}")
