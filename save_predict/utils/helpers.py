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
    """Create a binary mask from a LabelMe JSON, scaled to the target image_shape.

    If the JSON contains imageWidth/imageHeight and they differ from image_shape,
    the polygon points are scaled accordingly so the mask aligns with the target frame.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    target_h, target_w = image_shape[:2]
    src_w = data.get('imageWidth')
    src_h = data.get('imageHeight')
    # Compute per-axis scale; if source dims missing/zero, default to 1
    if src_w and src_w > 0 and src_h and src_h > 0:
        sx = float(target_w) / float(src_w)
        sy = float(target_h) / float(src_h)
    else:
        sx = sy = 1.0

    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    for shape in data.get('shapes', []):
        pts = np.array(shape.get('points', []), dtype=np.float32)
        if pts.size == 0:
            continue
        # Scale points from JSON coordinate space to target frame space
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 1)
    return mask

def write_labelme_json(image_path, coords, image_shape, label="shoreline", margin=10):
    import json
    import os
    import cv2
    import numpy as np
    # from . import streaming  # removed dependency on streaming.latest_frame

    if os.path.exists(image_path):
        saved_image = cv2.imread(image_path)
    else:
        saved_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    h, w = saved_image.shape[:2]
    # Previous clamping logic (commented out to preserve original coordinates and floats)
    # clamped = []
    # for x, y in coords:
    #     nx = int(min(max(x, margin), w - margin - 1))
    #     ny = int(min(max(y, margin), h - margin - 1))
    #     clamped.append([nx, ny])

    # Preserve original coords, cast to float for LabelMe compatibility
    points_formatted = [[float(x), float(y)] for (x, y) in coords]

    data = {
        "version": "0.4.16",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "text": "",
                "points": points_formatted,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return json_path
