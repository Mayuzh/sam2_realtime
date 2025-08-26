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
