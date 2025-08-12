import cv2
import numpy as np

# Load the image
image_path = "walton_lighthouse-2025-05-13-231928Z.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found at the specified path.")

# Image dimensions
height, width = image.shape[:2]

# Control points (pixel coordinates)
img_pts = np.float32([
    [791.375972, 628.378074],
    [1374.643124, 597.803646],
    [2368.665471, 1684.363178],
    [2097.046470, 581.855386],
    [2414.474452, 534.290337],
    [1249.132130, 494.491128]
])

# Map coordinates (real-world coordinates)
map_pts = np.float32([
    [-13582315.801382, 4433960.272701],
    [-13582301.339824, 4434056.329426],
    [-13581280.330045, 4433666.907128],
    [-13581996.969177, 4434090.939516],
    [-13581975.880623, 4434140.969220],
    [-13582633.161714, 4434194.951628]
])

# Residuals from ArcGIS
residuals = [
    [150.834034, 56.905607],
    [-3.808030, 8.640396],
    [-23.142663, -9.338814],
    [76.640841, 23.811841],
    [26.127321, 13.068383],
    [-226.651503, -93.087412]
]

# Scale down residuals to avoid over-correction
scale_factor = 0.3
corrected_img_pts = img_pts.copy()
for i in range(len(img_pts)):
    corrected_img_pts[i][0] += residuals[i][0] * scale_factor
    corrected_img_pts[i][1] += residuals[i][1] * scale_factor

# Compute affine transformation
print("Computing affine transformation...")
M = cv2.getAffineTransform(corrected_img_pts[:3], img_pts[:3])

# Warp the image
warped = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# Save the result
output_path = "distortion_corrected.jpg"
success = cv2.imwrite(output_path, warped)
if success:
    print(f"Corrected image saved to {output_path}")
else:
    print("Failed to save the corrected image.")

# Debug visualization
for i, pt in enumerate(corrected_img_pts):
    cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    cv2.putText(image, f"Point {i}", (int(pt[0]), int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite("debug_corrected_points.jpg", image)