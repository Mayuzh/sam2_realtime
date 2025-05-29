import os
import numpy as np
import pandas as pd
from pathlib import Path

shoreline_folder = Path("./csv")
output_folder = Path("./rectified_csv")
output_folder.mkdir(parents=True, exist_ok=True)

# # === World file values (from jgw/aux.xml) ===
# A = 0.0786478839   # pixel size in X
# B = 0.8764197294   # row rotation (usually 0)
# D = 0.1390958176   # column rotation (usually 0)
# E = -0.4843093623  # pixel size in Y (usually negative)
# C = -13582935.2430 # X of upper-left center pixel
# F = 4434153.9850   # Y of upper-left center pixel

A = 0.05365753507
D = -0.07059652503
B = -0.37528290919
E = -0.29418632452
C = -13580775.7415
F = 4434157.4295


for csv_file in shoreline_folder.glob("*.csv"):
    base_name = csv_file.stem

    try:
        # Load shoreline points
        shoreline_df = pd.read_csv(csv_file)
        points = np.array(shoreline_df[["x", "y"]])

        # Flip y-axis to match ArcGIS if JSON came from top-left origin
        #IMAGE_HEIGHT = 1920
        #points[:, 1] = IMAGE_HEIGHT - points[:, 1]

        # Apply world file affine transform directly
        map_x = A * points[:, 0] + B * points[:, 1] + C
        map_y = D * points[:, 0] + E * points[:, 1] + F

        # Save rectified output
        out_df = pd.DataFrame({
            "id": shoreline_df["id"],
            "map_x": map_x,
            "map_y": map_y
        })
        out_path = output_folder / f"{base_name}_rectified.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path.name}")

    except Exception as e:
        print(f"❌ Error processing {csv_file.name}: {e}")
