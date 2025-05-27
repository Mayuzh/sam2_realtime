import os
import json
import csv

# Path to your folder containing JSON files
input_folder = "../masks/"
output_folder = "./csv/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(input_folder, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        for shape in shapes:
            if shape["label"] == "shoreline":
                points = shape["points"]
                csv_filename = os.path.splitext(filename)[0] + ".csv"
                csv_path = os.path.join(output_folder, csv_filename)

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id", "x", "y"])
                    for idx, (x, y) in enumerate(points, 1):
                        writer.writerow([idx, x, y])

                print(f"Exported: {csv_path}")
