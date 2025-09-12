#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List

def rows_from_labelme_json(json_path: Path, keep_metadata: bool = False) -> Iterable[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("imagePath", "")
    image_width = data.get("imageWidth", "")
    image_height = data.get("imageHeight", "")
    shapes = data.get("shapes", [])

    feature_id = 0
    for shp in shapes:
        feature_id += 1
        label = shp.get("label", "")
        group_id = shp.get("group_id", "")
        shape_type = shp.get("shape_type", "polygon").lower()
        points = shp.get("points", []) or []

        if shape_type != "polygon":
            continue  # only polygons supported

        for vi, pt in enumerate(points):
            x, y = float(pt[0]), float(pt[1])
            row = {
                "feature_id": feature_id,
                "label": label,
                "vertex_index": vi,
                "x": x,
                "y": -y,
                "group_id": group_id
            }
            if keep_metadata:
                row.update({
                    "source_file": str(json_path),
                    "image_path": image_path,
                    "image_width": image_width,
                    "image_height": image_height
                })
            yield row

def gather_json_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.glob("*.json")])
    raise FileNotFoundError(f"Could not find JSON(s) at {input_path}")

def write_csv(rows: Iterable[Dict[str, Any]], out_path: Path, keep_metadata: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_metadata:
        fieldnames = [
            "source_file", "image_path", "image_width", "image_height",
            "feature_id", "label", "vertex_index", "x", "y", "group_id"
        ]
    else:
        fieldnames = [
            "feature_id", "label", "vertex_index", "x", "y", "group_id"
        ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Convert LabelMe polygon JSON annotations to CSV for ArcGIS XY import.")
    ap.add_argument("--input", default="./seabright/13/walton_lighthouse-2024-11-16-210138Z_000007.json", help="Path to a LabelMe JSON file or a folder of JSONs (default: ./input_jsons).")
    ap.add_argument("--output", default="./csv/walton_lighthouse-2024-11-16-210138Z_000007.csv", help="Path to the output CSV file (default: ./output/annotations.csv).")
    ap.add_argument("--keep-metadata", action="store_true", help="Keep source_file/image_path/width/height columns. Default = False.")
    args = ap.parse_args()

    input_path = Path(args.input)
    json_files = gather_json_files(input_path)
    all_rows = []
    for jp in json_files:
        for row in rows_from_labelme_json(jp, keep_metadata=args.keep_metadata):
            all_rows.append(row)

    write_csv(all_rows, Path(args.output), keep_metadata=args.keep_metadata)
    print(f"Wrote {len(all_rows)} rows from {len(json_files)} JSON file(s) to {args.output}")

if __name__ == "__main__":
    main()
