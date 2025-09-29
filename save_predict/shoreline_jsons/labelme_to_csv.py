#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List

def rows_from_labelme_json(json_path: Path, keep_metadata: bool = False, y_flip: bool = True, drop_closing_point: bool = True) -> Iterable[Dict[str, Any]]:
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

        # Accept both polygon and polyline-like shapes
        if shape_type not in {"polygon", "polyline", "linestrip"}:
            continue

        # Drop duplicate closing point (LabelMe polygons may include start==end)
        if drop_closing_point and len(points) >= 2:
            x0, y0 = points[0]
            xN, yN = points[-1]
            if float(x0) == float(xN) and float(y0) == float(yN):
                points = points[:-1]

        for vi, pt in enumerate(points):
            x, y = float(pt[0]), float(pt[1])
            yf = -y if y_flip else y
            row = {
                "feature_id": feature_id,
                "label": label,
                "vertex_index": vi,
                "x": x,
                "y": yf,
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
    ap = argparse.ArgumentParser(description="Convert LabelMe polygon/polyline JSON annotations to CSV for downstream plotting/GIS.")
    ap.add_argument("--input", default="./jennette_north/active/13", help="Path to a LabelMe JSON file or a folder of JSONs.")
    ap.add_argument("--output", default="./csv/jennette_north/active/13", help="Path to the output CSV file OR an output folder when --per-file is used.")
    ap.add_argument("--per-file", action="store_true", help="If set (or when --output is a directory), write one CSV per input JSON into the output folder.")
    ap.add_argument("--keep-metadata", action="store_true", help="Keep source_file/image_path/width/height columns. Default = False.")
    ap.add_argument("--no-y-flip", action="store_true", help="Do not flip image Y to Y-up (default flips by exporting y=-y).")
    ap.add_argument("--keep-closed", action="store_true", help="Keep duplicate closing point if present (default drops it).")
    args = ap.parse_args()

    input_path = Path(args.input)
    json_files = gather_json_files(input_path)

    out_path = Path(args.output)
    treat_output_as_dir = args.per_file or (out_path.exists() and out_path.is_dir()) or (out_path.suffix.lower() != ".csv")

    if treat_output_as_dir:
        out_dir = out_path if out_path.suffix == "" else out_path  # allow non-.csv paths
        out_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0
        for jp in json_files:
            rows = list(rows_from_labelme_json(jp, keep_metadata=args.keep_metadata, y_flip=not args.no_y_flip, drop_closing_point=not args.keep_closed))
            total_rows += len(rows)
            out_csv = out_dir / (jp.stem + ".csv")
            write_csv(rows, out_csv, keep_metadata=args.keep_metadata)
        print(f"Wrote {total_rows} rows across {len(json_files)} JSON file(s) into folder: {str(out_dir)}")
    else:
        all_rows = []
        for jp in json_files:
            for row in rows_from_labelme_json(jp, keep_metadata=args.keep_metadata, y_flip=not args.no_y_flip, drop_closing_point=not args.keep_closed):
                all_rows.append(row)
        write_csv(all_rows, out_path, keep_metadata=args.keep_metadata)
        print(f"Wrote {len(all_rows)} rows from {len(json_files)} JSON file(s) to {str(out_path)}")

if __name__ == "__main__":
    main()
