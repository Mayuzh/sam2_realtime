import os
import json
import argparse
from typing import List


def count_points(data: dict, label_filter: str = "shoreline") -> int:
    shapes = data.get("shapes", [])
    pts_counts: List[int] = []
    for s in shapes:
        if s.get("label") == label_filter:
            pts = s.get("points", [])
            pts_counts.append(len(pts))
    if not pts_counts:
        return 0  # no shoreline shape
    return max(pts_counts)  # use the most detailed shoreline polygon


def has_empty_shoreline(data: dict, label_filter: str = "shoreline") -> bool:
    for s in data.get("shapes", []):
        if s.get("label") == label_filter and len(s.get("points", [])) == 0:
            return True
    return False


def delete_pair(json_path: str, image_path: str, dry_run: bool, reason: str = ""):
    if dry_run:
        print(f"DRY-RUN would delete: {json_path} {f'({reason})' if reason else ''}")
        if image_path and os.path.exists(image_path):
            print(f"DRY-RUN would delete: {image_path}")
        return
    try:
        os.remove(json_path)
        print(f"Deleted JSON: {json_path} {f'({reason})' if reason else ''}")
    except OSError as e:
        print(f"Failed to delete {json_path}: {e}")
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Deleted PNG:  {image_path}")
        except OSError as e:
            print(f"Failed to delete {image_path}: {e}")


def process_folder(folder: str, min_points: int, dry_run: bool):
    total = 0
    removed = 0
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            json_path = os.path.join(root, fname)
            total += 1
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skipping unreadable JSON {json_path}: {e}")
                continue
            empty_poly = has_empty_shoreline(data, label_filter="shoreline")
            pts = count_points(data, label_filter="shoreline")
            # Determine image path
            image_name = data.get("imagePath")
            if image_name:
                image_path = os.path.join(root, image_name) if not os.path.isabs(image_name) else image_name
            else:
                image_path = os.path.splitext(json_path)[0] + '.png'
            if empty_poly:
                delete_pair(json_path, image_path, dry_run, reason="empty shoreline polygon")
                removed += 1
                continue  # no need to evaluate min_points
            if pts < min_points:
                delete_pair(json_path, image_path, dry_run, reason=f"points({pts}) < {min_points}")
                removed += 1
    print(f"Summary: scanned {total} JSON files, removed {removed} (< {min_points} points or empty). Dry-run={dry_run}")


def main():
    parser = argparse.ArgumentParser(description="Delete shoreline JSON/PNG pairs with too few points or empty polygons.")
    parser.add_argument('--dir', '-d', default='./trevone', help='Directory to scan (default: current)')
    parser.add_argument('--min-points', '-m', type=int, default=50, help='Minimum number of points required (default: 120)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without removing')
    args = parser.parse_args()

    process_folder(args.dir, args.min_points, args.dry_run)


if __name__ == '__main__':
    main()