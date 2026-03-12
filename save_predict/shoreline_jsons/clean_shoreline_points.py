import os
import json
import argparse
from typing import List, Tuple

# -----------------------------
# Geometry helpers (no external deps)
# -----------------------------

def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to test if point is inside polygon.
    Works for simple (non self-intersecting) polygons.
    Points exactly on an edge are treated as inside.
    """
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Check if point is on segment (x1,y1)-(x2,y2)
        # Using bounding box + cross product zero check
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        # Parametric projection test for near colinearity
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            # Cross product
            if abs((x - x1) * dy - (y - y1) * dx) < 1e-9:
                return True
        # Ray cast (horizontal ray to the right)
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if xinters == x:  # on boundary
                return True
            if xinters > x:
                inside = not inside
    return inside

def load_labelme_polygons(json_path: str, label_filter: str = None) -> List[List[Tuple[float, float]]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    polys = []
    for shape in data.get('shapes', []):
        if label_filter and shape.get('label') != label_filter:
            continue
        pts = shape.get('points', [])
        if len(pts) >= 3:
            polys.append([(float(p[0]), float(p[1])) for p in pts])
    return polys

def any_mask_contains(x: float, y: float, mask_polys: List[List[Tuple[float, float]]]) -> bool:
    for poly in mask_polys:
        if point_in_polygon(x, y, poly):
            return True
    return False

# -----------------------------
# Processing logic
# -----------------------------

def process_json(json_path: str, mask_polys: List[List[Tuple[float, float]]], edge_margin: int, backup: bool, label_filter: str = 'shoreline') -> Tuple[int, int]:
    """Return (original_points, kept_points).
    Always modifies the file in place if changes occur.
    If backup is True, writes a .bak containing the original (pre-clean) JSON once.
    """
    with open(json_path, 'r') as f:
        original_data = json.load(f)
    # Work on a mutable copy
    data = json.loads(json.dumps(original_data))

    width = data.get('imageWidth')
    height = data.get('imageHeight')
    changed = False
    total_orig = 0
    total_kept = 0

    for shape in data.get('shapes', []):
        if shape.get('label') != label_filter:
            continue
        points = shape.get('points', [])
        total_orig += len(points)
        filtered = []
        for pt in points:
            x, y = float(pt[0]), float(pt[1])
            # Edge exclusion
            if width is not None and height is not None:
                if (
                    x < edge_margin or x > (width - 1 - edge_margin) or
                    y < edge_margin or y > (height - 1 - edge_margin)
                ):
                    changed = True
                    continue
            # Mask exclusion
            if mask_polys and any_mask_contains(x, y, mask_polys):
                changed = True
                continue
            filtered.append([x, y])
        if len(filtered) != len(points):
            changed = True
        shape['points'] = filtered
        total_kept += len(filtered)

    if changed:
        if backup:
            bak_path = json_path + '.bak'
            if not os.path.exists(bak_path):
                with open(bak_path, 'w') as bf:
                    json.dump(original_data, bf, ensure_ascii=False, indent=2)
        with open(json_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return total_orig, total_kept

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Clean shoreline JSON points by removing ones inside mask & near edges.')
    parser.add_argument('--json-dir', '-d', default='./trevone/surfline_trevone_20260217_1620/', help='Directory of shoreline JSON files (default: ./twinlakes)')
    parser.add_argument('--mask-json', '-m', default='./mask/surfline_trevone_20260208_1355_000021.json', help='Path to rock mask JSON (default: ./masks/rock_mask.json)')
    parser.add_argument('--mask-label', default='rock', help='Mask label filter (default: rock)')
    parser.add_argument('--edge-margin', type=int, default=5, help='Pixel margin from each edge to remove points.')
    parser.add_argument('--backup', action='store_true', help='Create .bak with original JSON before first overwrite.')
    parser.add_argument('--label', default='shoreline', help='Target label to clean in shoreline JSONs.')

    args = parser.parse_args()

    mask_polys: List[List[Tuple[float, float]]] = []
    if args.mask_json:
        if os.path.isdir(args.mask_json):
            mask_files = [os.path.join(args.mask_json, f) for f in os.listdir(args.mask_json) if f.lower().endswith('.json')]
            for mf in mask_files:
                try:
                    mp = load_labelme_polygons(mf, args.mask_label)
                    mask_polys.extend(mp)
                except Exception as e:
                    print(f"Warning: could not load mask {mf}: {e}")
            print(f"Loaded {len(mask_polys)} mask polygon(s) from directory {args.mask_json} ({len(mask_files)} files)")
        elif os.path.isfile(args.mask_json):
            if not os.path.exists(args.mask_json):
                raise FileNotFoundError(f"Mask JSON not found: {args.mask_json}")
            mask_polys = load_labelme_polygons(args.mask_json, args.mask_label)
            print(f"Loaded {len(mask_polys)} mask polygon(s) from {args.mask_json}")
        else:
            print(f"Mask path {args.mask_json} not found; proceeding without mask.")
    else:
        print("No mask JSON provided; only edge filtering will be applied.")

    total_files = 0
    total_points_before = 0
    total_points_after = 0

    for root, _, files in os.walk(args.json_dir):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            fpath = os.path.join(root, fname)
            total_files += 1
            with open(fpath, 'r') as tf:
                try:
                    preview = json.load(tf)
                except Exception as e:
                    print(f"Skipping unreadable {fpath}: {e}")
                    continue
            if 'shapes' not in preview:
                continue
            before, after = process_json(fpath, mask_polys, args.edge_margin, args.backup, label_filter=args.label)
            total_points_before += before
            total_points_after += after
            print(f"Processed {fname}: {before} -> {after} points")

    print('\nSummary:')
    print(f"Files scanned: {total_files}")
    print(f"Total points (before): {total_points_before}")
    print(f"Total points (after):  {total_points_after}")

if __name__ == '__main__':
    main()
