#!/usr/bin/env python3
import json, math, sys, shutil, os
from collections import Counter, defaultdict

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

if len(sys.argv) != 2:
    print("Usage: python clean_visdrone_json.py /path/to/imagenet_vid_val.json")
    sys.exit(2)

PATH = sys.argv[1]
BACKUP = PATH + ".bak"

# Make a backup once (if it doesn't already exist)
if not os.path.exists(BACKUP):
    shutil.copy2(PATH, BACKUP)
    backed_up = True
else:
    backed_up = False

with open(PATH, "r") as f:
    data = json.load(f)

anns = data.get("annotations", []) or []
problems = Counter()
examples = defaultdict(list)
kept, dropped = 0, 0
clean_anns = []

for ann in anns:
    ann_id = ann.get("id")
    bbox = ann.get("bboxes")

    # Missing bbox
    if bbox is None:
        problems["null_bbox"] += 1
        examples["null_bbox"].append(ann_id)
        dropped += 1
        continue

    # Normalize [[x,y,w,h]] -> [x,y,w,h]
    if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
        bbox = bbox[0]
        problems["nested_bbox_unwrapped"] += 1
        if len(examples["nested_bbox_unwrapped"]) < 5:
            examples["nested_bbox_unwrapped"].append(ann_id)

    # Malformed length
    if (not isinstance(bbox, list)) or len(bbox) != 4:
        problems["malformed_bbox_len"] += 1
        examples["malformed_bbox_len"].append(ann_id)
        dropped += 1
        continue

    x, y, w, h = bbox

    # Non-finite values
    if not (finite(x) and finite(y) and finite(w) and finite(h)):
        problems["nonfinite_bbox"] += 1
        examples["nonfinite_bbox"].append(ann_id)
        dropped += 1
        continue

    # Non-positive width/height
    if w <= 0 or h <= 0:
        problems["nonpositive_wh"] += 1
        examples["nonpositive_wh"].append(ann_id)
        dropped += 1
        continue

    # Clean fields
    ann["bboxes"] = [float(x), float(y), float(w), float(h)]

    # iscrowd -> default 0 if invalid
    if ann.get("iscrowd") not in (0, 1):
        problems["fixed_iscrowd"] += 1
        if len(examples["fixed_iscrowd"]) < 5:
            examples["fixed_iscrowd"].append(ann_id)
        ann["iscrowd"] = 0

    # areas -> recompute if missing/invalid
    area = ann.get("areas")
    if not finite(area) or area <= 0:
        ann["areas"] = float(w * h)
        problems["fixed_area"] += 1
        if len(examples["fixed_area"]) < 5:
            examples["fixed_area"].append(ann_id)
    else:
        ann["areas"] = float(area)

    clean_anns.append(ann)
    kept += 1

data["annotations"] = clean_anns

with open(PATH, "w") as f:
    json.dump(data, f)

# Report
print(f"{'Backed up original to: ' + BACKUP if backed_up else 'Backup already exists: ' + BACKUP}")
print(f"Cleaned file written in-place: {PATH}")
print(f"Kept {kept} annotations, dropped {dropped} bad ones.")

if problems:
    print("\nSummary of issues found (likely cause of your pycocotools error):")
    for k, v in problems.items():
        print(f"  - {k}: {v}")
    print("\nExamples (up to 5 IDs per issue):")
    for k, ids in examples.items():
        if ids:
            print(f"  {k}: {ids[:5]}")
else:
    print("\nNo problems found. Your earlier error may be from predictions formatting instead of GT.")

# Exit non-zero if anything problematic was found (useful for CI/scripts)
sys.exit(1 if (dropped > 0 or any(k.startswith(("fixed_", "non", "malformed", "null")) for k in problems)) else 0)
