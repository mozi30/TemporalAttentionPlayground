#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

def safe_len(x):
    return 0 if x is None else len(x)

def summarize_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    annotations = data.get("annotations", []) or []
    categories = {c["id"]: c.get("name", f"cat_{c['id']}") for c in data.get("categories", [])}

    # Top-level counts (what you saw before)
    total_annotations = len(annotations)
    flat_bbox_none = 0
    flat_area_none = 0
    flat_bbox_present = 0

    # Video-aware stats
    total_frame_slots = 0          # sum over all len(bboxes)
    frame_slots_with_box = 0       # count of per-frame boxes present
    frame_slots_without_box = 0    # count of per-frame Nones

    ann_len_list = []              # per-annotation #frames (len of list)
    ann_present_list = []          # per-annotation frames with box
    ann_presence_rate = []         # present/len per annotation (where len>0)

    ann_all_none = 0               # annotations where every frame is None (or list missing)
    ann_some_present = 0           # annotations with at least one bbox
    ann_missing_list = 0           # annotations where bboxes key is missing or None

    # Per-category accumulators
    cat_frames = defaultdict(int)
    cat_present = defaultdict(int)
    cat_ann_counts = defaultdict(int)

    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_name = categories.get(cat_id, f"Unknown({cat_id})")
        cat_ann_counts[cat_name] += 1

        bboxes = ann.get("bboxes", None)
        areas = ann.get("areas", None)

        # Old flat counters (counting each None element)
        if bboxes is None:
            flat_bbox_none += 1  # whole list missing counts once
        else:
            for b in bboxes:
                if b is None:
                    flat_bbox_none += 1
                else:
                    flat_bbox_present += 1

        if areas is None:
            flat_area_none += 1
        else:
            for a in areas:
                if a is None:
                    flat_area_none += 1

        # Video-aware accounting
        if not bboxes:
            ann_missing_list += 1
            ann_len_list.append(0)
            ann_present_list.append(0)
            ann_all_none += 1
            continue

        n_frames = len(bboxes)
        present = sum(1 for b in bboxes if b is not None)
        missing = n_frames - present

        total_frame_slots += n_frames
        frame_slots_with_box += present
        frame_slots_without_box += missing

        ann_len_list.append(n_frames)
        ann_present_list.append(present)

        if present == 0:
            ann_all_none += 1
        else:
            ann_some_present += 1
            if n_frames > 0:
                ann_presence_rate.append(present / n_frames)

        # Per-category
        cat_frames[cat_name] += n_frames
        cat_present[cat_name] += present

    # Print summary
    print(f"📄 File: {annotation_file}")
    print(f"-----------------------------------")
    print(f"Total annotations: {total_annotations}")
    print(f"Annotations with at least one box: {ann_some_present}")
    print(f"Annotations with all frames None/missing: {ann_all_none}")
    print(f"Annotations missing 'bboxes' or empty list: {ann_missing_list}")

    # Frame-level view
    print("\nFrame-level presence (video-aware):")
    print(f"  Total frame slots: {total_frame_slots}")
    print(f"  Frames with boxes: {frame_slots_with_box}")
    print(f"  Frames without boxes (None): {frame_slots_without_box}")
    if total_frame_slots > 0:
        pct = 100.0 * frame_slots_without_box / total_frame_slots
        print(f"  % frames without boxes: {pct:.2f}%")

    # Per-annotation presence
    if ann_len_list:
        lengths = [x for x in ann_len_list if x > 0]
        presences = [p for p in ann_present_list]
        rates = ann_presence_rate

        print("\nPer-annotation stats:")
        print(f"  Mean frames per annotation: {mean(lengths) if lengths else 0:.2f}")
        print(f"  Median frames per annotation: {median(lengths) if lengths else 0:.0f}")
        print(f"  Mean frames-with-box per annotation: {mean(presences):.2f}")
        if rates:
            print(f"  Mean presence rate per annotation: {100*mean(rates):.2f}%")
            print(f"  Median presence rate per annotation: {100*median(rates):.2f}%")
        else:
            print("  Presence rates unavailable (no non-empty tracks).")

    # Old flat counters (for comparison to your previous output)
    print("\n(Flat element counts for compatibility):")
    print(f"  Total bounding boxes (non-None elements): {flat_bbox_present}")
    print(f"  Bounding boxes None elements: {flat_bbox_none}")
    print(f"  Areas None elements: {flat_area_none}")

    # Per-category presence rates
    print("\nPer-category presence rates:")
    for cat, n_frames in sorted(cat_frames.items(), key=lambda x: -x[1]):
        present = cat_present[cat]
        rate = (100.0 * present / n_frames) if n_frames else 0.0
        print(f"  {cat}: frames={n_frames}, with_box={present}, presence_rate={rate:.2f}% (ann_count={cat_ann_counts[cat]})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 annotation-summary.py <annotation_file.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    summarize_annotations(str(path))
