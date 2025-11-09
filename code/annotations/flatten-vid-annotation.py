#!/usr/bin/env python3
import json, math, sys, argparse
from pathlib import Path
from collections import defaultdict

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def main():
    ap = argparse.ArgumentParser(description="Flatten video-style VisDrone annotations to COCO format and summarize.")
    ap.add_argument("IN", help="Input JSON (video-style annotations)")
    ap.add_argument("OUT", help="Output JSON (COCO-style annotations)")
    ap.add_argument("--img-prefix", default="", help="Prefix to prepend to image file_name")
    ap.add_argument("--subsample", type=int, default=1, help="Keep every N-th frame (>=1).")
    ap.add_argument("--min-area", type=float, default=0.0, help="Drop boxes with area<w*h < min-area.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")
    args = ap.parse_args()

    with open(args.IN, "r") as f:
        src = json.load(f)

    videos = src.get("videos", [])
    anns   = src.get("annotations", []) or []
    cats   = src.get("categories", [])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    # ------------------------------------------------------------------
    # Build images (one per frame)
    # ------------------------------------------------------------------
    images = []
    img_id = 1
    vid_frame_to_img = {}
    vid_meta = {}

    videos_sorted = sorted(videos, key=lambda v: v["id"])
    for v in videos_sorted:
        vid = v["id"]
        W, H = v["width"], v["height"]
        files = v.get("file_names", []) or []
        T = len(files)
        vid_meta[vid] = (W, H, T)
        for t, fn in enumerate(files):
            if args.subsample > 1 and (t % args.subsample != 0):
                continue
            file_name = f"{args.img_prefix}{fn}" if args.img_prefix else fn
            images.append({
                "id": img_id,
                "width": W,
                "height": H,
                "file_name": file_name,
                "video_id": vid,
                "frame_id": t,
            })
            vid_frame_to_img[(vid, t)] = img_id
            img_id += 1

    # ------------------------------------------------------------------
    # Flatten annotations (skip Nones / invalid boxes)
    # ------------------------------------------------------------------
    coco_anns = []
    ann_id = 1
    counts = defaultdict(int)
    cat_counts = defaultdict(int)

    for a in anns:
        counts["total_annotations"] += 1
        vid = a.get("video_id")
        cid = a.get("category_id")
        iscrowd = int(a.get("iscrowd", 0))
        bbox_list = a.get("bboxes")
        area_list = a.get("areas")

        if vid not in vid_meta:
            counts["skipped_missing_video"] += 1
            continue
        W, H, T = vid_meta[vid]

        if bbox_list is None:
            counts["skipped_no_bbox_list"] += 1
            continue

        # Normalize single bbox -> list
        if isinstance(bbox_list, list) and len(bbox_list) == 4 and all(isinstance(v, (int, float)) for v in bbox_list):
            bbox_list = [bbox_list]

        if not isinstance(bbox_list, list):
            counts["skipped_bad_bbox_list"] += 1
            continue

        if isinstance(area_list, list) and len(area_list) != len(bbox_list):
            counts["warn_len_mismatch"] += 1

        for t, bbox in enumerate(bbox_list):
            if (vid, t) not in vid_frame_to_img:
                # dropped due to subsample or out of range
                if t >= T:
                    counts["skipped_out_of_range"] += 1
                continue

            counts["frames_iterated"] += 1

            if bbox is None:
                counts["skipped_null_bbox"] += 1
                continue
            if (not isinstance(bbox, list)) or len(bbox) != 4:
                counts["skipped_bad_bbox"] += 1
                continue

            x, y, w, h = bbox
            if not (finite(x) and finite(y) and finite(w) and finite(h)):
                counts["skipped_nonfinite"] += 1
                continue
            if w <= 0 or h <= 0:
                counts["skipped_nonpos_wh"] += 1
                continue

            # Clip to image bounds
            x = float(max(0.0, min(x, W)))
            y = float(max(0.0, min(y, H)))
            w = float(max(0.0, min(w, W - x)))
            h = float(max(0.0, min(h, H - y)))
            if w <= 0 or h <= 0:
                counts["skipped_clipped_to_zero"] += 1
                continue

            # Area
            if isinstance(area_list, list) and t < len(area_list) and finite(area_list[t]):
                area = float(area_list[t])
            else:
                area = float(w * h)
            if area < args.min_area:
                counts["skipped_tiny_area"] += 1
                continue

            img_id_ref = vid_frame_to_img[(vid, t)]
            coco_anns.append({
                "id": ann_id,
                "image_id": img_id_ref,
                "category_id": cid,
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": iscrowd,
                "segmentation": [],
                "track_id": a.get("id"),
            })
            ann_id += 1
            counts["kept"] += 1
            cat_counts[cid] += 1

    # ------------------------------------------------------------------
    # Save COCO output
    # ------------------------------------------------------------------
    coco = {
        "info": src.get("info", {}),
        "images": images,
        "annotations": coco_anns,
        "categories": cats,
    }

    kw = {"ensure_ascii": False}
    if args.pretty:
        kw["indent"] = 2
    else:
        kw["separators"] = (",", ":")

    with open(args.OUT, "w") as f:
        json.dump(coco, f, **kw)

    # ------------------------------------------------------------------
    # Annotation Summary & Diagnostics
    # ------------------------------------------------------------------
    print("\n================= SUMMARY =================")
    print(f"📄 Input file: {args.IN}")
    print(f"💾 Output file: {args.OUT}")
    print("-------------------------------------------")
    print(f"Images created:       {len(images):,}")
    print(f"Annotations kept:     {counts['kept']:,}")
    print(f"Total annotations in: {counts['total_annotations']:,}")
    print("-------------------------------------------")
    print("Skipped counts:")
    for k, v in sorted(counts.items()):
        if k.startswith("skipped") or k.startswith("warn"):
            print(f"  {k:25s}: {v}")
    print("-------------------------------------------")
    print("Category distribution (kept annotations):")
    for cid, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        cname = cat_id_to_name.get(cid, f"cat_{cid}")
        print(f"  {cname:20s}: {n:,}")
    print("===========================================\n")

    print(json.dumps({
        "images": len(images),
        "annotations_kept": counts["kept"],
        "diagnostics": counts,
        "out_path": args.OUT
    }, indent=2))


if __name__ == "__main__":
    main()
