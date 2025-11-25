

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from PIL import Image, ImageDraw
import os
import time

SPLID_TO_ANNOTATION_MAP = {
    "train": "annotations/imagenet_vid_train.json",
    "val": "annotations/imagenet_vid_val.json",
    "test": "annotations/imagenet_vid_test.json"
}


def black_out_region(dataset_path, ir, videos):
    video_id = ir.get("video_id")
    frame_id = ir.get("frame_id")

    videos_filtered = [v for v in videos if v["id"] == video_id]
    if not videos_filtered:
        return False  # nothing processed

    video = videos_filtered[0]
    image_paths = video.get("file_names", []) or []
    if frame_id < 0 or frame_id >= len(image_paths):
        print(f"Warning: frame_id {frame_id} out of range for video_id {video_id}. Skipping.")
        return False

    image_path = image_paths[frame_id]
    full_image_path = os.path.join(dataset_path, image_path)

    if not os.path.exists(full_image_path):
        print(f"Warning: image {full_image_path} not found. Skipping.")
        return False

    bbox = ir.get("bbox", [])
    if len(bbox) != 4:
        print(f"Warning: invalid bbox {bbox} for video_id {video_id}, frame_id {frame_id}. Skipping.")
        return False

    x, y, w, h = bbox

    # open, modify, save image
    try:
        img = Image.open(full_image_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], fill="black")
        img.save(full_image_path)
    except Exception as e:
        print(f"Error processing image {full_image_path}: {e}")
        return False
    return True  # successfully processed


def process_ignored_regions(dataset_path, splid):
    path_to_annotation = f"{dataset_path}{SPLID_TO_ANNOTATION_MAP[splid]}"
    with open(path_to_annotation, "r") as f:
        src = json.load(f)

    ignored_regions = src.get("ignored_regions", []) or []
    videos = src.get("videos", []) or []

    num_ignored = len(ignored_regions)
    print(f"Found {num_ignored} ignored regions in {path_to_annotation}")
    if num_ignored == 0:
        print("No ignored regions to process. Exiting.")
        return

    print("Processing ignored regions...")

    processed_regions = 0
    last_print = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(black_out_region, dataset_path, ir, videos)
            for ir in ignored_regions
        ]

        for future in as_completed(futures):
            # propagate errors if any
            result = future.result()
            if result:
                processed_regions += 1

            now = time.time()
            if now - last_print > 10:
                print(f"Processed {processed_regions}/{num_ignored} ignored regions...")
                last_print = now

    print(f"Finished processing ignored regions. Total processed: {processed_regions}/{num_ignored}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Black out ignored regions in video-style annotations.")
    ap.add_argument("--datasetPath", help="Input JSON (video-style annotations)")
    ap.add_argument("--datasetSplit", choices=["train", "val", "test","all"], default="train", help="Type of dataset (default: train)")
    args = ap.parse_args()
    dataset_path = args.datasetPath
    dataset_split = args.datasetSplit

    if dataset_split == "all":
        for split in ["train", "val", "test"]:
             print(f"Processing split: {split}")
             process_ignored_regions(dataset_path,split)
    else:
        process_ignored_regions(dataset_path, dataset_split)

    




        


    