#!/usr/bin/env python3
"""
Prepare VisDrone-VID for TransVOD++:
- Optionally place images under out_root/Data/VID via symlink/copy/hardlink (with --force and fallback)
- Generate COCO-VID style JSONs: annotations/imagenet_vid_{train,val}.json

File names in JSON are relative to: {out_root}/Data/VID
Layout created/expected:
  out_root/
    Data/VID/VisDrone-VID/train/sequences/<video>/*.jpg
    Data/VID/VisDrone-VID/val/sequences/<video>/*.jpg
    annotations/imagenet_vid_train.json
    annotations/imagenet_vid_val.json
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

from PIL import Image

# Default 10-class setup (drop 0: ignored, 11: others)
DEFAULT_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}
DEFAULT_IGNORED = {0, 11}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_categories(categories_json: Optional[str], keep_others: bool) -> Tuple[Dict[int, str], Set[int]]:
    """Load categories mapping from JSON or defaults.

    categories_json should map string ids to names. When None, defaults are used.
    If keep_others=True, category id 11 is kept; otherwise it is filtered out.
    """
    if categories_json is None:
        cats = DEFAULT_CATEGORIES.copy()
        ignored = set(DEFAULT_IGNORED)
        if keep_others:
            cats[11] = "others"
            ignored = {0}
        return cats, ignored

    with open(categories_json, "r") as f:
        raw = json.load(f)
    raw_int = {int(k): v for k, v in raw.items()}

    cats = {k: v for k, v in raw_int.items() if k != 0}
    ignored = {0}
    if not keep_others and 11 in cats:
        cats.pop(11)

    if 0 in cats:
        raise ValueError("Category id 0 must not be included (reserved as ignored).")
    if not cats:
        raise ValueError("No categories left after filtering.")

    return cats, ignored if not keep_others else {0}


def list_video_dirs(sequences_root: str) -> List[str]:
    if not os.path.isdir(sequences_root):
        raise FileNotFoundError(f"Missing sequences directory: {sequences_root}")
    vids = [p for p in glob.glob(os.path.join(sequences_root, "*")) if os.path.isdir(p)]
    vids.sort()
    return vids


def iter_frames(seq_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(seq_dir, f"*{ext}")))
    files.sort()
    return files


def _force_remove(path: str):
    p = Path(path)
    try:
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)  # type: ignore[arg-type]
        elif p.is_dir():
            shutil.rmtree(path)
    except Exception as e:
        raise RuntimeError(f"Failed to remove existing path '{path}': {e}")


def safe_symlink(src: str, dst: str, force: bool = False):
    src_real = os.path.realpath(src)
    dst_path = Path(dst)
    if dst_path.exists() or dst_path.is_symlink():
        # If already correct, keep; else handle per 'force'
        if dst_path.is_symlink() and os.path.realpath(dst) == src_real:
            return
        if force:
            _force_remove(dst)
        else:
            raise FileExistsError(f"Destination exists and is not the expected symlink: {dst}")
    # Ensure parent exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=os.path.isdir(src))


def link_or_copy_sequences(
    src_sequences_root: str,
    dst_sequences_root: str,
    mode: str,
    force: bool,
    fallback: Optional[str],
):
    """Place sequences into destination using symlink/copy/hardlink.

    If mode fails and fallback is provided, tries the fallback (hardlink -> copy).
    """
    os.makedirs(os.path.dirname(dst_sequences_root), exist_ok=True)

    def _do(mode_inner: str):
        if mode_inner == "symlink":
            safe_symlink(src_sequences_root, dst_sequences_root, force=force)
            return

        # For copy/hardlink, ensure destination is a real directory
        if os.path.islink(dst_sequences_root):
            if force:
                _force_remove(dst_sequences_root)
                os.makedirs(dst_sequences_root, exist_ok=True)
            else:
                raise FileExistsError(
                    f"Destination is a symlink (not a real directory): {dst_sequences_root}. Use --force or remove it."
                )
        elif os.path.exists(dst_sequences_root) and not os.path.isdir(dst_sequences_root):
            if force:
                _force_remove(dst_sequences_root)
                os.makedirs(dst_sequences_root, exist_ok=True)
            else:
                raise FileExistsError(
                    f"Destination exists and is not a directory: {dst_sequences_root}. Use --force or remove it."
                )
        else:
            os.makedirs(dst_sequences_root, exist_ok=True)
        for vdir in list_video_dirs(src_sequences_root):
            vname = os.path.basename(vdir)
            dst_vdir = os.path.join(dst_sequences_root, vname)
            os.makedirs(dst_vdir, exist_ok=True)
            for src_img in iter_frames(vdir):
                dst_img = os.path.join(dst_vdir, os.path.basename(src_img))
                if os.path.exists(dst_img):
                    continue
                if mode_inner == "copy":
                    shutil.copy2(src_img, dst_img)
                elif mode_inner == "hardlink":
                    try:
                        os.link(src_img, dst_img)
                    except OSError:
                        # Fallback silently to copy if individual hardlink fails
                        shutil.copy2(src_img, dst_img)
                else:
                    raise ValueError(f"Unknown link mode: {mode_inner}")

    try:
        _do(mode)
    except Exception as e:
        if not fallback or fallback == "none":
            raise
        print(f"[{mode}] failed with: {e}. Trying fallback: {fallback}")
        # Ensure a clean destination if we are switching strategies and --force is set
        if force and mode != fallback and os.path.exists(dst_sequences_root):
            # If switching from a bad symlink to copy/hardlink, remove it first
            if os.path.islink(dst_sequences_root) or not os.path.isdir(dst_sequences_root):
                _force_remove(dst_sequences_root)
        _do(fallback)


def parse_split(vis_split_root: str, rel_prefix: str, categories: Dict[int, str], ignored_ids: Set[int]) -> Dict:
    seq_dir_root = os.path.join(vis_split_root, "sequences")
    ann_dir_root = os.path.join(vis_split_root, "annotations")

    if not os.path.isdir(seq_dir_root):
        raise FileNotFoundError(f"Missing sequences: {seq_dir_root}")
    if not os.path.isdir(ann_dir_root):
        raise FileNotFoundError(f"Missing annotations: {ann_dir_root}")

    videos, images, anns = [], [], []
    image_id = 1
    ann_id = 1
    video_id = 1

    for vid_path in list_video_dirs(seq_dir_root):
        vid_name = os.path.basename(vid_path)
        videos.append({"id": video_id, "name": vid_name})

        frame_to_imgid: Dict[int, int] = {}

        for f in iter_frames(vid_path):
            base = os.path.splitext(os.path.basename(f))[0]
            try:
                frame_idx = int(base)
            except ValueError:
                frame_idx = len(frame_to_imgid) + 1

            with Image.open(f) as im:
                w, h = im.size

            rel_file = os.path.join(rel_prefix, "sequences", vid_name, os.path.basename(f))
            images.append({
                "id": image_id,
                "file_name": rel_file.replace("\\", "/"),
                "width": int(w),
                "height": int(h),
                "frame_id": int(frame_idx),
                "video_id": int(video_id),
            })
            frame_to_imgid[frame_idx] = image_id
            image_id += 1

        ann_file = os.path.join(ann_dir_root, f"{vid_name}.txt")
        if os.path.exists(ann_file):
            with open(ann_file, "r") as fh:
                for line in fh:
                    parts = line.strip().split(",")
                    # frame_id, target_id, x, y, w, h, score, category, truncation, occlusion
                    if len(parts) < 10:
                        continue
                    try:
                        frame_idx = int(parts[0])
                        x, y, w, h = map(float, parts[2:6])
                        cid = int(parts[7])
                    except Exception:
                        continue

                    if cid in ignored_ids or cid not in categories:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    if frame_idx not in frame_to_imgid:
                        continue

                    anns.append({
                        "id": ann_id,
                        "image_id": frame_to_imgid[frame_idx],
                        "category_id": cid,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    })
                    ann_id += 1

        video_id += 1

    categories_list = [{"id": k, "name": v} for k, v in sorted(categories.items())]
    return {"videos": videos, "images": images, "annotations": anns, "categories": categories_list}


def verify_paths(json_path: str, vid_path: str, limit: int = 5) -> bool:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to open {json_path}: {e}")
        return False
    ok = True
    base = os.path.join(vid_path, "Data", "VID")
    imgs = data.get("images", [])
    for img in imgs[:limit]:
        p = os.path.join(base, img["file_name"])
        if not os.path.exists(p):
            print(f"Missing: {p}")
            ok = False
    if ok:
        print(f"Verified first {min(limit, len(imgs))} image paths for {json_path}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Prepare VisDrone-VID for TransVOD++ (copy/symlink + COCO-VID JSON).")
    ap.add_argument("--visdrone-train", required=True, help="Path to VisDrone2019-VID-train (sequences/, annotations/)")
    ap.add_argument("--visdrone-val", required=True, help="Path to VisDrone2019-VID-val (sequences/, annotations/)")
    ap.add_argument("--out-root", required=True, help="Output dataset root used as --vid_path in TransVOD++")
    ap.add_argument("--categories-json", default=None, help="Optional JSON mapping of class ids to names")
    ap.add_argument("--keep-others", action="store_true", help="Keep class id 11 ('others')")
    ap.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "hardlink", "none"],
        default="symlink",
        help="How to place images under out-root/Data/VID (default: symlink)",
    )
    ap.add_argument("--force", action="store_true", help="Force replace existing destination paths when linking/copying")
    ap.add_argument(
        "--fallback",
        choices=["hardlink", "copy", "none"],
        default="copy",
        help="Fallback strategy if link-mode fails (default: copy)",
    )
    ap.add_argument("--verify", action="store_true", help="Verify a few resolved image paths after conversion")
    args = ap.parse_args()

    # Ensure out_root base exists
    out_root = Path(args.out_root)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    (out_root / "Data" / "VID").mkdir(parents=True, exist_ok=True)

    categories, ignored_ids = load_categories(args.categories_json, args.keep_others)

    # Prepare train placement
    rel_train_prefix = os.path.join("VisDrone-VID", "train")
    src_train_seq = os.path.join(args.visdrone_train, "sequences")
    dst_train_seq = os.path.join(args.out_root, "Data", "VID", rel_train_prefix, "sequences")
    if args.link_mode != "none":
        print(f"Placing train images via {args.link_mode} ...")
        link_or_copy_sequences(src_train_seq, dst_train_seq, args.link_mode, force=args.force, fallback=args.fallback)

    print("Converting train split ...")
    train_json = parse_split(
        vis_split_root=args.visdrone_train,
        rel_prefix=rel_train_prefix,
        categories=categories,
        ignored_ids=ignored_ids,
    )
    train_out = os.path.join(args.out_root, "annotations", "imagenet_vid_train.json")
    with open(train_out, "w") as f:
        json.dump(train_json, f)
    print(f"Wrote {train_out}  images={len(train_json['images'])}  anns={len(train_json['annotations'])}")

    # Prepare val placement
    rel_val_prefix = os.path.join("VisDrone-VID", "val")
    src_val_seq = os.path.join(args.visdrone_val, "sequences")
    dst_val_seq = os.path.join(args.out_root, "Data", "VID", rel_val_prefix, "sequences")
    if args.link_mode != "none":
        print(f"Placing val images via {args.link_mode} ...")
        link_or_copy_sequences(src_val_seq, dst_val_seq, args.link_mode, force=args.force, fallback=args.fallback)

    print("Converting val split ...")
    val_json = parse_split(
        vis_split_root=args.visdrone_val,
        rel_prefix=rel_val_prefix,
        categories=categories,
        ignored_ids=ignored_ids,
    )
    val_out = os.path.join(args.out_root, "annotations", "imagenet_vid_val.json")
    with open(val_out, "w") as f:
        json.dump(val_json, f)
    print(f"Wrote {val_out}  images={len(val_json['images'])}  anns={len(val_json['annotations'])}")

    if args.verify:
        ok_train = verify_paths(train_out, args.out_root, limit=5)
        ok_val = verify_paths(val_out, args.out_root, limit=5)
        if not (ok_train and ok_val):
            print("WARNING: Some image paths did not resolve. Check your placement/link-mode.")

    print("Done. Use --vid_path:", args.out_root)


if __name__ == "__main__":
    main()
