#!/usr/bin/env python3
"""
build_coco_structure.py

Create a COCO-style layout *without year suffix*:

coco/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json    # only if test split exists
├── train/
├── val/
└── test/

Sources (auto-detected):
- VisDrone:
    JSON:  <root>/COCO-VID/{train.json,val.json,test.json} (or legacy per-split dirs)
    IMGs:  <root>/VisDrone2019-VID-{train,val,test}/sequences/<video>/<frame>
- UAVDT:
    JSON:  <root>/COCO-VID-UAVDT/{train.json,val.json} (or legacy per-split dirs)
    IMGs:  <root>/UAVDT-{train,val}/sequences/<video>/<frame>

Usage
-----
# VisDrone
python build_coco_structure.py --dataset visdrone --root ./datasets/visdrone --out ./datasets/visdrone/coco

# UAVDT
python build_coco_structure.py --dataset uavdt --root ./datasets/uavdt --out ./datasets/uavdt/coco
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Optional, Dict

def log(msg: str): print(msg, flush=True)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def safe_link_or_copy(src: Path, dst: Path, prefer_symlink: bool):
    ensure_dir(dst.parent)
    if dst.exists(): return
    try:
        if prefer_symlink:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        else:
            raise OSError("copy requested")
    except Exception:
        shutil.copy2(src, dst)

def first_json(dirpath: Path) -> Optional[Path]:
    cand = dirpath / "_annotations.coco.json"
    if cand.exists(): return cand
    js = sorted(dirpath.glob("*.json"))
    return js[0] if js else None

def resolve_src_json(coco_root: Path, split: str, flat_name: str) -> Optional[Path]:
    p = coco_root / flat_name
    if p.exists(): return p
    legacy = coco_root / split
    if legacy.exists():
        j = first_json(legacy)
        if j: return j
    return None

def mirror_split(
    src_json: Path,
    img_root: Path,
    out_split_dir: Path,
    out_ann_json: Path,
    prefer_symlink: bool,
    force: bool
):
    if out_ann_json.exists() and out_split_dir.exists() and not force:
        log(f"[skip] {out_ann_json} already exists")
        return

    ensure_dir(out_split_dir)
    ensure_dir(out_ann_json.parent)

    with open(src_json, "r") as f:
        data = json.load(f)

    # normalize file_name with video_name if needed; mirror images
    total = 0
    for im in data.get("images", []):
        fn = Path(im["file_name"])
        vname = im.get("video_name")
        frame_idx = im.get("frame_index")
        rel = fn

        if fn.parent == Path("."):
            if not vname:
                raise KeyError(f"image id={im.get('id')} missing 'video_name' in {src_json}")
            im["file_name"] = f"{vname}/{fn.name}"
            rel = Path(im["file_name"])

        src = img_root / rel
        if not src.exists():
            # extension fallbacks
            for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".bmp"):
                alt = src.with_suffix(ext)
                if alt.exists():
                    src = alt
                    break
        if not src.exists() and vname and frame_idx is not None:
            # fallback: reconstruct from frame_index (7 digits), keep original suffix
            ext = fn.suffix if fn.suffix else ".jpg"
            rebuilt = Path(vname) / f"{int(frame_idx):07d}{ext}"
            cand = img_root / rebuilt
            if cand.exists():
                src = cand
                im["file_name"] = str(rebuilt)
                rel = rebuilt

        if not src.exists():
            raise FileNotFoundError(f"Image not found for: {img_root/rel}")

        dst = out_split_dir / rel
        safe_link_or_copy(src, dst, prefer_symlink)
        total += 1

    with open(out_ann_json, "w") as f:
        json.dump(data, f)
    log(f"[ok] wrote {out_ann_json} and mirrored {total} images to {out_split_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "uavdt"], required=True)
    ap.add_argument("--root", type=str, required=True, help="Dataset root")
    ap.add_argument("--out", type=str, required=True, help="Output COCO folder (will contain annotations/, train/, val/, test/)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink (default symlink)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    prefer_symlink = not args.copy

    # locate sources
    if args.dataset == "visdrone":
        coco_src = root / "COCO-VID"
        img_roots: Dict[str, Path] = {
            "train": root / "VisDrone2019-VID-train" / "sequences",
            "val":   root / "VisDrone2019-VID-val" / "sequences",
            "test":  root / "VisDrone2019-VID-test" / "sequences",
        }
        plan = [
            ("train", "instances_train.json", "train.json"),
            ("val",   "instances_val.json",   "val.json"),
            ("test",  "instances_test.json",  "test.json"),
        ]
    else:  # uavdt
        coco_src = root / "COCO-VID-UAVDT"
        img_roots = {
            "train": root / "UAVDT-train" / "sequences",
            "val":   root / "UAVDT-val" / "sequences",
        }
        plan = [
            ("train", "instances_train.json", "train.json"),
            ("val",   "instances_val.json",   "val.json"),
            # no official test; create only if a JSON exists
            ("test",  "instances_test.json",  "test.json"),
        ]

    # build structure
    ann_dir = out_root / "annotations"
    train_dir = out_root / "train"
    val_dir   = out_root / "val"
    test_dir  = out_root / "test"
    for d in (ann_dir, train_dir, val_dir, test_dir):
        ensure_dir(d)

    for split, ann_name, flat_name in plan:
        src_json = resolve_src_json(coco_src, split, flat_name)
        if not src_json:
            log(f"[skip] no source COCO json for split '{split}'")
            continue
        img_root = img_roots.get(split)
        if img_root is None or not img_root.exists():
            log(f"[skip] no image root for split '{split}' at {img_root}")
            continue

        out_split_dir = {"train": train_dir, "val": val_dir, "test": test_dir}[split]
        out_ann_json  = ann_dir / ann_name

        mirror_split(
            src_json=src_json,
            img_root=img_root,
            out_split_dir=out_split_dir,
            out_ann_json=out_ann_json,
            prefer_symlink=prefer_symlink,
            force=args.force,
        )

    log(f"[done] COCO structure ready at: {out_root}")

if __name__ == "__main__":
    main()
