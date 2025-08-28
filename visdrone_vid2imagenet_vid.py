#!/usr/bin/env python3
"""
visdrone_vid2imagenet_vid.py  (now with UAVDT DET support)

Converts:
- VisDrone-VID (MOT-style or per-frame .txt)
- UAVDT DET groundtruth files named like: M0101_gt_whole.txt

…into ImageNet VID-style Pascal VOC XML (one XML per frame), preserving track IDs.

Usage (UAVDT):
python visdrone_vid2imagenet_vid.py \
  --ann-root /path/to/UAVDT/GT           # folder containing M####_gt_whole.txt files
  --img-root /path/to/UAVDT/sequences    # folder containing M####/0000001.jpg ...
  --out-root /path/to/out_xml \
  --split train \
  --category-map default \
  --frame-digits 7   # adjust to your padding (6/7/8), or omit

See the module docstring for VisDrone usage.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

try:
    from PIL import Image
except Exception:
    Image = None


# ------------------------------- Category mapping -------------------------------

def load_category_mapper(mode: str):
    """
    mode:
      - 'default' -> names like 'cls_<id>'
      - path to JSON -> map str(id) or int id to class name
    """
    if mode == 'default':
        def _mapper(cid):
            return f"cls_{cid}"
        return _mapper
    with open(mode, 'r') as f:
        data = json.load(f)
    def _mapper(cid):
        key = str(cid) if str(cid) in data else (cid if cid in data else str(cid))
        return data.get(key, f"cls_{cid}")
    return _mapper


# ------------------------------- Discovery --------------------------------------

def list_sequences_generic(ann_root: Path):
    """
    For VisDrone-style layouts (kept from your original):
    - a directory per sequence with gt.txt (MOT), or per-frame .txt files
    - or a single .txt per sequence directly in ann_root

    Returns entries: (sequence_name, 'unknown', path)
    """
    seqs = []
    for p in sorted(ann_root.iterdir()):
        if p.is_dir():
            if any(p.glob("*.txt")):
                seqs.append((p.name, 'unknown', p))
        elif p.is_file() and p.suffix == '.txt':
            seqs.append((p.stem, 'unknown', p))
    return seqs


def list_sequences_uavdt_whole(ann_root: Path):
    """
    UAVDT DET GT layout: ann_root contains files like M0101_gt_whole.txt
    Returns entries: (sequence_name, 'uavdt_det', file_path)
    """
    seqs = []
    for f in sorted(ann_root.glob("*_gt_whole.txt")):
        seq = f.name.split("_gt_whole.txt")[0]
        if seq:
            seqs.append((seq, 'uavdt_det', f))
    return seqs


def detect_format_for_sequence(seq_entry):
    """
    For VisDrone-only flows: peek to decide mot vs perframe.
    UAVDT is already labeled 'uavdt_det' upstream and bypasses this.
    """
    name, typ, path = seq_entry
    if typ == 'uavdt_det':
        return seq_entry

    # Case A: file: treat as MOT-style per-sequence file
    if path.is_file():
        return (name, 'mot', path)

    # Case B: directory
    gt = path / 'gt.txt'
    if gt.exists():
        return (name, 'mot', gt)

    frame_txts = sorted(path.glob("*.txt"))
    if not frame_txts:
        return (name, 'unknown', path)
    with open(frame_txts[0], 'r') as f:
        line = f.readline().strip()
    if not line:
        return (name, 'perframe', path)
    cols = [c.strip() for c in line.split(',')]
    if len(cols) in (8, 9):   # per-frame (no track id)
        return (name, 'perframe', path)
    return (name, 'mot', frame_txts[0])


# ------------------------------- IO helpers -------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_voc_xml(out_xml_path: Path, image_info, objects):
    """
    image_info: dict(file_name, width, height, folder)
    objects: list of dict(name, bbox=[xmin,ymin,xmax,ymax], difficult, trackid)
    """
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = image_info.get('folder', '')
    ET.SubElement(annotation, 'filename').text = image_info['file_name']

    size = ET.SubElement(annotation, 'size')
    if image_info.get('width') is not None:
        ET.SubElement(size, 'width').text = str(int(image_info['width']))
        ET.SubElement(size, 'height').text = str(int(image_info['height']))
        ET.SubElement(size, 'depth').text = str(image_info.get('depth', 3))
    else:
        ET.SubElement(size, 'width').text = '0'
        ET.SubElement(size, 'height').text = '0'
        ET.SubElement(size, 'depth').text = '3'

    for obj in objects:
        o = ET.SubElement(annotation, 'object')
        ET.SubElement(o, 'name').text = obj['name']
        ET.SubElement(o, 'pose').text = 'Unspecified'
        ET.SubElement(o, 'truncated').text = '0'
        ET.SubElement(o, 'difficult').text = str(int(obj.get('difficult', 0)))
        ET.SubElement(o, 'trackid').text = str(int(obj.get('trackid', -1)))
        b = ET.SubElement(o, 'bndbox')
        xmin, ymin, xmax, ymax = obj['bbox']
        ET.SubElement(b, 'xmin').text = str(int(round(xmin)))
        ET.SubElement(b, 'ymin').text = str(int(round(ymin)))
        ET.SubElement(b, 'xmax').text = str(int(round(xmax)))
        ET.SubElement(b, 'ymax').text = str(int(round(ymax)))

    tree = ET.ElementTree(annotation)
    ensure_dir(out_xml_path.parent)
    tree.write(out_xml_path, encoding='utf-8')


def read_image_size(img_root: Path, split: str, seq_name: str, frame_name: str):
    if Image is None:
        return None, None
    # Try split-aware path
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = img_root / split / seq_name / (frame_name + ext)
        if p.exists():
            with Image.open(p) as im:
                return im.width, im.height
    # Try flat path (no split)
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = img_root / seq_name / (frame_name + ext)
        if p.exists():
            with Image.open(p) as im:
                return im.width, im.height
    return None, None


# ------------------------------- Converters -------------------------------------

def process_mot_sequence(seq_name, ann_file: Path, split, img_root: Path, out_root: Path,
                         cat_mapper, ignore_ids, no_read_size,
                         frame_digits=7,
                         difficult_if_score0=True, difficult_occ_ge=2, difficult_trunc_ge=2):
    """
    Generic MOT-style:
    frame_id, target_id, left, top, width, height, score, category_id, trunc, occl
    """
    frames = defaultdict(list)
    with open(ann_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                row = [c.strip() for c in row[0].split(',')]
            try:
                frame_id = int(float(row[0]))
                target_id = int(float(row[1]))
                left = float(row[2]); top = float(row[3])
                width = float(row[4]); height = float(row[5])
                score = float(row[6]) if len(row) > 6 else 1.0
                category_id = int(float(row[7])) if len(row) > 7 else 1
                trunc = int(float(row[8])) if len(row) > 8 else 0
                occl = int(float(row[9])) if len(row) > 9 else 0
            except Exception:
                continue

            if category_id in ignore_ids or target_id == -1:
                continue

            name = cat_mapper(category_id)
            xmin, ymin = left, top
            xmax, ymax = left + width, top + height

            difficult = 0
            if difficult_if_score0 and score <= 0:
                difficult = 1
            if occl is not None and occl >= difficult_occ_ge:
                difficult = 1
            if trunc is not None and trunc >= difficult_trunc_ge:
                difficult = 1

            frames[frame_id].append({
                "name": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "difficult": difficult,
                "trackid": max(0, target_id)
            })

    for frame_id, objs in frames.items():
        frame_name = f"{frame_id:0{frame_digits}d}"
        file_name = f"{frame_name}.JPEG"
        width = height = None
        if not no_read_size and img_root is not None:
            w, h = read_image_size(img_root, split, seq_name, frame_name)
            width, height = w, h

        out_dir = out_root / split / seq_name
        out_xml = out_dir / f"{frame_name}.xml"
        image_info = {"file_name": file_name, "width": width, "height": height, "folder": f"{split}/{seq_name}"}
        write_voc_xml(out_xml, image_info, objs)


def process_perframe_sequence(seq_name, ann_dir: Path, split, img_root: Path, out_root: Path,
                              cat_mapper, ignore_ids, no_read_size,
                              frame_digits=7,
                              difficult_if_score0=True, difficult_occ_ge=2, difficult_trunc_ge=2):
    """
    Per-frame TXT without track ids (VisDrone per-frame style).
    """
    txts = sorted(ann_dir.glob("*.txt"))
    for txt in txts:
        frame_name = txt.stem
        objs = []
        with open(txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [c.strip() for c in line.split(',')]
                if len(parts) < 8:
                    continue
                left, top, width, height = map(float, parts[:4])
                score = float(parts[4])
                category_id = int(float(parts[5]))
                trunc = int(float(parts[6]))
                occl = int(float(parts[7]))

                if category_id in ignore_ids:
                    continue

                name = cat_mapper(category_id)
                xmin, ymin = left, top
                xmax, ymax = left + width, top + height

                difficult = 0
                if difficult_if_score0 and score <= 0:
                    difficult = 1
                if occl >= difficult_occ_ge:
                    difficult = 1
                if trunc >= difficult_trunc_ge:
                    difficult = 1

                trackid = len(objs) + 1  # synthesized
                objs.append({"name": name, "bbox": [xmin, ymin, xmax, ymax], "difficult": difficult, "trackid": trackid})

        file_name = f"{frame_name}.JPEG"
        width = height = None
        if not no_read_size and img_root is not None:
            w, h = read_image_size(img_root, split, seq_name, frame_name)
            width, height = w, h

        out_dir = out_root / split / seq_name
        out_xml = out_dir / f"{frame_name}.xml"
        image_info = {"file_name": file_name, "width": width, "height": height, "folder": f"{split}/{seq_name}"}
        write_voc_xml(out_xml, image_info, objs)


def process_uavdt_det_sequence(seq_name, ann_file: Path, split, img_root: Path, out_root: Path,
                               cat_mapper, ignore_ids, no_read_size,
                               frame_digits=7,
                               difficult_out_ge=2, difficult_occ_ge=3):
    """
    UAVDT DET Groundtruth (*_gt_whole.txt):
      frame_index, target_id, left, top, width, height, out_of_view, occlusion, object_category

    Notes:
    - 'out_of_view': 1=no-out, 2=medium-out, 3=small-out  (spec text is a bit odd; treat >=2 as harder by default)
    - 'occlusion': 1=no, 2=large, 3=medium, 4=small      (we mark difficult if >=3 by default)
    """
    frames = defaultdict(list)
    with open(ann_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                row = [c.strip() for c in row[0].split(',')]
            if len(row) < 9:
                continue
            try:
                frame_id = int(float(row[0]))
                target_id = int(float(row[1]))
                left = float(row[2]); top = float(row[3])
                width = float(row[4]); height = float(row[5])
                out_view = int(float(row[6]))
                occl = int(float(row[7]))
                category_id = int(float(row[8]))
            except Exception:
                continue

            if category_id in ignore_ids or target_id == -1:
                continue

            name = cat_mapper(category_id)
            xmin, ymin = left, top
            xmax, ymax = left + width, top + height

            difficult = 0
            if out_view >= difficult_out_ge:
                difficult = 1
            if occl >= difficult_occ_ge:
                difficult = 1

            frames[frame_id].append({
                "name": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "difficult": difficult,
                "trackid": max(0, target_id)
            })

    for frame_id, objs in frames.items():
        frame_name = f"{frame_id:0{frame_digits}d}"
        file_name = f"{frame_name}.JPEG"
        width = height = None
        if not no_read_size and img_root is not None:
            w, h = read_image_size(img_root, split, seq_name, frame_name)
            width, height = w, h

        out_dir = out_root / split / seq_name
        out_xml = out_dir / f"{frame_name}.xml"
        image_info = {"file_name": file_name, "width": width, "height": height, "folder": f"{split}/{seq_name}"}
        write_voc_xml(out_xml, image_info, objs)


# ------------------------------- Main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-root", type=str, required=True,
                    help="Root containing annotations. For UAVDT, point to folder with *_gt_whole.txt. For VisDrone, an annotations split dir or a single seq file.")
    ap.add_argument("--img-root", type=str, required=False,
                    help="Images root (sequence folders). Used to read image sizes; optional.")
    ap.add_argument("--out-root", type=str, required=True,
                    help="Output root for ImageNet VID-style XMLs.")
    ap.add_argument("--split", type=str, default="train", help="Split name used in output (train/val/test).")
    ap.add_argument("--category-map", type=str, default="default", help="'default' or path to JSON mapping category_id->name.")
    ap.add_argument("--ignore-class-ids", type=int, nargs='*', default=[], help="IDs to ignore.")
    ap.add_argument("--no-read-size", action="store_true", help="Do not read image sizes from --img-root.")
    ap.add_argument("--assume-perframe", action="store_true", help="Force per-frame mode for VisDrone.")
    ap.add_argument("--frame-digits", type=int, default=7, help="Zero-padding for frame filenames, e.g., 6, 7, or 8.")
    ap.add_argument("--force-uavdt", action="store_true", help="Treat inputs as UAVDT DET regardless of file names.")
    args = ap.parse_args()

    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)
    img_root = Path(args.img_root) if args.img_root else None

    cat_mapper = load_category_mapper(args.category_map)
    ignore_ids = set(args.ignore_class_ids or [])

    seqs = []
    if not ann_root.exists():
        raise SystemExit(f"Annotation root not found: {ann_root}")

    # Prefer UAVDT detection if *_gt_whole.txt files are present OR user forces it
    uavdt_entries = list_sequences_uavdt_whole(ann_root) if ann_root.is_dir() else []
    if args.force_uavdt:
        if ann_root.is_file():
            # single file (assume name like M0101_gt_whole.txt)
            stem = ann_root.name.replace("_gt_whole.txt", "")
            seqs = [(stem, 'uavdt_det', ann_root)]
        else:
            seqs = uavdt_entries
        if not seqs:
            raise SystemExit("No UAVDT *_gt_whole.txt files found (and --force-uavdt was set).")
    elif uavdt_entries:
        seqs = uavdt_entries
    else:
        # Fall back to VisDrone-style discovery
        if ann_root.is_dir():
            for s in list_sequences_generic(ann_root):
                seqs.append(detect_format_for_sequence(s))
        elif ann_root.is_file():
            seqs.append(detect_format_for_sequence((ann_root.stem, 'unknown', ann_root)))

    if not seqs:
        raise SystemExit("No sequences detected to process.")

    for seq_name, ftype, path in seqs:
        if ftype == 'uavdt_det':
            print(f"[info] UAVDT DET: {seq_name}")
            process_uavdt_det_sequence(seq_name, path, args.split, img_root, out_root,
                                       cat_mapper, ignore_ids, args.no_read_size,
                                       frame_digits=args.frame_digits)
        elif args.assume_perframe:
            print(f"[info] Per-frame (forced): {seq_name}")
            ann_dir = path if path.is_dir() else path.parent
            process_perframe_sequence(seq_name, ann_dir, args.split, img_root, out_root,
                                      cat_mapper, ignore_ids, args.no_read_size,
                                      frame_digits=args.frame_digits)
        elif ftype == 'mot':
            print(f"[info] MOT-style: {seq_name}")
            ann_file = path if path.is_file() else (path / 'gt.txt')
            if not ann_file.exists():
                candidate = path / f"{seq_name}.txt"
                if candidate.exists():
                    ann_file = candidate
                else:
                    print(f"[warn] MOT file not found for seq {seq_name}, skipping.")
                    continue
            process_mot_sequence(seq_name, ann_file, args.split, img_root, out_root,
                                 cat_mapper, ignore_ids, args.no_read_size,
                                 frame_digits=args.frame_digits)
        elif ftype == 'perframe':
            print(f"[info] Per-frame: {seq_name}")
            ann_dir = path if path.is_dir() else path.parent
            process_perframe_sequence(seq_name, ann_dir, args.split, img_root, out_root,
                                      cat_mapper, ignore_ids, args.no_read_size,
                                      frame_digits=args.frame_digits)
        else:
            print(f"[warn] Unknown annotation format for sequence {seq_name}; skipping.")

    print(f"[done] Wrote ImageNet VID-style XMLs under: {out_root}")


if __name__ == "__main__":
    main()
