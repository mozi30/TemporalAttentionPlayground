#!/usr/bin/env python3
"""
build_ilsvrc_structure.py

Build ILSVRC-VID folder STRUCTURE (images + XMLs) from VisDrone or UAVDT,
without enforcing ILSVRC2015 names (only the hierarchy).

ILSVRC/
├── Data/VID/{train, val, test}/
│   ├── train: train_0000/<seq>/frame.JPEG ...
│   ├── val:   <seq>/frame.JPEG ...
│   └── test:  <seq>/frame.JPEG ...
├── Annotations/VID/{train, val, test}/
│   ├── mirrors Data/VID with XMLs if available (test usually empty)
└── ImageSets/VID/{train.txt, val.txt, test.txt}

Auto-fixes:
- Detects annotations root by searching for 'Annotations-ImageNetVID*' under --root.
- Flattens nested 'train/train' or 'val/val' inside the annotations root if present.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional

def log(msg: str): print(msg, flush=True)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def safe_link_or_copy(src: Path, dst: Path, prefer_symlink: bool):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        if prefer_symlink:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        else:
            raise OSError("copy requested")
    except Exception:
        shutil.copy2(src, dst)

def list_sequences(dirpath: Optional[Path]) -> List[Path]:
    if not dirpath or not dirpath.exists():
        return []
    return sorted([p for p in dirpath.iterdir() if p.is_dir()])

def remove_if_exists(path: Path):
    if not path.exists(): return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)

def link_seq_files(src_seq: Path, dst_seq: Path, prefer_symlink: bool, pattern: str):
    ensure_dir(dst_seq)
    for f in sorted(src_seq.glob(pattern)):
        if f.is_file():
            safe_link_or_copy(f, dst_seq / f.name, prefer_symlink)

def flatten_if_double_nested(root: Path, split: str):
    """If annotations have accidental split/split nesting, flatten it."""
    s = root / split
    inner = s / split
    if inner.exists() and inner.is_dir():
        log(f"[fix] Flattening nested {split}/{split} under {root}")
        for item in inner.iterdir():
            shutil.move(str(item), str(s / item.name))
        try:
            inner.rmdir()
        except OSError:
            pass

def detect_annotations_root(root: Path, dataset: str) -> Optional[Path]:
    """
    Try common names, else scan *one level* for 'Annotations-ImageNetVID*'.
    - VisDrone: Annotations-ImageNetVID
    - UAVDT:    Annotations-ImageNetVID-UAVDT
    """
    preferred = {
        "visdrone": root / "Annotations-ImageNetVID",
        "uavdt":    root / "Annotations-ImageNetVID-UAVDT",
    }
    cand = preferred[dataset]
    if cand.exists():
        return cand
    hits = [p for p in root.glob("Annotations-ImageNetVID*") if p.is_dir()]
    if hits:
        # choose the one that contains either 'train' or 'val'
        for h in hits:
            if (h / "train").exists() or (h / "val").exists():
                return h
        return hits[0]
    return None

def build_split(
    split: str,
    src_img_root: Optional[Path],
    src_xml_root: Optional[Path],
    out_data_split: Path,
    out_ann_split: Path,
    imagesets_file: Path,
    prefer_symlink: bool,
    force: bool,
    group_size: int,
):
    ensure_dir(out_data_split)
    ensure_dir(out_ann_split)
    ensure_dir(imagesets_file.parent)

    seq_dirs = list_sequences(src_img_root)
    if not seq_dirs:
        log(f"[warn] No image sequences for {split} at {src_img_root}; writing empty list.")
        with open(imagesets_file, "w") as f:
            pass
        return

    lines: List[str] = []
    for idx, seq_dir in enumerate(seq_dirs):
        if split == "train":
            group = idx // group_size
            group_name = f"train_{group:04d}"
            rel_seq_path = Path(group_name) / seq_dir.name
        else:
            rel_seq_path = Path(seq_dir.name)

        dst_seq_img = out_data_split / rel_seq_path
        dst_seq_xml = out_ann_split / rel_seq_path

        # Skip or rebuild
        if (dst_seq_img.exists() or dst_seq_xml.exists()) and not force:
            lines.append(str(rel_seq_path).replace("\\", "/"))
            continue
        if force:
            remove_if_exists(dst_seq_img)
            remove_if_exists(dst_seq_xml)

        # Link images (accept common extensions)
        link_seq_files(seq_dir, dst_seq_img, prefer_symlink, pattern="*.*")

        # Link XMLs for train/val if available
        if src_xml_root and src_xml_root.exists() and split in ("train", "val"):
            src_xml_seq = src_xml_root / seq_dir.name
            if src_xml_seq.exists():
                link_seq_files(src_xml_seq, dst_seq_xml, prefer_symlink, pattern="*.xml")
            else:
                ensure_dir(dst_seq_xml)  # keep structure even if no xml for this seq
        else:
            ensure_dir(dst_seq_xml)

        lines.append(str(rel_seq_path).replace("\\", "/"))

    with open(imagesets_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    log(f"[ok] {split}: {len(lines)} sequences → {out_data_split}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "uavdt"], required=True)
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root (e.g., ./datasets/visdrone or ./datasets/uavdt)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking (default symlink).")
    ap.add_argument("--group-size", type=int, default=100,
                    help="Train sequences per group folder (default 100).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild sequences even if destination exists.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    prefer_symlink = not args.copy

    # Image roots
    if args.dataset == "visdrone":
        img_roots = {
            "train": root / "VisDrone2019-VID-train" / "sequences",
            "val":   root / "VisDrone2019-VID-val" / "sequences",
            "test":  root / "VisDrone2019-VID-test" / "sequences",
        }
    else:
        img_roots = {
            "train": root / "UAVDT-train" / "sequences",
            "val":   root / "UAVDT-val" / "sequences",
            "test":  None,  # usually none for UAVDT
        }

    # Detect annotations root and fix common nesting mistakes
    ann_root = detect_annotations_root(root, args.dataset)
    if ann_root:
        for sp in ("train", "val"):
            flatten_if_double_nested(ann_root, sp)
        xml_roots = {
            "train": ann_root / "train" if (ann_root / "train").exists() else None,
            "val":   ann_root / "val"   if (ann_root / "val").exists()   else None,
            "test":  ann_root / "test"  if (ann_root / "test").exists()  else None,
        }
    else:
        log(f"[warn] Could not locate annotations root under {root}; proceeding without XMLs.")
        xml_roots = {"train": None, "val": None, "test": None}

    # Dest roots
    ilsvrc_root = root / "ILSVRC"
    data_vid = ilsvrc_root / "Data" / "VID"
    ann_vid  = ilsvrc_root / "Annotations" / "VID"
    imagesets = ilsvrc_root / "ImageSets" / "VID"

    # Build splits
    build_split(
        "train",
        img_roots.get("train"),
        xml_roots.get("train"),
        data_vid / "train",
        ann_vid / "train",
        imagesets / "train.txt",
        prefer_symlink,
        args.force,
        args.group_size,
    )
    build_split(
        "val",
        img_roots.get("val"),
        xml_roots.get("val"),
        data_vid / "val",
        ann_vid / "val",
        imagesets / "val.txt",
        prefer_symlink,
        args.force,
        args.group_size,
    )
    build_split(
        "test",
        img_roots.get("test"),
        xml_roots.get("test"),
        data_vid / "test",
        ann_vid / "test",
        imagesets / "test.txt",
        prefer_symlink,
        args.force,
        args.group_size,
    )

    log(f"[done] ILSVRC structure built at: {ilsvrc_root}")

if __name__ == "__main__":
    main()
