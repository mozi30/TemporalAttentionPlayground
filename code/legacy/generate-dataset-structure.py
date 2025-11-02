#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

Datasets = ["visdrone", "uavdt"]

def _bool_flag(args_list: list[str], flag: str, enabled: bool):
    if enabled:
        args_list.append(flag)

def visdroneToImagenetVID(root: Path, dataset: str, category_map_visdrone: Path,
                          no_read_size: bool, assume_perframe: bool, frame_digits: int):
    if dataset != "visdrone":
        return

    expected = [
        root/"VisDrone2019-VID-train",
        root/"VisDrone2019-VID-val",
        root/"VisDrone2019-VID-test-dev",
    ]
    if not all(p.exists() for p in expected):
        print("Error: VisDrone-VID dataset not found in", root)

    def _run(args): subprocess.run(args, check=True)

    # Train
    out_train = root/"Annotations-ImageNetVID"/"train"
    if not out_train.exists():
        args = [
            "python3", "visdrone_vid2imagenet_vid.py",
            "--ann-root", str(root/"VisDrone2019-VID-train"/"annotations"),
            "--img-root",  str(root/"VisDrone2019-VID-train"/"sequences"),
            "--out-root",  str(out_train),
            "--category-map", str(category_map_visdrone),
            "--ignore-class-ids", "0",
            "--split", "train",
            "--frame-digits", str(frame_digits),
        ]
        _bool_flag(args, "--no-read-size", no_read_size)
        _bool_flag(args, "--assume-perframe", assume_perframe)
        _run(args)

    # Val
    out_val = root/"Annotations-ImageNetVID"/"val"
    if not out_val.exists():
        args = [
            "python3", "visdrone_vid2imagenet_vid.py",
            "--ann-root", str(root/"VisDrone2019-VID-val"/"annotations"),
            "--img-root",  str(root/"VisDrone2019-VID-val"/"sequences"),
            "--out-root",  str(out_val),
            "--category-map", str(category_map_visdrone),
            "--ignore-class-ids", "0",
            "--split", "val",
            "--frame-digits", str(frame_digits),
        ]
        _bool_flag(args, "--no-read-size", no_read_size)
        _bool_flag(args, "--assume-perframe", assume_perframe)
        _run(args)

    # Test
    out_test = root/"Annotations-ImageNetVID"/"test"
    if not out_test.exists():
        args = [
            "python3", "visdrone_vid2imagenet_vid.py",
            "--ann-root", str(root/"VisDrone2019-VID-test-dev"/"annotations"),
            "--img-root",  str(root/"VisDrone2019-VID-test"/"sequences"),
            "--out-root",  str(out_test),
            "--category-map", str(category_map_visdrone),
            "--ignore-class-ids", "0",
            "--split", "test",
            "--frame-digits", str(frame_digits),
        ]
        _bool_flag(args, "--no-read-size", no_read_size)
        _bool_flag(args, "--assume-perframe", assume_perframe)
        _run(args)


def uavdtToImagenetVID(root: Path, dataset: str, category_map_uavdt: Path,
                       ignore_ids: list[int] | None,
                       no_read_size: bool, frame_digits: int):
    if dataset != "uavdt":
        return
    ignore_ids = ignore_ids or []

    expected = [root/"UAVDT-train", root/"UAVDT-val"]
    if not all(p.exists() for p in expected):
        print("Error: UAVDT dataset not found under", root)

    def _ignore_args():
        out = []
        for cid in ignore_ids:
            out += ["--ignore-class-ids", str(cid)]
        return out

    def _run(args): subprocess.run(args, check=True)

    # Train
    out_train = root/"Annotations-ImageNetVID-UAVDT"/"train"
    if not out_train.exists():
        args = [
            "python3", "visdrone_vid2imagenet_vid.py",
            "--ann-root", str(root/"UAVDT-train"/"annotations"),
            "--img-root",  str(root/"UAVDT-train"/"sequences"),
            "--out-root",  str(out_train),
            "--category-map", str(category_map_uavdt),
            "--split", "train",
            "--frame-digits", str(frame_digits),
            "--force-uavdt",
        ] + _ignore_args()
        _bool_flag(args, "--no-read-size", no_read_size)
        _run(args)

    # Val
    out_val = root/"Annotations-ImageNetVID-UAVDT"/"val"
    if not out_val.exists():
        args = [
            "python3", "visdrone_vid2imagenet_vid.py",
            "--ann-root", str(root/"UAVDT-val"/"annotations"),
            "--img-root",  str(root/"UAVDT-val"/"sequences"),
            "--out-root",  str(out_val),
            "--category-map", str(category_map_uavdt),
            "--split", "val",
            "--frame-digits", str(frame_digits),
            "--force-uavdt",
        ] + _ignore_args()
        _bool_flag(args, "--no-read-size", no_read_size)
        _run(args)

def imagenetVIDToCOCO(root: Path, dataset: str, category_map_coco: Path):
    if(dataset == "visdrone"):
        arguments_train = "--ann-root " + str(root/"Annotations-ImageNetVID"/"train") + " --img-root " + str(root/"VisDrone2019-VID-train"/"sequences") + " --out " + str(root/"COCO-VID"/"train.json") + " --categories " + str(category_map_coco)
        subprocess.run(["python3", "vid2coco.py"] + arguments_train.split(" "))
        arguments_val = "--ann-root " + str(root/"Annotations-ImageNetVID"/"val") + " --img-root " + str(root/"VisDrone2019-VID-val"/"sequences") + " --out " + str(root/"COCO-VID"/"val.json") + " --categories " + str(category_map_coco)
        subprocess.run(["python3", "vid2coco.py"] + arguments_val.split(" "))
        arguments_test = "--ann-root " + str(root/"Annotations-ImageNetVID"/"test") + " --img-root " + str(root/"VisDrone2019-VID-test"/"sequences") + " --out " + str(root/"COCO-VID"/"test.json") + " --categories " + str(category_map_coco)
        subprocess.run(["python3", "vid2coco.py"] + arguments_test.split(" "))
    elif(dataset == "uavdt"):
        # Note: using the UAVDT-specific ImageNetVID directory we wrote above
        arguments_train = "--ann-root " + str(root/"Annotations-ImageNetVID-UAVDT"/"train") + " --img-root " + str(root/"UAVDT-train"/"sequences") + " --out " + str(root/"COCO-VID-UAVDT"/"train.json") + " --categories " + str(category_map_coco)
        subprocess.run(["python3", "vid2coco.py"] + arguments_train.split(" "))
        arguments_val = "--ann-root " + str(root/"Annotations-ImageNetVID-UAVDT"/"val") + " --img-root " + str(root/"UAVDT-val"/"sequences") + " --out " + str(root/"COCO-VID-UAVDT"/"val.json") + " --categories " + str(category_map_coco)
        subprocess.run(["python3", "vid2coco.py"] + arguments_val.split(" "))
    else:
        print("Dataset not supported yet.")
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=Datasets, required=True)
    ap.add_argument("--category-map-vd-to-ivid", type=str, required=True)
    ap.add_argument("--category-map-ivid-to-coco", type=str, required=True)
    ap.add_argument("--ignore-class-ids", type=int, nargs='*', default=[])
    ap.add_argument("--no-read-size", action="store_true")
    ap.add_argument("--assume-perframe", action="store_true")
    ap.add_argument("--frame-digits", type=int, default=7, help="Force same zero-padding for all datasets")
    args = ap.parse_args()

    folder_root = Path(args.root)
    dataset = args.dataset
    category_map_visdrone = Path(args.category_map_vd_to_ivid)
    category_map_coco = Path(args.category_map_ivid_to_coco)
    ignore_ids = args.ignore_class_ids

    if dataset == "visdrone":
        visdroneToImagenetVID(folder_root, dataset, category_map_visdrone,
                              args.no_read_size, args.assume_perframe, args.frame_digits)
        
    elif dataset == "uavdt":
        uavdtToImagenetVID(folder_root, dataset, category_map_visdrone,
                           ignore_ids, args.no_read_size, args.frame_digits)
        

    imagenetVIDToCOCO(folder_root, dataset, category_map_coco)
