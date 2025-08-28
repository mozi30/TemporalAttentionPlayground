#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str]):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    repo = Path(__file__).resolve().parent

    # Optional download steps (uncomment if you use them)
    # run(["python3", str(repo / "downloadVisdrone.py"), "--root", "datasets/visdrone"])
    # run(["python3", str(repo / "downloadUavdt.py"), "--root", "datasets/uavdt"])
    
    # --- VisDrone: generate structure ---
    run([
        "python3", str(repo / "generate-dataset-structure.py"),
        "--root", "datasets/visdrone",
        "--dataset", "visdrone",
        "--category-map-vd-to-ivid", "visdrone_categories_10.json",
        "--category-map-ivid-to-coco", "visdrone_coco_categories_10.json"
    ])

    # --- UAVDT: generate structure ---
    run([
        "python3", str(repo / "generate-dataset-structure.py"),
        "--root", "datasets/uavdt",
        "--dataset", "uavdt",
        "--category-map-vd-to-ivid", "uavdt_categories_3.json",
        "--category-map-ivid-to-coco", "uavdt_coco_categories_3.json",
        "--frame-digits", "7"
    ])
    
    # --- Finalize both datasets into RF-DETR + ILSVRC layouts ---
    # If you saved the script under a different name (e.g., finaliseDatasets.py),
    # update the filename below accordingly.
    run([
    "python3", str(repo / "buildImageNetVid.py"),
    "--dataset", "visdrone",
    "--root", "datasets/visdrone",
    "--copy",
    ])

    run([
    "python3", str(repo / "buildCocoVid.py"),
    "--dataset", "visdrone",
    "--root", "datasets/visdrone",
    "--out", "datasets/visdrone/COCO",
    "--copy",
    ])

    run([
    "python3", str(repo / "buildImageNetVid.py"),
    "--dataset", "uavdt",
    "--root", "datasets/uavdt",
    "--copy",
    ])

    run([
    "python3", str(repo / "buildCocoVid.py"),
    "--dataset", "uavdt",
    "--root", "datasets/uavdt",
    "--out", "datasets/uavdt/COCO",
    "--copy",
    ])

    print("All done.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[error] Command failed with exit code {e.returncode}:\n  {' '.join(e.cmd)}",
              file=sys.stderr)
        sys.exit(e.returncode)
