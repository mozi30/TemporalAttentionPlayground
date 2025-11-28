"""Helper to import convert_ovis_coco from the YOLOV/yolox package.

This file runs outside the package tree (it's in the `code/` folder), so we
add the project root to sys.path at runtime and import the module by its
package path.
"""
from pathlib import Path
import sys
import importlib.util
# Add project root so the `YOLOV` package can be imported. Adjust if your layout differs.

ROOT = Path(__file__).resolve().parents[2]  # /root/TemporalAttentionPlayground
sys.path.insert(0, str(ROOT))
ovis_path = ROOT / "YOLOV" / "yolox" / "data" / "datasets" / "ovis.py"
spec = importlib.util.spec_from_file_location("ovis_module", str(ovis_path))
ovis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ovis)
convert_ovis_coco = ovis.convert_ovis_coco

def build_visdrone_ovis(visdrone_root: Path):
    """Convert VisDrone dataset to OVIS COCO format using YOLOX's convert_ovis_coco.

    Args:
        visdrone_root (Path): Root directory of the VisDrone dataset containing
            'VisDrone2019-VID-train' and 'VisDrone2019-VID-val' folders.
        out_root (Path): Output root directory where the converted dataset will be stored.
        link_mode (str): Mode for handling files. Options are 'symlink', 'hardlink', or 'copy'.
    """
    train_dir = visdrone_root / "imagenet_vid_train.json"
    train_out_dir = visdrone_root / "yolox_train.json"
    val_dir = visdrone_root / "imagenet_vid_val.json"
    val_out_dir = visdrone_root / "yolox_val.json"

    # Check if output files already exist
    if train_out_dir.exists():
        print(f"Training output already exists: {train_out_dir}, skipping conversion.")
    else:
        convert_ovis_coco(
            data_dir=train_dir,
            save_dir=train_out_dir,
        )
        print(f"Converted training data to {train_out_dir}")

    if val_out_dir.exists():
        print(f"Validation output already exists: {val_out_dir}, skipping conversion.")
    else:
        convert_ovis_coco(
            data_dir=val_dir,
            save_dir=val_out_dir,
        )
        print(f"Converted validation data to {val_out_dir}")
    
    print(f"VisDrone OVIS to COCO conversion complete at {visdrone_root}")

if __name__ == "__main__":

    build_visdrone_ovis(
        visdrone_root=Path.home() / "datasets/visdrone/yolov/annotations",
    )
