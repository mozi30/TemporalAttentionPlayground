import shutil
import pathlib
import zipfile
import gdown
import os
import sys

ROOT = pathlib.Path("./datasets/uavdt").resolve()
ROOT.mkdir(parents=True, exist_ok=True)

files = {
    "UAV-benchmark-M.zip": "1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc",       # DET/MOT (images)
    "UAV-benchmark-MOTD_v1.0.zip": "19498uJd7T9w4quwnQEy62nibt3uyT9pq" # MOTD (GT)
}

# --- Download ---------------------------------------------------------------
for zip_name, fid in files.items():
    zip_path = ROOT / zip_name
    if not zip_path.exists():
        print(f"Downloading {zip_name} ...")
        ok = gdown.download(id=fid, output=str(zip_path), quiet=False)
        if not ok or not zip_path.exists():
            sys.exit(f"Error: failed to download {zip_name} from Google Drive id={fid}")

# --- Extract ----------------------------------------------------------------
for z in ROOT.glob("*.zip"):
    extract_marker = ROOT / f".extracted_{z.stem}"
    if extract_marker.exists():
        continue
    print("Extracting", z, "...")
    with zipfile.ZipFile(z) as zz:
        zz.extractall(ROOT)
    extract_marker.touch()

# --- Locate MOTD directory (for GT) -----------------------------------------
dst_gt = ROOT / "GT"
motd_dir = None
if not dst_gt.exists():
    for cand in sorted(ROOT.glob("UAV-benchmark-MOTD*")):
        if cand.is_dir():
            motd_dir = cand
            break

    if motd_dir is None:
        sys.exit(f"Error: UAVDT MOTD folder not found under {ROOT}")

    src_gt = motd_dir / "GT"
    if not src_gt.exists():
        sys.exit(f"Error: GT folder not found inside {motd_dir}")

    dst_gt.mkdir(exist_ok=True)

    # --- Move GT files into ROOT/GT -----------------------------------------
    for file in src_gt.iterdir():
        if file.is_file() and file.name.endswith("_gt_whole.txt"):
            shutil.move(str(file), str(dst_gt / file.name))

    print("[done] Extracted to", ROOT)

# ==================== NEW: 60%/20%/20% SPLIT ====================

VAL_DIR = ROOT / "UAVDT-val"
TRAIN_DIR = ROOT / "UAVDT-train"
TEST_DIR = ROOT / "UAVDT-test"

VAL_SEQ = VAL_DIR / "sequences"
VAL_ANN = VAL_DIR / "annotations"
TRAIN_SEQ = TRAIN_DIR / "sequences"
TRAIN_ANN = TRAIN_DIR / "annotations"
TEST_SEQ = TEST_DIR / "sequences"
TEST_ANN = TEST_DIR / "annotations"

for d in [VAL_SEQ, VAL_ANN, TRAIN_SEQ, TRAIN_ANN, TEST_SEQ, TEST_ANN]:
    d.mkdir(parents=True, exist_ok=True)

# possible image roots
possible_image_roots = {ROOT, *(p for p in ROOT.glob("UAV-benchmark-*") if p.is_dir())}

def find_seq_dir(seq_name: str) -> pathlib.Path | None:
    for base in possible_image_roots:
        cand = base / seq_name
        if cand.is_dir():
            return cand
    for base in possible_image_roots:
        for sub in base.iterdir():
            if sub.is_dir():
                cand = sub / seq_name
                if cand.is_dir():
                    return cand
    return None

def pad_seq_filenames_inplace(seq_dir: pathlib.Path, digits: int = 7) -> int:
    """
    Rename image files in seq_dir so their numeric stems are zero-padded to `digits`.
    Also removes 'img' prefix if present.
    E.g., img000001.jpg -> 0000001.jpg (for digits=7).
    Two-phase rename to avoid collisions.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    mapping = {}
    for p in files:
        stem = p.stem
        # Remove 'img' prefix if present
        if stem.startswith('img'):
            stem = stem[3:]
        try:
            num = int(stem)  # handles "000001" etc.
        except ValueError:
            # Non-numeric stem: leave it as-is
            continue
        new_name = f"{num:0{digits}d}{p.suffix.lower()}"
        if p.name != new_name:
            mapping[p] = seq_dir / new_name

    if not mapping:
        return 0

    # Phase 1: move to unique temp names
    temps = []
    for idx, (src, dst) in enumerate(mapping.items()):
        tmp = src.with_name(f"__tmp__{idx}__{src.name}")
        src.rename(tmp)
        temps.append((tmp, dst))

    # Phase 2: move temps to final padded names
    renamed = 0
    for tmp, dst in temps:
        if dst.exists():
            raise RuntimeError(f"Destination already exists: {dst}")
        tmp.rename(dst)
        renamed += 1

    return renamed

# collect sequences
gt_files = sorted([p for p in dst_gt.glob("*_gt_whole.txt") if p.is_file()])
sequences = []
for gt in gt_files:
    seq = gt.stem.split("_")[0]  # e.g. M0101
    seq_dir = find_seq_dir(seq)
    sequences.append((seq, gt, seq_dir))

pairs = [(seq, gt, seq_dir) for (seq, gt, seq_dir) in sequences if seq_dir is not None]
pairs.sort(key=lambda x: x[0])

n = len(pairs)
k_val = max(1, round(0.2 * n))
k_test = max(1, round(0.2 * n))
k_train = n - k_val - k_test

val_pairs = pairs[:k_val]
test_pairs = pairs[k_val:k_val + k_test]
train_pairs = pairs[k_val + k_test:]

def move_seq_and_gt(seq: str, gt_path: pathlib.Path, seq_dir: pathlib.Path,
                    dest_seq: pathlib.Path, dest_ann: pathlib.Path,
                    pad_digits: int = 7):
    # move images (sequence folder)
    dst_seq_dir = dest_seq / seq
    if not dst_seq_dir.exists():
        shutil.move(str(seq_dir), str(dst_seq_dir))
    # ensure images are zero-padded to `pad_digits`
    renamed = pad_seq_filenames_inplace(dst_seq_dir, digits=pad_digits)
    if renamed:
        print(f"Padded {renamed} files to {pad_digits} digits in {dst_seq_dir}")

    # move annotation
    dst_gt_file = dest_ann / gt_path.name
    if not dst_gt_file.exists():
        shutil.move(str(gt_path), str(dst_gt_file))

print(f"Splitting {n} sequences → {k_val} val / {k_test} test / {k_train} train")

for seq, gt, seq_dir in val_pairs:
    move_seq_and_gt(seq, gt, seq_dir, VAL_SEQ, VAL_ANN, pad_digits=7)

for seq, gt, seq_dir in test_pairs:
    move_seq_and_gt(seq, gt, seq_dir, TEST_SEQ, TEST_ANN, pad_digits=7)

for seq, gt, seq_dir in train_pairs:
    move_seq_and_gt(seq, gt, seq_dir, TRAIN_SEQ, TRAIN_ANN, pad_digits=7)

# cleanup: remove residuals (GT, zips, extracted folders)
try:
    if dst_gt.exists():
        shutil.rmtree(dst_gt)
    if motd_dir and motd_dir.exists():
        shutil.rmtree(motd_dir)

    for marker in ROOT.glob(".extracted_*"):
        marker.unlink()

    for z in ROOT.glob("*.zip"):
        z.unlink()

    for d in ROOT.glob("UAV-benchmark-M*"):
        if d.is_dir():
            shutil.rmtree(d)
        elif d.is_file():
            d.unlink()

except Exception:
    print("Warning: failed to remove folders.")

print(f"[done] Final structure:\n- {VAL_DIR}\n- {TEST_DIR}\n- {TRAIN_DIR}")
