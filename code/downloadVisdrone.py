import os
import pathlib
import zipfile
import subprocess
import argparse
import shutil
from typing import List, Optional

def parse_args():
    p = argparse.ArgumentParser(description="Download VisDrone2019-VID via mega-get and unzip")
    p.add_argument("--root", type=str, default="datasets/visdrone", help="Ziel-Root-Verzeichnis für VisDrone-Daten")
    return p.parse_args()

args = parse_args()
ROOT = pathlib.Path(args.root)
ROOT.mkdir(parents=True, exist_ok=True)

files = {
    # Mega.nz URLs (public)
    "VisDrone2019-VID-train.zip": "https://mega.nz/file/4jwBBAJa#yhtv7GCulkXSqvz269Sw3cecXJUpN_2FBqNBgQ1Cn4M",
    "VisDrone2019-VID-val.zip": "https://mega.nz/file/A7pklJKJ#BhSjtVF-8DeUWlmjtNb5CEZFMkRBSOc6hMHP7pTVarA",
    "VisDrone2019-VID-test-dev.zip": "https://mega.nz/file/FrYzmIxY#OQ6qQLHYgqgHfxUcfSXDtTcjTup1QwN2_Vun6c84Kj4",
}

# Erwartete Zielordner nach dem Entpacken (gleicher Name wie ZIP ohne .zip)
expected_dirs = {name: pathlib.Path(name[:-4]) for name in files.keys()}

def run(cmd: List[str], cwd: Optional[pathlib.Path] = None):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


for name, url in files.items():
    out = ROOT / name
    exp_dir = expected_dirs[name]

    # 1) Falls bereits entpackter Ordner existiert (unter ROOT oder ggf. parent), überspringen
    dir_here = ROOT / exp_dir
    dir_parent = ROOT.parent / exp_dir
    if dir_here.exists() and dir_here.is_dir():
        print(f"[skip] already extracted: {dir_here}")
        continue
    if dir_parent.exists() and dir_parent.is_dir():
        print(f"[skip] already extracted in parent: {dir_parent}")
        # Optional: nicht verschieben, um teure Moves zu vermeiden
        continue

    # 2) Falls ZIP bereits im Ziel existiert, später nur entpacken
    if out.exists():
        print(f"[skip-download] zip already exists: {out}")
    else:
        # 2a) Falls ZIP bereits im übergeordneten datasets-Ordner liegt (manuell geladen), verschieben
        alt = ROOT.parent / name
        if alt.exists():
            print(f"[move] found existing {alt}, moving to {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(alt), str(out))
        else:
            # 2b) Download via mega-get direkt ins Zielverzeichnis
            run(["mega-get", url], cwd=ROOT)
            if not out.exists():
                # mega-get legt die Datei gewöhnlich im CWD ab; fallback: prüfen, ob eine gleichnamige Datei im CWD liegt
                candidate = ROOT / pathlib.Path(url).name
                if candidate.exists() and candidate != out:
                    shutil.move(str(candidate), str(out))
            if not out.exists():
                raise RuntimeError(f"Download failed or unexpected output name for {name}")

    # 3) Entpacken (nur wenn Zielordner noch nicht existiert)
    if not (ROOT / exp_dir).exists():
        print(f"[unzip] {out}")
        with zipfile.ZipFile(out) as zz:
            zz.extractall(ROOT)
    else:
        print(f"[skip-unzip] target exists: {ROOT / exp_dir}")
    # 4) Optional: ZIP löschen
    try:
        os.remove(out)
    except OSError:
        pass
print("All done.")