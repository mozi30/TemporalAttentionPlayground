import os, pathlib, gdown, zipfile

ROOT = pathlib.Path("./datasets/visdrone")
ROOT.mkdir(parents=True, exist_ok=True)

files = {
    "VisDrone2019-VID-train.zip": "1NSNapZQHar22OYzQYuXCugA3QlMndzvw",
    "VisDrone2019-VID-val.zip": "1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B",
    "VisDrone2019-VID-test-dev.zip": "1-BEq--FcjshTF1UwUabby_LHhYj41os5"
}

for name, fid in files.items():
    out = ROOT / name
    if not out.exists():
        gdown.download(id=fid, output=str(out), quiet=False)

# unzip
for z in ROOT.glob("*.zip"):
    with zipfile.ZipFile(z) as zz:
        zz.extractall(ROOT)
    os.remove(z)  # optional: delete zip after extraction


print("All done.")