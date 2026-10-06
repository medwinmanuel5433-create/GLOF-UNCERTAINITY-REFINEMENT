import argparse
import json
import shutil
from pathlib import Path

import yaml


def convert(export, prefix, out):
    for split in ["train", "valid"]:
        ann = Path(export) / split / "_annotations.coco.json"
        if not ann.exists():
            continue
        coco = json.load(open(ann))
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        polys = {}
        for a in coco["annotations"]:
            for seg in a.get("segmentation") or []:
                if isinstance(seg, list) and len(seg) >= 6:
                    polys.setdefault(a["image_id"], []).append(seg)
        for im in coco["images"]:
            src = Path(export) / split / im["file_name"]
            name = f"{prefix}_{Path(im['file_name']).stem}"
            shutil.copy(src, out / split / "images" / f"{name}{src.suffix}")
            W, H = im["width"], im["height"]
            lines = ["0 " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in zip(s[0::2], s[1::2]))
                     for s in polys.get(im["id"], [])]
            (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True, help="Roboflow COCO-segmentation export folders")
    ap.add_argument("--prefixes", nargs="+", default=["v6", "v8"])
    ap.add_argument("--out", default="data/roboflow")
    a = ap.parse_args()
    out = Path(a.out)
    for e, p in zip(a.exports, a.prefixes):
        convert(e, p, out)
    yaml.safe_dump(dict(path=str(out.resolve()), train="train/images", val="valid/images", nc=1, names=["lake"]),
                   open(out / "data.yaml", "w"))


if __name__ == "__main__":
    main()
