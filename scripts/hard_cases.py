import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof import core as G
from glof.data import IMG_EXT, load_mask, mask_from_yolo_label
from glof.models import load_models
from glof.viz import hard_case_figure

ROOT = Path(__file__).resolve().parents[1]


def find(root, stem):
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.stem.endswith(stem) and p.parent.name == "images":
            return p
    raise FileNotFoundError(stem)


def ground_truth(ip, shape):
    split = ip.parent.parent
    for m in [split / "masks" / f"{ip.stem}.png", split / "labels" / f"{ip.stem}.txt"]:
        if m.exists():
            return load_mask(m, shape) if m.suffix == ".png" else mask_from_yolo_label(m, shape)
    raise FileNotFoundError(f"no mask or label for {ip}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow")
    ap.add_argument("--cases", default=str(ROOT / "configs/hard_cases.csv"))
    ap.add_argument("--yolo", default="weights/yolo11m_seg_best.pt")
    ap.add_argument("--sam2", default="weights/sam2_hiera_base_plus.pt")
    ap.add_argument("--out", default="results/hard_cases")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    yolo, R, _, _ = load_models(a.yolo, a.sam2)
    cfg = {k: G.CONFIGS[k] for k in ("B", "H")}
    rows, figs = [], {}
    for _, c in pd.read_csv(a.cases).iterrows():
        ip = find(a.data, c.image)
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        gt = ground_truth(ip, (H, W))
        dets = G.detect(yolo, img)
        R.set_image(img)
        res, _ = G.run_image(R, dets, H, W, cfg, ip.name, diag=False, keep_hist=())
        row = {"Case": f"{c.case} (Fig. {c.figure})"}
        for key, name in [("B", "YOLO+SAM2"), ("H", "Proposed")]:
            m = G.metrics_from(*G.confusion(res[key]["mask"], gt))
            row.update({f"{name} {q}": m[q] for q in ["IoU", "Dice", "Precision", "Recall"]})
        row["Clicks"] = len(res["H"]["clicks"])
        rows.append(row)
        figs.setdefault(c.case, []).append(dict(img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gt=gt, B=res["B"]["mask"],
                                                H=res["H"]["mask"], clicks=res["H"]["clicks"], m=row))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Case": "Mean (9 images)", **df.drop(columns="Case").mean().to_dict()}
    df.to_csv(out / "table9_hard_cases.csv", index=False, float_format="%.3f")
    for i, (case, items) in enumerate(figs.items()):
        hard_case_figure(items, f"Hard case: {case.lower()} (Roboflow dataset)", out / f"fig{10 + i}_{case.split()[0].lower()}.png")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
