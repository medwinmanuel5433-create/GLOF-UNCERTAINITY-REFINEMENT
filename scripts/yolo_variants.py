import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof.core import metrics_from

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow/data.yaml")
    ap.add_argument("--images", default="data/kaggle_test/images")
    ap.add_argument("--masks", default="data/kaggle_test/masks")
    ap.add_argument("--sam2", default="weights/sam2_hiera_base_plus.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--variants", default="n,s,m,l,x")
    ap.add_argument("--out", default="results/yolo_variants")
    a = ap.parse_args()
    rows = []
    for v in a.variants.split(","):
        w = Path("weights") / f"yolo11{v}_seg_{a.epochs}ep.pt"
        if not w.exists():
            subprocess.run([sys.executable, ROOT / "scripts/train_yolo.py", "--data", a.data,
                            "--model", f"yolo11{v}-seg.pt", "--epochs", str(a.epochs), "--out", str(w)], check=True)
        run = Path(a.out) / v
        subprocess.run([sys.executable, ROOT / "scripts/evaluate.py", "--images", a.images, "--masks", a.masks,
                        "--yolo", str(w), "--sam2", a.sam2, "--out", str(run), "--configs", "A,B,H"], check=True)
        df = pd.read_csv(run / "per_image.csv")
        iou = {k: metrics_from(*g[["TP", "FP", "FN", "TN"]].sum().values)["IoU"] for k, g in df.groupby("config")}
        rows.append({"Detector": f"YOLOv11{v}-seg", "YOLO only": iou["A"], "+ SAM2": iou["B"], "Proposed": iou["H"],
                     "Training": f"{a.epochs} epochs"})
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.out) / "table2b_yolo_variants.csv", index=False, float_format="%.3f")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
