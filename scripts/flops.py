import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof import core as G
from glof.models import load_models


def gflops(fn):
    with FlopCounterMode(display=False) as fc:
        with torch.inference_mode():
            fn()
    return fc.get_total_flops() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="weights/yolo11m_seg_best.pt")
    ap.add_argument("--sam2", default="weights/sam2_hiera_base_plus.pt")
    ap.add_argument("--run", default="results/test")
    a = ap.parse_args()
    yolo, R, _, _ = load_models(a.yolo, a.sam2)
    img = np.zeros((400, 400, 3), np.uint8)
    f_yolo = gflops(lambda: yolo.model.float().to(G.DEVICE)(torch.zeros(1, 3, G.P["YOLO_IMGSZ"], G.P["YOLO_IMGSZ"], device=G.DEVICE)))
    f_enc = gflops(lambda: R.set_image(img))
    f_dec = gflops(lambda: R.call(np.array([10, 10, 100, 100], np.float32), None, None, None, dict(t_dec=0.0, dec_calls=0)))
    rows = pd.read_csv(Path(a.run) / "per_image.csv")
    calls = rows.groupby("config").dec_calls.mean()
    df = pd.DataFrame([{"YOLO GFLOPs": f_yolo, "SAM2 encoder GFLOPs": f_enc, "SAM2 decoder GFLOPs per call": f_dec,
                        "YOLO+SAM2 GFLOPs per image": f_yolo + f_enc + f_dec * calls["B"],
                        "proposed GFLOPs per image": f_yolo + f_enc + f_dec * calls["H"]}]).T
    df.rename(columns={0: "value"}).to_csv(Path(a.run) / "tables" / "table7_flops.csv", index_label="metric", float_format="%.2f")
    print(df.to_string(header=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
