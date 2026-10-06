import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

TRAIN_ARGS = dict(imgsz=640, batch=4, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, momentum=0.937,
                  weight_decay=0.0005, warmup_epochs=3.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, degrees=5.0,
                  translate=0.08, scale=0.4, fliplr=0.5, flipud=0.15, mosaic=0.8, mixup=0.0, copy_paste=0.0,
                  close_mosaic=15, patience=25, seed=42, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/roboflow/data.yaml")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="weights/yolo11m_seg_best.pt")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    name = Path(a.model).stem
    YOLO(a.model).train(data=a.data, epochs=a.epochs, project="runs", name=name, exist_ok=True,
                        verbose=False, plots=False, device=a.device, **TRAIN_ARGS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path("runs") / name / "weights" / "best.pt", a.out)


if __name__ == "__main__":
    main()
