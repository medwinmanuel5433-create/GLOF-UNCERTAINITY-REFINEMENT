import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof.core import confusion, metrics_from
from glof.data import IMG_EXT, list_pairs, load_mask, mask_from_yolo_label

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def yolo_split(root):
    root = Path(root)
    imgs = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    return [(p, root / "labels" / f"{p.stem}.txt") for p in imgs]


def to_tensor(img, size):
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size)).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()


class RoboflowSeg(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, lp = self.items[i]
        img = cv2.imread(str(ip))
        m = mask_from_yolo_label(lp, img.shape[:2]) if lp.exists() else np.zeros(img.shape[:2], bool)
        if self.train:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.15:
                img, m = img[::-1], m[::-1]
        y = cv2.resize(m.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return to_tensor(np.ascontiguousarray(img), self.size), torch.from_numpy(y[None].astype(np.float32))


def build(arch):
    if arch == "unet":
        return smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
    return smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", classes=1)


@torch.no_grad()
def predict(model, img, size):
    H, W = img.shape[:2]
    p = torch.sigmoid(model(to_tensor(img, size)[None].to(DEV)))[0, 0].cpu().numpy()
    return cv2.resize(p, (W, H)) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/roboflow/train")
    ap.add_argument("--valid", default="data/roboflow/valid")
    ap.add_argument("--images", default="data/kaggle_test/images")
    ap.add_argument("--masks", default="data/kaggle_test/masks")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="results/baselines")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    tr = DataLoader(RoboflowSeg(yolo_split(a.train), a.size, True), batch_size=8, shuffle=True, num_workers=2)
    va = yolo_split(a.valid)
    loss_dice = smp.losses.DiceLoss("binary")
    loss_bce = torch.nn.BCEWithLogitsLoss()
    rows = []
    for arch, label in [("unet", "U-Net"), ("deeplabv3plus", "DeepLabV3+")]:
        model = build(arch).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        best, ckpt = -1.0, out / f"{arch}.pt"
        for _ in range(a.epochs):
            model.train()
            for x, y in tr:
                x, y = x.to(DEV), y.to(DEV)
                logit = model(x)
                loss = loss_bce(logit, y) + loss_dice(logit, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            cm = np.zeros(4, np.int64)
            for ip, lp in va:
                img = cv2.imread(str(ip))
                gt = mask_from_yolo_label(lp, img.shape[:2]) if lp.exists() else np.zeros(img.shape[:2], bool)
                cm += confusion(predict(model, img, a.size), gt)
            iou = metrics_from(*cm)["IoU"]
            if iou > best:
                best = iou
                torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=DEV))
        model.eval()
        cm = np.zeros(4, np.int64)
        for ip, mp in list_pairs(a.images, a.masks):
            img = cv2.imread(str(ip))
            cm += confusion(predict(model, img, a.size), load_mask(mp, img.shape[:2]))
        m = metrics_from(*cm)
        rows.append({"Method": label, "IoU": m["IoU"], "Dice": m["Dice"], "Precision": m["Precision"], "Recall": m["Recall"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table3a_baselines.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
