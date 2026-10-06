import math
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_pairs(images_dir, masks_dir):
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = [(p, masks[p.stem]) for p in sorted(images_dir.iterdir(), key=_natural)
             if p.suffix.lower() in IMG_EXT and p.stem in masks]
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs in {images_dir} and {masks_dir}")
    return pairs


def _natural(p):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def load_mask(path, shape=None):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if shape is not None and g.shape != shape:
        g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return g > (127 if g.max() > 1 else 0)


def mask_from_yolo_label(label_path, shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for line in Path(label_path).read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xy = np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
    return m.astype(bool)


def shape_stats(gt):
    g = gt.astype(np.uint8)
    area = int(g.sum())
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return dict(gt_area_px=area, gt_area_frac=area / g.size, gt_components=len(cnts),
                boundary_complexity=(per ** 2) / (4 * math.pi * area) if area else np.nan)


def condition_proxies(img, gt):
    g = gt > 0
    if not g.any():
        return dict(snow_ice_cloud_frac=np.nan, boundary_contrast=np.nan)
    f = img.astype(np.float32) / 255.0
    bright, sat = f.mean(2), f.max(2) - f.min(2)
    ring = cv2.dilate(g.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~g
    ring = ring if ring.any() else ~g
    inner = g & ~cv2.erode(g.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    outer = cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~g
    band = inner | outer
    contrast = abs(bright[inner].mean() - bright[outer].mean()) / (bright[band].std() + 1e-6) \
        if inner.any() and outer.any() else np.nan
    return dict(snow_ice_cloud_frac=float(((bright > 0.70) & (sat < 0.15))[ring].mean()),
                boundary_contrast=float(contrast))
