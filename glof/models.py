from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .core import DEVICE, SAM2Runner

SAM2_CFG = "configs/sam2/sam2_hiera_b+.yaml"


def load_models(yolo_weights, sam2_ckpt, sam2_cfg=SAM2_CFG):
    yolo = YOLO(str(yolo_weights))
    sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=DEVICE)
    n_yolo = sum(p.numel() for p in yolo.model.parameters())
    n_sam2 = sum(p.numel() for p in sam2.parameters())
    return yolo, SAM2Runner(SAM2ImagePredictor(sam2)), n_yolo, n_sam2
