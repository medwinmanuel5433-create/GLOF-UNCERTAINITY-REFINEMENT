# Glacial Lake Segmentation using YOLOv11m-seg and SAM2 with Uncertainty-Guided Refinement

Glacial lakes are segmented in three steps. YOLOv11m-seg finds each lake and gives a box and a rough mask. SAM2 (Hiera-B+) segments the lake from the box. Where the YOLO mask and the SAM2 mask disagree, the framework places a corrective click: a positive click where YOLO sees lake and SAM2 does not, and a negative click where SAM2 sees lake and YOLO does not. The click goes to the point where SAM2's attention entropy is highest (where SAM2 is least certain). SAM2 is run again with the click, up to five times, and stops when the mask no longer changes (IoU between masks ≥ 0.95) or no uncertain disagreement is left. A 3×3 opening and closing cleans the final mask. The correction needs no training and no user input.

## Results

All results are on the Kaggle test set (410 Sentinel-2 images). Metrics are pixel-level, pooled over all test images.

### Component-wise ablation

| # | Configuration | IoU | Dice/F1 | Precision | Recall | Specificity | Pixel Acc. | Balanced Acc. |
|---|---|---|---|---|---|---|---|---|
| 1 | YOLOv11m-seg only | 0.7007 | 0.8240 | 0.8549 | 0.7953 | 0.9794 | 0.9550 | 0.8873 |
| 2 | YOLOv11m-seg + SAM2 (box prompt) | 0.7157 | 0.8343 | 0.8583 | 0.8116 | 0.9795 | 0.9572 | 0.8956 |
| 3 | + Boundary-aware box | 0.6459 | 0.7849 | 0.7764 | 0.7936 | 0.9651 | 0.9423 | 0.8793 |
| 4 | + Boundary-aware box and points | 0.6804 | 0.8098 | 0.8024 | 0.8174 | 0.9692 | 0.9491 | 0.8933 |
| 5 | Box prompt + random clicks (same number) | 0.7092 | 0.8299 | 0.8499 | 0.8108 | 0.9781 | 0.9559 | 0.8944 |
| 6 | Box prompt + distance-based clicks (same number) | 0.7049 | 0.8269 | 0.8436 | 0.8109 | 0.9770 | 0.9550 | 0.8940 |
| 7 | Box prompt + predictive-entropy clicks | 0.7232 | 0.8394 | 0.8648 | 0.8154 | 0.9805 | 0.9586 | 0.8980 |
| **8** | **Box prompt + attention-entropy clicks (proposed)** | **0.7209** | **0.8378** | **0.8660** | **0.8114** | **0.9808** | **0.9583** | **0.8961** |
| 9 | Full (boundary-aware box, points and attention-entropy clicks) | 0.7200 | 0.8372 | 0.8569 | 0.8184 | 0.9791 | 0.9578 | 0.8988 |

### Ablation study

**YOLOv11-seg variants** (IoU: YOLO only → + SAM2 → proposed)

| Detector | Training | YOLO only | + SAM2 | Proposed |
|---|---|---|---|---|
| YOLOv11n-seg | 10 epochs | 0.690 | 0.686 | 0.707 |
| YOLOv11s-seg | 10 epochs | 0.601 | 0.605 | 0.609 |
| YOLOv11m-seg | 10 epochs | 0.643 | 0.653 | 0.666 |
| YOLOv11l-seg | 10 epochs | 0.648 | 0.667 | 0.674 |
| YOLOv11x-seg | 10 epochs | 0.569 | 0.590 | 0.593 |
| YOLOv11m-seg (final) | 120 epochs | 0.701 | 0.716 | 0.721 |

**Performance after each iteration** (proposed framework)

| Iteration | IoU | Dice/F1 | Precision | Recall |
|---|---|---|---|---|
| Initial SAM2 | 0.7157 | 0.8343 | 0.8583 | 0.8116 |
| 1 | 0.7142 | 0.8333 | 0.8572 | 0.8107 |
| 2 | 0.7180 | 0.8359 | 0.8629 | 0.8104 |
| 3 | 0.7184 | 0.8361 | 0.8623 | 0.8115 |
| 4 | 0.7214 | 0.8381 | 0.8667 | 0.8114 |
| 5 | 0.7209 | 0.8378 | 0.8660 | 0.8114 |

**Uncertainty map vs. SAM2 error pixels** (AUROC): attention entropy 0.728, distance to boundary 0.693, predictive entropy 0.559.

**Same data, same test set**: U-Net 0.5145 IoU, DeepLabV3+ 0.6403 IoU, YOLOv11m-seg + SAM2 0.7157 IoU, proposed 0.7209 IoU.

## Dataset

| Split | Source | Images |
|---|---|---|
| Training | Roboflow glacial-lake datasets: [GLOF_MARK1](https://universe.roboflow.com/medwins-workspace/glof-wyr2m) (333) + [glacier-lake](https://universe.roboflow.com/glacierlake/glacier-lake) (423) | 756 |
| Validation | Same two Roboflow datasets (95 + 41) | 136 |
| Test | [Kaggle Glacial Lake Dataset](https://www.kaggle.com/datasets/aatishshresthaa/glacial-lake-dataset) (Sentinel-2, 10 m, Himalaya) | 410 |

One class (`lake`). The test set comes from a different provider and sensor and is never used for training or tuning. Both sources are CC BY 4.0.

Put the data in this layout:

```
data/
├── roboflow/                 # YOLO-seg format (made by scripts/prepare_roboflow.py)
│   ├── data.yaml
│   ├── train/{images,labels}/
│   └── valid/{images,labels}/
└── kaggle_test/
    ├── images/               # image_1.png ... image_410.png
    └── masks/                # binary masks with the same file names
```

To build `data/roboflow` from the two Roboflow COCO-segmentation exports:

```bash
python scripts/prepare_roboflow.py --exports path/to/GLOF_MARK1_export path/to/glacier-lake_export --prefixes v6 v8
```

## Repository layout

```
glof-uncertainty-prompting/
├── glof/
│   ├── core.py            # detection, SAM2 runner, attention entropy, corrective clicks, all configurations
│   ├── data.py            # data loading, ground-truth masks, image statistics for grouping
│   ├── models.py          # loads YOLOv11m-seg and SAM2
│   └── viz.py             # figure drawing
├── scripts/
│   ├── setup.sh           # installs packages, SAM2 and the SAM2 Hiera-B+ checkpoint
│   ├── prepare_roboflow.py
│   ├── train_yolo.py      # YOLOv11m-seg training (120 epochs, 640 px, AdamW)
│   ├── evaluate.py        # runs every configuration on the test set
│   ├── tables.py          # ablation, iterations, AUROC, groups, significance, convergence, cost
│   ├── flops.py           # GFLOPs per component
│   ├── figures.py         # Figs. 6–9, 13, 14
│   ├── hard_cases.py      # Roboflow shadow / snow / debris cases (Table 9, Figs. 10–12)
│   ├── yolo_variants.py   # YOLOv11 n/s/m/l/x
│   └── baselines.py       # U-Net and DeepLabV3+
├── configs/hard_cases.csv
├── notebooks/GLOF_colab.ipynb
├── requirements.txt
└── LICENSE
```

## How to run

**Option 1: Google Colab.** Open `notebooks/GLOF_colab.ipynb`, set the runtime to T4 GPU, put `data/` in Google Drive and run all cells.

**Option 2: command line.**

```bash
bash scripts/setup.sh
python scripts/train_yolo.py                 # or copy your trained model to weights/yolo11m_seg_best.pt
python scripts/evaluate.py --save-masks      # results/test/per_image.csv and more
python scripts/tables.py                     # results/test/tables/*.csv
python scripts/flops.py
python scripts/figures.py                    # results/figures/
python scripts/hard_cases.py                 # results/hard_cases/
python scripts/yolo_variants.py              # results/yolo_variants/
python scripts/baselines.py                  # results/baselines/
```

## License

Code: MIT. Data: CC BY 4.0 (Roboflow datasets and Kaggle Glacial Lake Dataset).
