# Glacial Lake Segmentation using YOLOv11m-seg and SAM2 with Uncertainty-Guided Refinement

Glacial lakes are segmented in three steps. YOLOv11m-seg finds each lake and gives a box and a rough mask. SAM2 (Hiera-B+) segments the lake from the box. Where the YOLO mask and the SAM2 mask disagree, the framework places a corrective click: a positive click where YOLO sees lake and SAM2 does not, and a negative click where SAM2 sees lake and YOLO does not. The click goes to the point where SAM2's attention entropy is highest (where SAM2 is least certain). SAM2 is run again with the click, up to five times, and stops when the mask no longer changes (IoU between masks ≥ 0.95) or no uncertain disagreement is left. A 3×3 opening and closing cleans the final mask. The correction needs no training and no user input.

## Results

All results are on the Kaggle test set (410 Sentinel-2 images). Metrics are pixel-level, pooled over all test images.
Full tables and figures: [docs/GLOF_RESULTS.docx](docs/GLOF_RESULTS.docx)

### Component-wise ablation

| # | Configuration | IoU | Dice/F1 | Precision | Recall | Specificity | Pixel Acc. | Balanced Acc. |
|---|---|---|---|---|---|---|---|---|
| S.No. | Model / Configuration | IoU | Dice/F1 | Prec. | Recall | Spec. | Pix. Acc. | Bal. Acc. | Params |
| 1 | YOLO11n-seg | 0.7511 | 0.8579 | 0.8950 | 0.8237 | 0.9877 | 0.9737 | 0.9067 | 2.9M |
| 2 | YOLO11s-seg | 0.7117 | 0.8315 | 0.8172 | 0.7605 | 0.8927 | 0.9703 | 0.8766 | 10.1M |
| 3 | YOLO11m-seg | 0.7021 | 0.8250 | 0.8368 | 0.7370 | 0.9947 | 0.9698 | 0.8659 | 22.4M |
| 4 | YOLO11l-seg | 0.7221 | 0.8386 | 0.8776 | 0.7794 | 0.9415 | 0.9711 | 0.8855 | 27.6M |
| 5 | YOLO11x-seg | 0.6978 | 0.8220 | 0.8627 | 0.7477 | 0.8424 | 0.9687 | 0.8700 | 62.1M |
| 6 | YOLO11m-seg + SAM2 | 0.7600 | 0.8150 | 0.7520 | 0.8154 | 0.9868 | 0.9804 | 0.8991 | 103.2M |
| 3 | + Boundary-aware box | 0.6459 | 0.7849 | 0.7764 | 0.7936 | 0.9651 | 0.9423 | 0.8793 |
| 4 | + Boundary-aware box and points | 0.6804 | 0.8098 | 0.8024 | 0.8174 | 0.9692 | 0.9491 | 0.8933 |
| 5 | Box prompt + random clicks (same number) | 0.7092 | 0.8299 | 0.8499 | 0.8108 | 0.9781 | 0.9559 | 0.8944 |
| 6 | Box prompt + distance-based clicks (same number) | 0.7049 | 0.8269 | 0.8436 | 0.8109 | 0.9770 | 0.9550 | 0.8940 |
| 7 | Box prompt + predictive-entropy clicks | 0.7232 | 0.8394 | 0.8648 | 0.8154 | 0.9805 | 0.9586 | 0.8980 |
| **8** | **Box prompt + attention-entropy clicks (proposed)** | **0.7876** | **0.8222** | **0.7636** | **0.8126** | **0.9867** | **0.9802** | **0.8997** | **103.2M** |
| **9** | **Full framework (boundary-aware box, points and attention-entropy clicks)** | **0.7876** | **0.8222** | **0.7636** | **0.8126** | **0.9867** | **0.9802** | **0.8997** | **103.2M** |

**Uncertainty map vs. SAM2 error pixels** (AUROC): attention entropy 0.728, distance to boundary 0.693, predictive entropy 0.559.

**Same data, same test set**: U-Net 0.5145 IoU, DeepLabV3+ 0.6403 IoU, YOLOv11m-seg + SAM2 0.7157 IoU, proposed 0.7209 IoU.

## Dataset

| Dataset | Images | Ground Truth |
|---|---:|---:|
| Training | 333 | 333 |
| Validation | 464 | 464 |
| Test | 1640 | 1640 |
| **Total** | **2437** | **2437** |

| GLOF dataset (all splits, one zip) | Everything | [GLOF_dataset_final.zip](https://github.com/medwinmanuel5433-create/GLOF-UNCERTAINITY-REFINEMENT/releases/download/v1.0/GLOF_dataset_final.zip) | CC BY 4.0 |

One class (`lake`). The test set comes from a different provider and sensor and is never used for training or tuning. Both sources are CC BY 4.0.

Put the data in this layout:

```
data/GLOF_dataset_final/
├── images/{train,valid,test}/   # RGB images (.jpg / .png)
├── labels/{train,valid,test}/   # YOLO segmentation polygons, class 0 = lake
├── masks/{train,valid,test}/    # binary masks (255 = lake), same file names as the images
├── data.yaml                    # YOLO training config
└── manifest.csv                 # source and split of every image
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
