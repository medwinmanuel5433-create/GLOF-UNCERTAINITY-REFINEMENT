import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof import core as G
from glof.data import list_pairs, load_mask, shape_stats, condition_proxies
from glof.models import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/kaggle_test/images")
    ap.add_argument("--masks", default="data/kaggle_test/masks")
    ap.add_argument("--yolo", default="weights/yolo11m_seg_best.pt")
    ap.add_argument("--sam2", default="weights/sam2_hiera_base_plus.pt")
    ap.add_argument("--out", default="results/test")
    ap.add_argument("--configs", default=",".join(G.CONFIGS))
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = a.configs.split(",")
    configs = {k: G.CONFIGS[k] for k in keys}
    for k, c in list(configs.items()):
        if c.get("budget_from") and c["budget_from"] not in configs:
            configs[c["budget_from"]] = G.CONFIGS[c["budget_from"]]

    yolo, R, n_yolo, n_sam2 = load_models(a.yolo, a.sam2)
    json.dump(dict(params=G.P, yolo_params=n_yolo, sam2_params=n_sam2, device=G.DEVICE,
                   gpu=torch.cuda.get_device_name(0) if G.CUDA else "cpu"),
              open(out / "run_info.json", "w"), indent=1)
    if a.save_masks:
        (out / "masks").mkdir(exist_ok=True)

    pairs = list_pairs(a.images, a.masks)[: a.limit or None]
    state_f = out / "state.pkl"
    S = pickle.load(open(state_f, "rb")) if state_f.exists() else \
        dict(rows=[], iters=[], lakes=[], uq=[], gbapr=[], timing=[], done=set())

    for n, (ip, mp) in enumerate(pairs):
        if ip.name in S["done"]:
            continue
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        gt = load_mask(mp, (H, W))
        if G.CUDA:
            torch.cuda.reset_peak_memory_stats()
        t0 = G.tic()
        dets = G.detect(yolo, img)
        t_yolo = G.toc(t0)
        t0 = G.tic()
        R.set_image(img)
        t_enc = G.toc(t0)
        res, extra = G.run_image(R, dets, H, W, configs, ip.name, gt=gt, keep_hist=("H", "J"))
        meta = dict(image=ip.name, **shape_stats(gt), **condition_proxies(img, gt))
        for k, o in res.items():
            cm = G.confusion(o["mask"], gt)
            S["rows"].append(dict(config=k, **dict(zip(["TP", "FP", "FN", "TN"], cm)), **G.metrics_from(*cm),
                                  **{x: o["info"][x] for x in ["n_det", "clicks", "pos_clicks", "neg_clicks",
                                                                "dec_calls", "t_dec", "t_post"]},
                                  t_yolo=t_yolo, t_enc=t_enc if configs[k].get("sam", True) else 0.0, **meta))
            if o["hist"] is not None:
                for t, hm in enumerate(o["hist"]):
                    S["iters"].append(dict(config=k, image=ip.name, iter=t,
                                           **dict(zip(["TP", "FP", "FN", "TN"], G.confusion(hm, gt)))))
        S["lakes"] += [dict(image=ip.name, **x) for x in extra["lakes"]]
        S["gbapr"] += [dict(image=ip.name, **x) for x in extra["gbapr"]]
        S["uq"] += extra["uq"]
        S["timing"].append(dict(image=ip.name, t_yolo=t_yolo, t_enc=t_enc, n_det=len(dets),
                                peak_mem_MB=torch.cuda.max_memory_allocated() / 2 ** 20 if G.CUDA else np.nan))
        if a.save_masks:
            np.savez_compressed(out / "masks" / f"{ip.stem}.npz", gt=gt,
                                **{k: res[k]["mask"] for k in ("A", "B", "H") if k in res},
                                U=extra["U"].astype(np.float16),
                                boxes=np.array([list(d["box"]) + [d["conf"]] for d in dets], np.float32).reshape(-1, 5),
                                clicks=json.dumps(res["H"]["clicks"]) if "H" in res else "[]")
        S["done"].add(ip.name)
        if len(S["done"]) % 25 == 0 or n == len(pairs) - 1:
            pickle.dump(S, open(state_f, "wb"))
            print(f"{len(S['done'])}/{len(pairs)}", flush=True)

    pickle.dump(S, open(state_f, "wb"))
    pd.DataFrame(S["rows"]).to_csv(out / "per_image.csv", index=False)
    pd.DataFrame(S["iters"]).to_csv(out / "per_iteration.csv", index=False)
    pd.DataFrame(S["lakes"]).to_csv(out / "lakes.csv", index=False)
    pd.DataFrame(S["gbapr"]).to_csv(out / "gbapr_prompts.csv", index=False)
    pd.DataFrame(S["timing"]).to_csv(out / "timing.csv", index=False)
    np.savez_compressed(out / "uncertainty_samples.npz",
                        **{k: np.concatenate([u[k] for u in S["uq"]]) for k in
                           ["err", "attention_entropy", "boundary_distance", "predictive_entropy"]})


if __name__ == "__main__":
    main()
