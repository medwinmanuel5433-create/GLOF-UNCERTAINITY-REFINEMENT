import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof.core import CONFIGS, ABLATION_ORDER, metrics_from, auroc

COLS = ["IoU", "Dice", "Precision", "Recall", "Specificity", "Pixel_Accuracy", "Balanced_Accuracy"]


def pooled(df):
    return metrics_from(*df[["TP", "FP", "FN", "TN"]].sum().values)


def thirds(s):
    n = len(s)
    c = math.ceil(n / 3)
    r = s.rank(method="first")
    return pd.Series(np.where(r <= c, 0, np.where(r <= n - c, 1, 2)), index=s.index)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/test")
    a = ap.parse_args()
    run = Path(a.run)
    T = run / "tables"
    T.mkdir(exist_ok=True)
    rows = pd.read_csv(run / "per_image.csv")
    iters = pd.read_csv(run / "per_iteration.csv")
    lakes = pd.read_csv(run / "lakes.csv")
    info = json.load(open(run / "run_info.json"))
    p_yolo, p_all = info["yolo_params"] / 1e6, (info["yolo_params"] + info["sam2_params"]) / 1e6
    by = {k: g.set_index("image") for k, g in rows.groupby("config")}

    t2 = pd.DataFrame([dict(Key=k, Configuration=CONFIGS[k]["name"], **pooled(by[k]),
                            Params=f"{p_yolo if k == 'A' else p_all:.1f}M")
                       for k in ABLATION_ORDER if k in by])
    t2.to_csv(T / "table2a_ablation.csv", index=False, float_format="%.4f")

    t4 = []
    for t in sorted(iters[iters.config == "H"].iter.unique()):
        m = pooled(iters[(iters.config == "H") & (iters.iter == t)])
        t4.append(dict(Iteration="Initial SAM2" if t == 0 else str(t), **{c: m[c] for c in COLS[:4]}))
    pd.DataFrame(t4).to_csv(T / "table4_iterations.csv", index=False, float_format="%.4f")

    uq = np.load(run / "uncertainty_samples.npz")
    err = uq["err"].astype(bool)
    pd.DataFrame([dict(Map=lab, AUROC=auroc(uq[k], err)) for lab, k in
                  [("Attention entropy (proposed)", "attention_entropy"),
                   ("Distance to boundary", "boundary_distance"),
                   ("Predictive entropy", "predictive_entropy")]]
                 ).to_csv(T / "table5_uncertainty_auroc.csv", index=False, float_format="%.3f")

    meta, B, Hp = by["A"], by["B"], by["H"]
    size, shape, contrast = thirds(meta.gt_area_frac), thirds(meta.boundary_complexity), thirds(meta.boundary_contrast)
    snow = meta.snow_ice_cloud_frac > 0.15
    groups = [("Small lakes", size == 0), ("Medium lakes", size == 1), ("Large lakes", size == 2),
              ("Regular boundary", shape == 0), ("Moderately irregular boundary", shape == 1),
              ("Highly irregular boundary", shape == 2),
              ("Single lake", meta.gt_components == 1), ("Multiple lakes", meta.gt_components >= 2),
              ("Snow, ice or cloud surroundings", snow), ("Low-contrast boundary", contrast == 0),
              ("Clear (none of the above two)", ~(snow | (contrast == 0)))]
    t6 = []
    for lab, sel in groups:
        ids = meta.index[sel.values]
        mb, mp = pooled(B.loc[ids]), pooled(Hp.loc[ids])
        t6.append({"Condition": lab, "Images": len(ids), "IoU (YOLO+SAM2)": mb["IoU"], "Dice (YOLO+SAM2)": mb["Dice"],
                   "IoU (Proposed)": mp["IoU"], "Dice (Proposed)": mp["Dice"]})
    pd.DataFrame(t6).to_csv(T / "table6_groups.csv", index=False, float_format="%.4f")

    pairs = [("A", "B"), ("B", "C"), ("B", "D"), ("B", "H"), ("F", "H"), ("E", "H"), ("H", "I"), ("D", "J")]
    sig = []
    for x, y in pairs:
        X, Y = by[x], by[y].loc[by[x].index]
        d = (Y.IoU - X.IoU).values
        nz = d[np.abs(d) > 1e-9]
        sig.append(dict(From=x, To=y, Delta_IoU_pp=100 * (pooled(Y)["IoU"] - pooled(X)["IoU"]),
                        Wilcoxon_p=wilcoxon(nz).pvalue if len(nz) >= 10 else np.nan,
                        Better=int((d > 1e-6).sum()), Tie=int((np.abs(d) <= 1e-6).sum()), Worse=int((d < -1e-6).sum())))
    sig = pd.DataFrame(sig)
    sig["Holm_p"] = holm(sig.Wilcoxon_p.fillna(1.0))
    sig.to_csv(T / "significance.csv", index=False, float_format="%.4g")

    seeds = [pooled(by[k])["IoU"] for k in ["E", "E_s1", "E_s2"] if k in by]
    pd.DataFrame([dict(Control="Random clicks (same number)", Seeds=len(seeds),
                       IoU_mean=np.mean(seeds), IoU_sd=np.std(seeds, ddof=1) if len(seeds) > 1 else np.nan)]
                 ).to_csv(T / "random_seeds.csv", index=False, float_format="%.4f")

    lk = lakes[lakes.config == "H"]
    conv = dict(lakes=len(lk), clicks_per_lake=lk.clicks.mean(), clicks_per_image=Hp.clicks.mean(),
                decoder_calls_per_image_B=B.dec_calls.mean(), decoder_calls_per_image_H=Hp.dec_calls.mean())
    conv.update({f"{c} clicks %": 100 * (lk.clicks == c).mean() for c in range(6)})
    conv.update({f"stop: {r} %": 100 * v for r, v in lk.reason.value_counts(normalize=True).items()})
    st = lk.dropna(subset=["stability"]).groupby("image").stability.mean()
    conv["stability_vs_IoU_spearman"] = spearmanr(st.values, Hp.loc[st.index].IoU.values).correlation
    pd.DataFrame([conv]).T.rename(columns={0: "value"}).to_csv(T / "convergence.csv", index_label="metric", float_format="%.4f")

    gb = pd.read_csv(run / "gbapr_prompts.csv")
    pd.DataFrame([{"lakes": len(gb),
                   "centroid point not on lake %": 100 * gb.centroid_off_lake.mean(),
                   "negative points per lake": gb.n_neg.mean(),
                   "lakes with a negative point on the lake %": 100 * gb.neg_on_lake.mean(),
                   "expanded-box area that is lake %": 100 * gb.ring_lake_px.sum() / max(gb.ring_px.sum(), 1)}]
                 ).T.rename(columns={0: "value"}).to_csv(T / "gbapr_diagnostic.csv", index_label="metric", float_format="%.2f")

    has = meta[meta.gt_area_px > 0]
    miss = {"images with a lake": len(has), "images with no YOLO detection": int((has.n_det == 0).sum()),
            "lake pixels in those images %": 100 * has[has.n_det == 0].gt_area_px.sum() / has.gt_area_px.sum()}
    mdir = run / "masks"
    if mdir.exists():
        cov = []
        for f in sorted(mdir.glob("*.npz")):
            z = np.load(f)
            n, cc, stt, _ = cv2.connectedComponentsWithStats(z["gt"].astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if stt[i, cv2.CC_STAT_AREA] < 100:
                    continue
                c = cc == i
                cov.append([(z[k] & c).sum() / c.sum() for k in ("A", "B", "H")])
        cov = np.array(cov)
        missed = cov[cov[:, 0] < 0.10]
        miss.update({"lakes >= 100 px": len(cov), "lakes missed by YOLO (<10% covered)": len(missed),
                     "missed lakes recovered by SAM2 (>=50% covered)": int((missed[:, 1] >= 0.5).sum()),
                     "missed lakes partly recovered by proposed (>=10% covered)": int((missed[:, 2] >= 0.1).sum())})
    pd.DataFrame([miss]).T.rename(columns={0: "value"}).to_csv(T / "missed_lakes.csv", index_label="metric", float_format="%.2f")

    tim = pd.read_csv(run / "timing.csv")
    ms = lambda g: 1000 * (g.t_yolo + g.t_enc + g.t_dec + g.t_post).mean()
    pd.DataFrame([{"YOLO ms": 1000 * tim.t_yolo.mean(), "SAM2 encoder ms": 1000 * tim.t_enc.mean(),
                   "SAM2 decoder ms per call": 1000 * Hp.t_dec.sum() / max(Hp.dec_calls.sum(), 1),
                   "decoder calls per image (B)": B.dec_calls.mean(), "decoder calls per image (H)": Hp.dec_calls.mean(),
                   "clicks per image (H)": Hp.clicks.mean(), "YOLO+SAM2 total ms": ms(B), "proposed total ms": ms(Hp),
                   "proposed images/s": 1000 / ms(Hp), "peak GPU memory GB": tim.peak_mem_MB.max() / 1024,
                   "parameters M": p_all, "device": info["gpu"]}]
                 ).T.rename(columns={0: "value"}).to_csv(T / "table7_compute.csv", index_label="metric")

    for f in sorted(T.glob("*.csv")):
        print(f"\n{f.stem}\n{pd.read_csv(f).to_string(index=False)}")


if __name__ == "__main__":
    main()
