import math
import time
import hashlib

import cv2
import numpy as np
import torch
from scipy.stats import rankdata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = DEVICE == "cuda"

P = dict(
    YOLO_IMGSZ=640, YOLO_CONF=0.25, YOLO_IOU=0.45, YOLO_MAX_DET=100,
    GBAPR_BOX_EXPANSION=0.06, GBAPR_NEGATIVE_POINTS=4, GBAPR_NEGATIVE_MARGIN=0.06,
    MAX_ITERS=5, CONVERGE_IOU=0.95, TAU=0.5,
    MIN_DISAGREE_PX=16, MIN_DISAGREE_FRAC=0.002, MORPH_KERNEL_SIZE=3,
)

CONFIGS = {
    "A": dict(name="YOLOv11m-seg only", sam=False),
    "B": dict(name="YOLOv11m-seg + SAM2 (box prompt)"),
    "C": dict(name="+ Boundary-aware box", start="xbox"),
    "D": dict(name="+ Boundary-aware box and points", start="gbapr"),
    "E": dict(name="Box prompt + random clicks (same number)", mode="random", budget_from="H"),
    "F": dict(name="Box prompt + distance-based clicks (same number)", mode="distance", budget_from="H"),
    "I": dict(name="Box prompt + predictive-entropy clicks", mode="pred_entropy"),
    "H": dict(name="Box prompt + attention-entropy clicks (proposed)", mode="attn"),
    "J": dict(name="Full (boundary-aware box, points and attention-entropy clicks)", start="gbapr", mode="attn"),
    "E_s1": dict(name="Random clicks, seed 1", mode="random", budget_from="H", seed=1),
    "E_s2": dict(name="Random clicks, seed 2", mode="random", budget_from="H", seed=2),
}
ABLATION_ORDER = ["A", "B", "C", "D", "E", "F", "I", "H", "J"]


def tic():
    if CUDA:
        torch.cuda.synchronize()
    return time.perf_counter()


def toc(t0):
    if CUDA:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def cleanup(mask):
    k = P["MORPH_KERNEL_SIZE"]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (np.asarray(mask) > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker).astype(bool)


def confusion(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    TP = int(np.count_nonzero(p & g))
    FP = int(np.count_nonzero(p & ~g))
    FN = int(np.count_nonzero(~p & g))
    return TP, FP, FN, int(p.size - TP - FP - FN)


def metrics_from(TP, FP, FN, TN):
    u = TP + FP + FN
    rec = TP / (TP + FN) if TP + FN else 0.0
    spec = TN / (TN + FP) if TN + FP else 0.0
    return dict(
        IoU=TP / u if u else 1.0,
        Dice=2 * TP / (2 * TP + FP + FN) if u else 1.0,
        Precision=TP / (TP + FP) if TP + FP else 0.0,
        Recall=rec,
        Specificity=spec,
        Pixel_Accuracy=(TP + TN) / max(TP + TN + FP + FN, 1),
        Balanced_Accuracy=(rec + spec) / 2,
    )


def mask_iou(a, b):
    a, b = a > 0, b > 0
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def clip_box(box, W, H):
    x1, y1, x2, y2 = np.asarray(box, np.float32).reshape(-1)[:4]
    x1, x2 = sorted([float(np.clip(x1, 0, W - 1)), float(np.clip(x2, 0, W - 1))])
    y1, y2 = sorted([float(np.clip(y1, 0, H - 1)), float(np.clip(y2, 0, H - 1))])
    return np.array([x1, y1, x2, y2], np.float32)


def expand_box(box, a, W, H):
    x1, y1, x2, y2 = np.asarray(box, np.float32)
    dx, dy = a * max(1.0, x2 - x1), a * max(1.0, y2 - y1)
    return clip_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], W, H)


def box_region(box, H, W):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    c = np.zeros((H, W), bool)
    c[max(0, y1):min(H, y2 + 1), max(0, x1):min(W, x2 + 1)] = True
    return c


def detect(model, img):
    H, W = img.shape[:2]
    r = model.predict(source=img, imgsz=P["YOLO_IMGSZ"], conf=P["YOLO_CONF"], iou=P["YOLO_IOU"],
                      max_det=P["YOLO_MAX_DET"], retina_masks=True, device=DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    dets = []
    for i, b in enumerate(boxes):
        b = clip_box(b, W, H)
        if b[2] - b[0] < 1 or b[3] - b[1] < 1:
            continue
        if masks is not None and i < len(masks):
            m = masks[i] > 0.5
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        else:
            m = np.zeros((H, W), bool)
        if not m.any():
            x1, y1, x2, y2 = b.astype(int)
            m[y1:y2 + 1, x1:x2 + 1] = True
        dets.append(dict(box=b, conf=float(confs[i]), mask=m))
    dets.sort(key=lambda d: -d["conf"])
    return dets


class SAM2Runner:
    def __init__(self, predictor):
        self.pred = predictor
        self.xattn = predictor.model.sam_mask_decoder.transformer.layers[-1].cross_attn_image_to_token
        self.qk = {"q": None, "k": None}

        def prehook(module, args, kwargs):
            q = kwargs.get("q", args[0] if len(args) > 0 else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            self.qk["q"] = q.detach() if q is not None else None
            self.qk["k"] = k.detach() if k is not None else None

        self.xattn.register_forward_pre_hook(prehook, with_kwargs=True)

    def set_image(self, img_bgr):
        with torch.inference_mode():
            self.pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def call(self, box, pts, lbl, lowres, stats):
        self.qk["q"] = self.qk["k"] = None
        kw = dict(box=np.asarray(box, np.float32), multimask_output=True, return_logits=True)
        if pts is not None and len(pts):
            kw["point_coords"] = np.asarray(pts, np.float32)
            kw["point_labels"] = np.asarray(lbl, np.int32)
        if lowres is not None:
            kw["mask_input"] = np.asarray(lowres, np.float32)[None]
        t0 = tic()
        with torch.inference_mode():
            logits, scores, low = self.pred.predict(**kw)
        stats["t_dec"] += toc(t0)
        stats["dec_calls"] += 1
        b = int(np.argmax(scores))
        return dict(logits=logits, mask=logits[b] > 0.0, lowres=low[b],
                    q=self.qk["q"], k=self.qk["k"], cache={})


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)


@torch.no_grad()
def u_attention(R, call, H, W):
    q, k = call["q"], call["k"]
    if q is None or k is None:
        return np.zeros((H, W), np.float32)
    X = R.xattn
    Q = X._separate_heads(X.q_proj(q.float()), X.num_heads)
    K = X._separate_heads(X.k_proj(k.float()), X.num_heads)
    A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]), dim=-1)
    ent = (-(A * torch.log(A + 1e-8)).sum(-1) / math.log(A.shape[-1])).mean(1)[0]
    side = int(round(math.sqrt(ent.shape[0])))
    m = ent[:side * side].reshape(side, side).cpu().numpy()
    return minmax(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR))


def u_predictive(R, call, H, W):
    p = 1.0 / (1.0 + np.exp(-np.clip(call["logits"], -30, 30)))
    pb = np.clip(p.mean(0), 1e-6, 1 - 1e-6)
    return (-(pb * np.log2(pb) + (1 - pb) * np.log2(1 - pb))).astype(np.float32)


def u_boundary(pred):
    edge = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 5)
    return np.exp(-dist / 5.0).astype(np.float32)


U_FUNCS = {"attn": u_attention, "pred_entropy": u_predictive}


def uncertainty(R, mode, call, H, W):
    if mode not in call["cache"]:
        call["cache"][mode] = U_FUNCS[mode](R, call, H, W)
    return call["cache"][mode]


def auroc(score, label):
    label = label.astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stable_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def gbapr_prompts(det, W, H):
    mask = det["mask"]
    x1, y1, x2, y2 = det["box"]
    rbox = expand_box(det["box"], P["GBAPR_BOX_EXPANSION"], W, H)
    ys, xs = np.nonzero(mask)
    if len(xs):
        cx, cy = float(xs.mean()), float(ys.mean())
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
    else:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bx1, by1, bx2, by2 = x1, y1, x2, y2
    cx, cy = float(np.clip(cx, 0, W - 1)), float(np.clip(cy, 0, H - 1))
    mx = max(2, int((bx2 - bx1) * P["GBAPR_NEGATIVE_MARGIN"]))
    my = max(2, int((by2 - by1) * P["GBAPR_NEGATIVE_MARGIN"]))
    negs = []
    for px, py in [[bx1 - mx, cy], [bx2 + mx, cy], [cx, by1 - my], [cx, by2 + my]]:
        px, py = float(np.clip(px, 0, W - 1)), float(np.clip(py, 0, H - 1))
        if not mask[int(round(py)), int(round(px))]:
            negs.append([px, py])
    negs = negs[:P["GBAPR_NEGATIVE_POINTS"]]
    return rbox, np.asarray([[cx, cy]] + negs, np.float32), np.asarray([1] + [0] * len(negs), np.int32)


def start_prompts(det, start, W, H):
    if start == "box":
        return det["box"], None, None
    if start == "xbox":
        return expand_box(det["box"], P["GBAPR_BOX_EXPANSION"], W, H), None, None
    if start == "gbapr":
        return gbapr_prompts(det, W, H)
    raise ValueError(start)


def clean_region(D, min_area):
    D = cv2.morphologyEx(D.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(D, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def disagreement(cur, ymask, ctx):
    min_area = max(P["MIN_DISAGREE_PX"], P["MIN_DISAGREE_FRAC"] * ctx.sum())
    d_pos = clean_region(ymask & ~cur & ctx, min_area)
    d_neg = clean_region(cur & ~ymask & ctx, min_area)
    return d_pos, d_neg


def choose_click(mode, U, cur, ymask, ctx, rng):
    d_pos, d_neg = disagreement(cur, ymask, ctx)
    cands = []
    for D, lab in ((d_pos, 1), (d_neg, 0)):
        if not D.any():
            continue
        if mode in U_FUNCS:
            G = D & (U >= P["TAU"])
            if not G.any():
                continue
            n, cc = cv2.connectedComponents(G.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                Sc = np.where(cc == i, U, -1.0)
                y, x = np.unravel_index(int(np.argmax(Sc)), Sc.shape)
                cands.append((float(Sc[y, x]), dict(x=int(x), y=int(y), label=lab)))
        elif mode == "distance":
            n, cc = cv2.connectedComponents(D.astype(np.uint8), connectivity=8)
            dist = cv2.distanceTransform(np.pad(D.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            for i in range(1, n):
                Sc = np.where(cc == i, dist, -1.0)
                y, x = np.unravel_index(int(np.argmax(Sc)), Sc.shape)
                cands.append((float(Sc[y, x]), dict(x=int(x), y=int(y), label=lab)))
        elif mode == "random":
            n, cc = cv2.connectedComponents(D.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                pix = np.argwhere(cc == i)
                y, x = pix[rng.integers(len(pix))]
                cands.append((float(rng.random()), dict(x=int(x), y=int(y), label=lab)))
        else:
            raise ValueError(mode)
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    return cands[0][1]


def refine_lake(R, det, cfg, stats, rng, budget, H, W):
    box, p0, l0 = start_prompts(det, cfg.get("start", "box"), W, H)
    ymask = det["mask"]
    call0 = call = R.call(box, p0, l0, None, stats)
    mask = call["mask"]
    hist, clicks, reason, stab = [mask], [], "no_loop", float("nan")
    mode = cfg.get("mode")
    if mode:
        reason = "max_iters"
        ctx = box_region(box, H, W)
        base_pts = np.zeros((0, 2), np.float32) if p0 is None else np.asarray(p0, np.float32)
        base_lbl = np.zeros((0,), np.int32) if l0 is None else np.asarray(l0, np.int32)
        left = budget if budget is not None else 10 ** 9
        for _ in range(P["MAX_ITERS"] if budget is None else 10 ** 6):
            if left <= 0:
                reason = "budget"
                break
            U = uncertainty(R, mode, call, H, W) if mode in U_FUNCS else None
            ck = choose_click(mode, U, mask, ymask, ctx, rng)
            if ck is None:
                reason = "no_uncertain_disagreement"
                break
            clicks.append(ck)
            left -= 1
            pts = np.vstack([base_pts, np.array([[c["x"], c["y"]] for c in clicks], np.float32)])
            lbl = np.concatenate([base_lbl, np.array([c["label"] for c in clicks], np.int32)])
            new_call = R.call(box, pts, lbl, call["lowres"], stats)
            if call is not call0:
                call["logits"] = None
                call["cache"] = {}
            call = new_call
            stab = mask_iou(call["mask"], mask)
            mask = call["mask"]
            hist.append(mask)
            if stab >= P["CONVERGE_IOU"]:
                reason = "converged"
                break
    return dict(mask=mask, hist=hist, clicks=clicks, reason=reason, stability=stab, call0=call0, box=box,
                prompts=(p0, l0))


def gbapr_diagnostic(det, gt, H, W):
    rbox, pts, lbl = gbapr_prompts(det, W, H)
    cx, cy = pts[0]
    negs = pts[1:]
    ring = box_region(rbox, H, W) & ~box_region(det["box"], H, W)
    return dict(centroid_off_lake=int(not gt[int(round(cy)), int(round(cx))]),
                n_neg=len(negs),
                neg_on_lake=int(any(gt[int(round(y)), int(round(x))] for x, y in negs)),
                ring_px=int(ring.sum()), ring_lake_px=int((ring & gt).sum()))


def uq_samples(R, r, gt, H, W, key):
    call0 = r["call0"]
    pred = call0["mask"]
    ctx = box_region(r["box"], H, W)
    err = (pred != gt)[ctx]
    maps = {"attention_entropy": uncertainty(R, "attn", call0, H, W),
            "boundary_distance": u_boundary(pred),
            "predictive_entropy": uncertainty(R, "pred_entropy", call0, H, W)}
    rng = np.random.default_rng(stable_seed(*key))
    idx = rng.choice(err.size, min(err.size, 3000), replace=False)
    return dict(err=err[idx].astype(np.uint8), **{k: v[ctx][idx].astype(np.float32) for k, v in maps.items()})


def run_image(R, dets, H, W, configs, name, gt=None, keep_hist=("H",), diag=True):
    out, budgets = {}, {}
    extra = dict(lakes=[], uq=[], gbapr=[], U=np.zeros((H, W), np.float32))
    order = [k for k in configs if not configs[k].get("budget_from")] + \
            [k for k in configs if configs[k].get("budget_from")]
    for key in order:
        cfg = configs[key]
        st = dict(t_dec=0.0, dec_calls=0)
        info = dict(n_det=len(dets), clicks=0, pos_clicks=0, neg_clicks=0)
        m = np.zeros((H, W), bool)
        hist_union, clicks_all = None, []
        if not cfg.get("sam", True):
            for d in dets:
                m |= d["mask"]
        else:
            rng = np.random.default_rng(stable_seed(name, key, cfg.get("seed", 0)))
            hist_acc = [np.zeros((H, W), bool) for _ in range(P["MAX_ITERS"] + 1)] \
                if (cfg.get("mode") and key in keep_hist) else None
            for li, det in enumerate(dets):
                budget = budgets.get((cfg["budget_from"], li), 0) if cfg.get("budget_from") else None
                r = refine_lake(R, det, cfg, st, rng, budget, H, W)
                budgets[(key, li)] = len(r["clicks"])
                m |= r["mask"]
                info["clicks"] += len(r["clicks"])
                info["pos_clicks"] += sum(c["label"] == 1 for c in r["clicks"])
                clicks_all += [dict(lake=li, iter=t + 1, **c) for t, c in enumerate(r["clicks"])]
                if hist_acc is not None:
                    for t in range(P["MAX_ITERS"] + 1):
                        hist_acc[t] |= r["hist"][min(t, len(r["hist"]) - 1)]
                if cfg.get("mode") and not cfg.get("budget_from"):
                    extra["lakes"].append(dict(config=key, lake=li, clicks=len(r["clicks"]), reason=r["reason"],
                                               stability=r["stability"]))
                if diag and gt is not None and key == "B":
                    extra["uq"].append(uq_samples(R, r, gt, H, W, (name, li)))
                if key == "H":
                    ctx = box_region(r["box"], H, W)
                    extra["U"] = np.where(ctx, np.maximum(extra["U"], uncertainty(R, "attn", r["call0"], H, W)), extra["U"])
                if diag and gt is not None and key == "D":
                    extra["gbapr"].append(gbapr_diagnostic(det, gt, H, W))
            info["neg_clicks"] = info["clicks"] - info["pos_clicks"]
            if hist_acc is not None:
                hist_union = [cleanup(h) for h in hist_acc]
        t0 = time.perf_counter()
        fm = cleanup(m)
        info.update(dec_calls=st["dec_calls"], t_dec=st["t_dec"], t_post=time.perf_counter() - t0)
        out[key] = dict(mask=fm, hist=hist_union, info=info, clicks=clicks_all)
    return out, extra
