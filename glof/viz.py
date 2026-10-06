import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

GT_C, PR_C, POS_C, NEG_C, MISS_C, FIX_C, OMIT_C = "#FFD400", "#00E5FF", "#22C55E", "#EF4444", "#F59E0B", "#22C55E", "#E055E0"


def iou(m, g):
    u = (m | g).sum()
    return 1.0 if u == 0 else (m & g).sum() / u


def contour(ax, m, col, lw=2.0, ls="-"):
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=lw, linestyles=ls)


def setup(ax, img):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)


def regions(M, min_px=60, k=2):
    M = cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
    idx = [i for i in np.argsort(-st[1:, cv2.CC_STAT_AREA])[:k] + 1 if st[i, cv2.CC_STAT_AREA] >= min_px]
    return [st[i, :4] for i in idx]


def tag(ax, box, text, col, size=9):
    x, y, w, h = box
    ax.add_patch(Rectangle((x - 3, y - 3), w + 6, h + 6, fill=False, ec=col, lw=2.0, ls="--"))
    ax.text(x, y - 6 if y > 22 else y + h + 16, text, fontsize=size, fontweight="bold", color="black",
            bbox=dict(fc=col, ec="none", pad=1.5, alpha=0.95))


def draw_clicks(ax, clicks, r, numbered=False):
    for c in clicks:
        col = POS_C if c["label"] == 1 else NEG_C
        ax.add_patch(Circle((c["x"], c["y"]), r, fc=col, ec="white", lw=1.2, zorder=5))
        txt = str(c.get("iter", "")) if numbered else ("+" if c["label"] == 1 else "−")
        ax.text(c["x"], c["y"], txt, color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)


def load_case(img_path, npz_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    d = np.load(npz_path)
    return img, d, json.loads(str(d["clicks"]))


def step_figure(items, out):
    """items: list of (image_path, npz_path, row_label)."""
    cols = ["1  Input image", "2  YOLOv11m-seg detection", "3  SAM2 from YOLO box",
            "4  Uncertainty + corrective clicks", "5  Proposed framework"]
    fig, axes = plt.subplots(len(items), 5, figsize=(15, 3.15 * len(items) + 0.6))
    axes = np.atleast_2d(axes)
    for r, (ip, zp, label) in enumerate(items):
        img, d, clicks = load_case(ip, zp)
        gt, A, B, Hm, U = d["gt"], d["A"], d["B"], d["H"], d["U"].astype(np.float32)
        a = axes[r]
        for x in a:
            setup(x, img if x is not a[3] else (img * 0.45).astype(np.uint8))
        contour(a[0], gt, GT_C, 1.2)
        a[0].set_ylabel(label, fontsize=9.5, fontweight="bold")
        contour(a[1], A, PR_C)
        for b in d["boxes"]:
            a[1].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=1.0))
        contour(a[2], B, PR_C)
        contour(a[2], gt, GT_C, 1.0, "--")
        inb = U > 0
        lo, hi = np.percentile(U[inb], [5, 95]) if inb.any() else (0, 1)
        a[3].imshow(np.ma.masked_where(~inb, np.clip((U - lo) / (hi - lo + 1e-6), 0, 1)), cmap="inferno", alpha=0.8)
        contour(a[3], A & ~B, POS_C, 1.2)
        contour(a[3], B & ~A, NEG_C, 1.2)
        draw_clicks(a[3], clicks, 7, numbered=True)
        contour(a[4], Hm, PR_C)
        contour(a[4], gt, GT_C, 1.0, "--")
        t = [f"IoU {iou(A, gt):.2f}", f"IoU {iou(B, gt):.2f}", f"{len(clicks)} click(s)",
             f"IoU {iou(Hm, gt):.2f}"]
        for x, s in zip(a[1:], t):
            x.set_title((cols[list(a).index(x)] + "\n" if r == 0 else "") + s, fontsize=9)
        if r == 0:
            a[0].set_title(cols[0], fontsize=9)
    handles = [Line2D([], [], color=GT_C, lw=2, label="Ground-truth boundary"),
               Line2D([], [], color=PR_C, lw=2, label="Predicted boundary"),
               Line2D([], [], color=POS_C, lw=2, label="D+: YOLO lake, SAM2 background"),
               Line2D([], [], color=NEG_C, lw=2, label="D−: SAM2 lake, YOLO background"),
               Line2D([], [], color="none", marker="o", mfc=POS_C, mec="white", ms=9, label="Positive click (iteration)"),
               Line2D([], [], color="none", marker="o", mfc=NEG_C, mec="white", ms=9, label="Negative click (iteration)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def case_figure(items, title, out, omission=()):
    """items: list of (image_path, npz_path). Columns: input, ground truth, YOLO+SAM2, proposed."""
    fig, axes = plt.subplots(len(items), 4, figsize=(15, 3.9 * len(items) + 0.8))
    axes = np.atleast_2d(axes)
    for r, (ip, zp) in enumerate(items):
        img, d, clicks = load_case(ip, zp)
        gt, B, Hm = d["gt"], d["B"], d["H"]
        a = axes[r]
        for x in a:
            setup(x, img)
        contour(a[1], gt, GT_C)
        contour(a[2], B, PR_C)
        contour(a[3], Hm, PR_C)
        for b in regions(B & ~gt & ~Hm):
            tag(a[2], b, "Not lake", NEG_C)
            tag(a[3], b, "Removed", FIX_C)
        for b in regions(gt & ~B & Hm):
            tag(a[2], b, "Lake part missed", MISS_C)
            tag(a[3], b, "Recovered", FIX_C)
        if str(ip) in omission:
            for b in regions(Hm & ~gt, min_px=400, k=1):
                tag(a[1], b, "Water not labelled", OMIT_C)
                tag(a[3], b, "Detected as lake", OMIT_C)
        draw_clicks(a[3], clicks, max(5, img.shape[1] / 60))
        a[2].set_title(f"YOLOv11m-seg + SAM2    IoU {iou(B, gt):.2f}", fontsize=11)
        a[3].set_title(f"Proposed framework    IoU {iou(Hm, gt):.2f}", fontsize=11)
        if r == 0:
            a[0].set_title("Input image", fontsize=11)
            a[1].set_title("Ground truth", fontsize=11)
        a[0].set_ylabel(f"({chr(97 + r)})", fontsize=12, fontweight="bold", rotation=0, labelpad=16)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    handles = [Line2D([], [], color=GT_C, lw=2.5, label="Ground-truth lake boundary"),
               Line2D([], [], color=PR_C, lw=2.5, label="Predicted lake boundary"),
               Line2D([], [], color="none", marker="o", mfc=POS_C, mec="white", ms=11, label="Positive click"),
               Line2D([], [], color="none", marker="o", mfc=NEG_C, mec="white", ms=11, label="Negative click")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10.5, frameon=False)
    plt.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def missed_lake_figure(items, title, out):
    """items: list of (image_path, npz_path, ground-truth lake component index)."""
    fig, axes = plt.subplots(len(items), 4, figsize=(15, 3.9 * len(items) + 0.8))
    axes = np.atleast_2d(axes)
    for r, (ip, zp, idx) in enumerate(items):
        img, d, _ = load_case(ip, zp)
        gt, A, Hm = d["gt"], d["A"], d["H"]
        _, cc = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        lake = cc == idx
        ys, xs = np.nonzero(lake)
        cx, cy, rad = xs.mean(), ys.mean(), max(16, 0.8 * max(np.ptp(xs), np.ptp(ys)))
        a = axes[r]
        for x in a:
            setup(x, img)
        contour(a[1], gt, GT_C)
        contour(a[2], A, PR_C)
        for b in d["boxes"]:
            a[2].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=1.6))
        contour(a[3], Hm, PR_C)
        cov = lambda m: 100 * (m & lake).sum() / lake.sum()
        for x, txt, col in [(a[1], "Lake", "#FF00FF"), (a[2], f"Missed by YOLO ({cov(A):.0f}%)", MISS_C),
                            (a[3], f"Found by SAM2 ({cov(Hm):.0f}%)", FIX_C)]:
            x.add_patch(Circle((cx, cy), rad, fill=False, ec=col, lw=2.4, ls="--"))
            x.text(cx, cy - rad - 8 if cy - rad > 25 else cy + rad + 18, txt, fontsize=9.5, fontweight="bold",
                   ha="center", bbox=dict(fc=col, ec="none", pad=1.5))
        if r == 0:
            a[0].set_title("Input image", fontsize=11)
            a[1].set_title("Ground truth", fontsize=11)
        a[2].set_title("YOLOv11m-seg (mask and boxes)", fontsize=11)
        a[3].set_title(f"Proposed framework    IoU {iou(Hm, gt):.2f}", fontsize=11)
        a[0].set_ylabel(f"({chr(97 + r)})", fontsize=12, fontweight="bold", rotation=0, labelpad=16)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    handles = [Line2D([], [], color=GT_C, lw=2.5, label="Ground-truth lake boundary"),
               Line2D([], [], color=PR_C, lw=2.5, label="Predicted lake boundary"),
               Line2D([], [], color="white", lw=0, marker="s", mfc="none", mec="black", ms=11, label="YOLO detection box")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5, frameon=False)
    plt.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def hard_case_figure(cases, title, out):
    """cases: list of dicts with img (RGB), gt, B, H, clicks and a metrics dict m."""
    def overlay(img, m, col, a=0.38):
        o = img.astype(np.float32).copy()
        o[m] = (1 - a) * o[m] + a * np.array(col, np.float32)
        return o.astype(np.uint8)

    def errboxes(ax, m, gt):
        for E, txt, col in [(m & ~gt, "Not lake", NEG_C), (gt & ~m, "Lake missed", MISS_C)]:
            for b in regions(E, min_px=max(80, 0.002 * m.size), k=2):
                tag(ax, b, txt, col, size=8)

    asp = max(c["gt"].shape[0] / c["gt"].shape[1] for c in cases)
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, (3.9 * asp + 0.95) * len(cases) + 1.0))
    axes = np.atleast_2d(axes)
    heads = ["Input image", "Ground truth", "YOLOv11m-seg + SAM2", "Proposed framework"]
    for r, c in enumerate(cases):
        img, gt, B, Hm, m = c["img"], c["gt"], c["B"], c["H"], c["m"]
        panels = [img, overlay(img, gt, (255, 212, 0)), overlay(img, B, (0, 229, 255)), overlay(img, Hm, (0, 229, 255))]
        for ax, p in zip(axes[r], panels):
            setup(ax, p)
        contour(axes[r][1], gt, GT_C)
        for col, mk, key in [(2, B, "YOLO+SAM2"), (3, Hm, "Proposed")]:
            contour(axes[r][col], mk, PR_C)
            contour(axes[r][col], gt, GT_C, 1.0)
            axes[r][col].set_title(f"IoU {m[key + ' IoU']:.3f} · Dice {m[key + ' Dice']:.3f}\n"
                                   f"P {m[key + ' Precision']:.3f} · R {m[key + ' Recall']:.3f}", fontsize=9)
        errboxes(axes[r][2], B, gt)
        errboxes(axes[r][3], Hm, gt)
        draw_clicks(axes[r][3], c["clicks"], max(6, gt.shape[1] / 70))
        axes[r][0].set_title(f"{len(c['clicks'])} corrective click(s)", fontsize=9)
        axes[r][1].set_title(f"{int(gt.sum()):,} lake pixels", fontsize=9)
        axes[r][0].set_ylabel(f"({chr(97 + r)})", fontsize=12, fontweight="bold", rotation=0, labelpad=14)
        if r == 0:
            for ax, t in zip(axes[r], heads):
                ax.set_title(t + "\n" + ax.get_title(), fontsize=10.5)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.997)
    h = [Patch(fc=GT_C, ec=GT_C, alpha=0.6, label="Ground-truth lake"), Patch(fc=PR_C, ec=PR_C, alpha=0.6, label="Predicted lake"),
         Line2D([], [], color="none", marker="o", mfc=POS_C, mec="white", ms=10, label="Positive click"),
         Line2D([], [], color="none", marker="o", mfc=NEG_C, mec="white", ms=10, label="Negative click"),
         Patch(fc="none", ec=NEG_C, ls="--", label="Not lake (false positive)"),
         Patch(fc="none", ec=MISS_C, ls="--", label="Lake missed (false negative)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 0.985))
    fig.savefig(out, dpi=140)
    plt.close(fig)
