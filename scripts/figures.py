import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glof.viz import case_figure, missed_lake_figure, step_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/kaggle_test/images")
    ap.add_argument("--run", default="results/test")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    img_dir, z_dir, out = Path(a.images), Path(a.run) / "masks", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def it(n):
        ip = next(img_dir.glob(f"image_{n}.*"))
        return ip, z_dir / f"{ip.stem}.npz"

    step_figure([(*it(246), "Glacier ice and snow"), (*it(3), "Terrain shadow,\nlow-contrast edge"),
                 (*it(217), "Debris-covered ice\nand terrain shadow"), (*it(112), "Bright snow / haze\nsurroundings")],
                out / "fig06_step_by_step.png")
    case_figure([it(246), it(99), it(187)], "Glacier ice and snow", out / "fig07_snow_ice.png")
    case_figure([it(112), it(315), it(254)], "Bright cloud / haze-like surroundings", out / "fig08_cloud_haze.png")
    case_figure([it(391), it(145), it(153)], "Low-contrast lake boundary and lake water missing from the ground truth",
                out / "fig09_low_contrast.png", omission={str(it(391)[0])})
    step_figure([(*it(164), "Failure: glacier ice\nnext to the lake"), (*it(113), "Failure: dark lake,\nno snow")],
                out / "fig13_failures.png")
    missed_lake_figure([(*it(231), 41), (*it(342), 4), (*it(195), 10)],
                       "Lake missed by YOLOv11m-seg but segmented by SAM2", out / "fig14_missed_lakes.png")


if __name__ == "__main__":
    main()
