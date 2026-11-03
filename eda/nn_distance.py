"""최근접 이웃까지의 거리 분포: train 끼리 vs test -> train.

test 발전소는 train 끼리보다 이웃이 멀다 => 검증을 좌표 단위로 해야 하는 근거.

    python eda/nn_distance.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_data  # noqa: E402


def nearest(a, b, skip_self=False):
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    if skip_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def main():
    d = load_data(".")
    sxy, txy = d["src_xy"], d["tgt_xy"]
    tr, te = nearest(sxy, sxy, True), nearest(txy, sxy)
    print(f"train->train 최근접 거리 median={np.median(tr):.3f}, test->train median={np.median(te):.3f}")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    bins = np.linspace(0, 0.7, 25)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(tr, bins, alpha=0.7, color="#1f77b4", label=f"train → train (중앙값 {np.median(tr):.3f})")
    ax.hist(te, bins, alpha=0.7, color="#d62728", label=f"test → train (중앙값 {np.median(te):.3f})")
    ax.set_xlabel("가장 가까운 일사량 보유 위치까지의 거리"); ax.set_ylabel("위치 수")
    ax.set_title("최근접 이웃 거리 분포"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "nn_distance.png", dpi=150)


if __name__ == "__main__":
    main()
