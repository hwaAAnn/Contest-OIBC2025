"""발전소 좌표 시각화: 일사량을 아는 발전소(train) vs 예측 대상 발전소(test).

    python eda/plot_map.py --k 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_data  # noqa: E402

TRAIN_COLOR, TEST_COLOR = "#1f77b4", "#d62728"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--k", type=int, default=3, help="대상 발전소마다 이어줄 이웃 수")
    ap.add_argument("--out", default="docs/coords_map.png")
    a = ap.parse_args()

    d = load_data(a.data_dir)
    sxy, txy = d["src_xy"], d["tgt_xy"]
    dist = np.linalg.norm(txy[:, None] - sxy[None], axis=2)
    nn = np.argsort(dist, axis=1)[:, :a.k]

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, row in enumerate(nn):                       # 이웃 연결선
        for j in row:
            ax.plot([txy[i, 0], sxy[j, 0]], [txy[i, 1], sxy[j, 1]],
                    color="gray", lw=0.6, alpha=0.5, zorder=1)
    n_pv = np.array([len(p) for p in d["src_pv"]])
    ax.scatter(sxy[:, 0], sxy[:, 1], s=25 + 25 * n_pv, c=TRAIN_COLOR, alpha=0.75,
               edgecolor="white", lw=0.6, zorder=2,
               label=f"일사량 있음 (train) - 좌표 {len(sxy)}곳 / 발전소 {n_pv.sum()}개")
    ax.scatter(txy[:, 0], txy[:, 1], s=150, c=TEST_COLOR, marker="*",
               edgecolor="black", lw=0.5, zorder=3,
               label=f"일사량 없음 (test) - 예측 대상 {len(txy)}개")
    for (x, y), pv in zip(txy, d["tgt_pv"]):
        ax.annotate(pv[0].replace("PV_ID_", ""), (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=TEST_COLOR)
    ax.set_xlabel("coord1"); ax.set_ylabel("coord2")
    ax.set_title(f"발전소 좌표 분포 (회색선: 대상 발전소 -> 가까운 {a.k}개 이웃)")
    ax.legend(loc="best"); ax.grid(alpha=0.25)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
