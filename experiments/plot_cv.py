"""outputs/cv_results.csv -> 방법별 CV 성능 비교 그래프 (docs/cv_comparison.png).

    python experiments/plot_cv.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    r = pd.read_csv(ROOT / "outputs" / "cv_results.csv")
    pick = [
        ("ML 단독 (LightGBM)", r[r.method == "ML only"]),
        ("이웃 단순평균 (k=3)", r[(r.k == 3) & (r.power == 0) & (r.w_ml == 0)]),
        ("이웃 거리역수 (k=5, 1/d)", r[(r.k == 5) & (r.power == 1) & (r.w_ml == 0)]),
        ("ML 0.2 + 이웃 거리역수", r[(r.k == 5) & (r.power == 1) & (r.w_ml == 0.2)]),
    ]
    names = [n for n, _ in pick]
    mae = [df.MAE.iloc[0] for _, df in pick]
    rmse = [df.RMSE.iloc[0] for _, df in pick]
    colors = ["#9e9e9e", "#1f77b4", "#2ca02c", "#ff7f0e"]

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, title in zip(axes, (mae, rmse), ("MAE (작을수록 좋음)", "RMSE (작을수록 좋음)")):
        bars = ax.barh(names, vals, color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlim(0, max(vals) * 1.15)
        ax.grid(axis="x", alpha=0.25)
    axes[1].set_yticklabels([])
    fig.suptitle("좌표 단위 5-fold CV 성능 비교")
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "cv_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
