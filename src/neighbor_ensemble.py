"""근처 발전소 일사량 + ML 예측값 앙상블.

(1) 교차검증: 일사량을 아는 위치를 좌표 단위 fold 로 나눠 hold-out 하고
    ML 단독 / 이웃 단순평균 / 이웃 거리역수 가중평균 / ML+이웃 앙상블 을 비교,
    가장 좋은 앙상블 설정 (k, power, w) 을 고른다.
(2) 최종 예측: 선택된 설정으로 test 발전소의 nins 를 예측해 제출 csv 로 저장.

    python src/neighbor_ensemble.py --data-dir .
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import ml_model
from common import load_data, neighbor_predict, scores

W_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def cross_validate(d, n_splits, stride, ks, powers, seed):
    xy, Y, times = d["src_xy"], d["src_Y"], d["times"]
    day = ml_model.day_mask(Y)
    oof_ml = np.zeros_like(Y)
    folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(xy))
    for i, (tr, va) in enumerate(folds, 1):
        print(f"[CV] fold {i}/{n_splits}: ML 학습 (train 위치 {len(tr)}, hold-out {len(va)})", flush=True)
        oof_ml[va] = ml_model.fit_predict(xy[tr], Y[tr], xy[va], times, stride=stride)

    rows = [dict(method="ML only", k=None, power=None, w_ml=1.0, **scores(Y, oof_ml, day))]
    for k, p in product(ks, powers):
        nb = np.zeros_like(Y)
        for tr, va in folds:
            nb[va] = neighbor_predict(xy[tr], Y[tr], xy[va], k=k, power=p)
        rows.append(dict(method=f"neighbor k={k} power={p}", k=k, power=p, w_ml=0.0,
                         **scores(Y, nb, day)))
        for w in W_GRID:                                   # ML + 이웃 앙상블
            rows.append(dict(method=f"ML*{w} + neighbor(k={k}, power={p})", k=k, power=p, w_ml=w,
                             **scores(Y, w * oof_ml + (1 - w) * nb, day)))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2, help="ML 학습 시 시간축 솎기 간격")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--powers", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--skip-cv", action="store_true", help="교차검증 생략 (--k/--power/--w 직접 지정)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--power", type=float, default=0)
    ap.add_argument("--w", type=float, default=0.5, help="ML 가중치 (이웃은 1-w)")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    d = load_data(a.data_dir)
    k, p, w = a.k, a.power, a.w
    if not a.skip_cv:
        res = cross_validate(d, a.folds, a.stride, a.ks, a.powers, a.seed)
        res.to_csv(out / "cv_results.csv", index=False)
        print("\n=== CV 결과 (MAE 낮은 순 상위 15) ===")
        print(res.head(15).to_string(float_format=lambda x: f"{x:.3f}"))
        best = res[(res.w_ml > 0) & (res.w_ml < 1)].iloc[0]   # 앙상블 중 최고 설정 채택
        k, p, w = int(best.k), float(best.power), float(best.w_ml)
        json.dump(dict(k=k, power=p, w_ml=w), open(out / "best_config.json", "w"))
    print(f"\n최종 설정: k={k}, power={p}, ML 가중치={w}")

    # ---- 최종 예측: 모든 train 위치로 학습/이웃 -> test 위치 ----
    xy, Y, txy, times = d["src_xy"], d["src_Y"], d["tgt_xy"], d["times"]
    ml = ml_model.fit_predict(xy, Y, txy, times, stride=a.stride)
    nb = neighbor_predict(xy, Y, txy, k=k, power=p)
    final = np.clip(w * ml + (1 - w) * nb, 0, None)

    # (위치, 시각) -> test.csv 순서의 (pv_id, time) 로 되돌리기
    t_pos = {t: i for i, t in enumerate(times)}
    pv_loc = {pv: i for i, pvs in enumerate(d["tgt_pv"]) for pv in pvs}
    test = d["test"]
    tt = pd.to_datetime(test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    li = test["pv_id"].map(pv_loc).to_numpy()
    ti = tt.map(t_pos).to_numpy()
    sub = pd.DataFrame({"time": test["time"], "pv_id": test["pv_id"], "nins": final[li, ti]})
    sub.to_csv(out / "submission.csv", index=False)
    print("saved", out / "submission.csv", sub.shape)


if __name__ == "__main__":
    main()
