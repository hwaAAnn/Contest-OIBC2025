"""좌표 단위 교차검증 + ML/이웃 앙상블 + 제출 파일 생성."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import ml_model
from .metrics import scores
from .neighbors import neighbor_predict

W_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def cross_validate(d, n_splits, stride, ks, powers, seed) -> pd.DataFrame:
    """일사량을 아는 위치를 좌표 단위 fold 로 hold-out 해서 방법별 성능 비교.

    ML 단독 / 이웃 평균(power=0) / 이웃 거리역수 가중평균(power>0) / ML+이웃 블렌딩.
    """
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
        for w in W_GRID:
            rows.append(dict(method=f"ML*{w} + neighbor(k={k}, power={p})", k=k, power=p, w_ml=w,
                             **scores(Y, w * oof_ml + (1 - w) * nb, day)))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def predict_test(d, k, power, w, stride) -> pd.DataFrame:
    """모든 train 위치로 학습/이웃 -> test 발전소 예측. test.csv 순서의 (time, pv_id, nins)."""
    xy, Y, txy, times = d["src_xy"], d["src_Y"], d["tgt_xy"], d["times"]
    ml = ml_model.fit_predict(xy, Y, txy, times, stride=stride)
    nb = neighbor_predict(xy, Y, txy, k=k, power=power)
    final = np.clip(w * ml + (1 - w) * nb, 0, None)

    t_pos = {t: i for i, t in enumerate(times)}
    pv_loc = {pv: i for i, pvs in enumerate(d["tgt_pv"]) for pv in pvs}
    test = d["test"]
    tt = pd.to_datetime(test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    li = test["pv_id"].map(pv_loc).to_numpy()
    ti = tt.map(t_pos).to_numpy()
    return pd.DataFrame({"time": test["time"], "pv_id": test["pv_id"], "nins": final[li, ti]})
