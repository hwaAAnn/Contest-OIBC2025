"""ML 파트: 시간 피처 + 좌표 -> LightGBM 으로 일사량(nins) 예측.

TSbaseline.ipynb 의 파이프라인을 스크립트로 정리한 것.
  * 같은 좌표의 발전소는 평균 (average_nins_by_cluster)
  * 기상 변수는 전부 제거, 시간(sin/cos, 계절, year) + 좌표(coord1, coord2) 만 사용
  * 좌표 단위로 hold-out 해서 검증 -> 새로운 위치에 대한 일반화 성능 확인
  * 밤(모든 위치에서 nins==0 인 시각)은 학습에서 제외하고 0 으로 예측
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.7, subsample_freq=1, n_jobs=-1, random_state=47, verbose=-1)


def time_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """TSbaseline.ipynb 의 add_time_features 와 동일한 피처."""
    doy = times.dayofyear.to_numpy()
    minutes = (times.hour * 60 + times.minute).to_numpy()
    season = pd.Series(times.month).map(
        lambda m: 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3 if m in (9, 10, 11) else 4)
    return pd.DataFrame({
        "year": times.year.to_numpy(),
        "dayofyear_sin": np.sin(2 * np.pi * doy / 365), "dayofyear_cos": np.cos(2 * np.pi * doy / 365),
        "time_sin": np.sin(2 * np.pi * minutes / 1440), "time_cos": np.cos(2 * np.pi * minutes / 1440),
        "quarter": season.to_numpy(),
    })


def _frame(tf: pd.DataFrame, xy: np.ndarray, tidx: np.ndarray) -> pd.DataFrame:
    """(위치 x 시각) 전체를 long format 으로."""
    L = len(xy)
    X = tf.iloc[np.tile(tidx, L)].reset_index(drop=True)
    X.insert(0, "coord2", np.repeat(xy[:, 1], len(tidx)))
    X.insert(0, "coord1", np.repeat(xy[:, 0], len(tidx)))
    return X


def day_mask(src_Y: np.ndarray) -> np.ndarray:
    """어떤 위치에서든 일사량이 0 보다 큰 시각 (낮)."""
    return (src_Y > 0).any(axis=0)


def fit_predict(src_xy, src_Y, tgt_xy, times, stride: int = 2, params: dict | None = None) -> np.ndarray:
    """src 위치들로 학습해서 tgt 위치들의 일사량 (Lt, T) 을 예측.

    stride : 학습 시 시간축을 stride 간격으로 솎아 속도를 높인다 (예측은 전 시각).
    """
    tf = time_features(times)
    day = day_mask(src_Y)
    tr_t = np.where(day)[0][::stride]
    X = _frame(tf, src_xy, tr_t)
    y = src_Y[:, tr_t].reshape(-1)
    model = lgb.LGBMRegressor(**{**PARAMS, **(params or {})}).fit(X, y)

    pred = np.zeros((len(tgt_xy), len(times)))
    day_t = np.where(day)[0]
    p = model.predict(_frame(tf, tgt_xy, day_t)).reshape(len(tgt_xy), len(day_t))
    pred[:, day_t] = np.clip(p, 0, None)
    return pred
