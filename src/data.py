"""데이터 로딩: train/test csv -> 좌표 단위 일사량 행렬.

핵심 아이디어
-------------
* train 의 183개 발전소는 실제로는 88개의 서로 다른 좌표에 모여 있다.
  같은 좌표의 발전소는 평균내어 "위치(location) x 시간" 일사량 행렬 Y 로 만든다.
* test 의 27개 발전소는 일사량(nins)이 비어 있고, train 과 시간축이 동일하다.
  => 문제는 "새 위치에서의 일사량 보간(spatial interpolation)" 이 된다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COORD = ["coord1", "coord2"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _location_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """좌표 -> location id (좌표가 같으면 같은 위치)."""
    keys = df[COORD].round(6).astype(str).agg("_".join, axis=1) if len(df) < 10_000 else None
    if keys is not None:
        codes, uniq = pd.factorize(keys)
    else:
        codes, uniq = pd.factorize(
            df["coord1"].round(6).astype(str) + "_" + df["coord2"].round(6).astype(str)
        )
    return codes, uniq


def load_data(data_dir: str | Path = ".", use_cache: bool = True) -> dict:
    """train/test csv -> 위치 단위 데이터.

    returns
    -------
    dict(
        times   : DatetimeIndex (T,)           공통 시간축 (train/test 동일)
        src_xy  : (Ls, 2)  일사량을 아는 위치 좌표
        src_Y   : (Ls, T)  위치별 일사량 (같은 좌표 발전소 평균)
        src_pv  : list[list[str]] 위치별 pv_id
        tgt_xy  : (Lt, 2)  예측 대상 위치 좌표
        tgt_pv  : list[list[str]]
        test    : DataFrame(time, pv_id)  제출 파일 순서 (test.csv 순서)
    )
    """
    data_dir = Path(data_dir)
    cache = CACHE_DIR / "locations.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        times = pd.DatetimeIndex(z["times"]).tz_localize("UTC").tz_convert("Asia/Seoul")
        return dict(times=times, src_xy=z["src_xy"], src_Y=z["src_Y"], src_pv=list(z["src_pv"]),
                    tgt_xy=z["tgt_xy"], tgt_pv=list(z["tgt_pv"]),
                    test=pd.read_pickle(CACHE_DIR / "test_index.pkl"))

    # ---- train: (pv, time) -> (location, time) 평균 ----
    tr = pd.read_csv(data_dir / "train.csv", usecols=["time", "pv_id", *COORD, "nins"])
    tcodes, tuniq = pd.factorize(tr["time"])
    lcodes, luniq = _location_ids(tr)
    T, L = len(tuniq), len(luniq)
    s = np.zeros((L, T)); c = np.zeros((L, T))
    ok = tr["nins"].notna().to_numpy()
    np.add.at(s, (lcodes[ok], tcodes[ok]), tr["nins"].to_numpy()[ok])
    np.add.at(c, (lcodes[ok], tcodes[ok]), 1)
    Y = np.where(c > 0, s / np.maximum(c, 1), np.nan)

    order = np.argsort(pd.to_datetime(tuniq, utc=True).to_numpy())  # 시간 정렬
    times = pd.to_datetime(tuniq[order], utc=True).tz_convert("Asia/Seoul")
    Y = Y[:, order]
    first = tr.groupby(lcodes).agg(c1=("coord1", "first"), c2=("coord2", "first"))
    src_xy = first[["c1", "c2"]].to_numpy()
    src_pv = [sorted(tr.loc[lcodes == i, "pv_id"].unique()) for i in range(L)]
    # 관측이 비는 시각(드묾)은 해당 시각의 다른 위치 평균으로 채우지 않고 시간 보간
    Y = pd.DataFrame(Y.T).interpolate(limit_direction="both").to_numpy().T
    del tr

    # ---- test: 좌표/제출 순서만 필요 ----
    te = pd.read_csv(data_dir / "test.csv", usecols=["time", "pv_id", *COORD])
    tl, tu = _location_ids(te)
    tgt_xy = te.groupby(tl).agg(c1=("coord1", "first"), c2=("coord2", "first")).to_numpy()
    tgt_pv = [sorted(te.loc[tl == i, "pv_id"].unique()) for i in range(len(tu))]
    test = te[["time", "pv_id"]].copy()

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, times=times.tz_convert("UTC").tz_localize(None).to_numpy(), src_xy=src_xy,
             src_Y=Y, src_pv=np.array(src_pv, dtype=object), tgt_xy=tgt_xy,
             tgt_pv=np.array(tgt_pv, dtype=object))
    test.to_pickle(CACHE_DIR / "test_index.pkl")
    return dict(times=times, src_xy=src_xy, src_Y=Y, src_pv=src_pv,
                tgt_xy=tgt_xy, tgt_pv=tgt_pv, test=test)
