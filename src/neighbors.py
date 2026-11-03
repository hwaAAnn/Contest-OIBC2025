"""근처 발전소(위치) 일사량으로 대상 위치를 예측."""
import numpy as np


def neighbor_predict(src_xy, src_Y, tgt_xy, k: int = 3, power: float = 0.0,
                     eps: float = 1e-3) -> np.ndarray:
    """가장 가까운 k개 위치의 일사량으로 대상 위치 예측.

    power = 0 : 단순 평균 (우리 팀이 쓴 방식)
    power > 0 : 거리 역수 가중평균 w = 1 / d^power (1등 팀 방식)
    """
    d = np.linalg.norm(tgt_xy[:, None, :] - src_xy[None, :, :], axis=2)   # (Lt, Ls)
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    w = np.ones_like(dk) if power == 0 else 1.0 / np.maximum(dk, eps) ** power
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("lk,lkt->lt", w, src_Y[idx])
