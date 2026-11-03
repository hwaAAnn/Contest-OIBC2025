"""평가 지표."""
import numpy as np


def scores(y_true: np.ndarray, y_pred: np.ndarray, day: np.ndarray | None = None) -> dict:
    """MAE / RMSE (전체 시각) + 주간(day) MAE."""
    err = y_pred - y_true
    out = {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean()))}
    if day is not None:
        out["MAE_day"] = float(np.abs(err[:, day]).mean())
    return out
