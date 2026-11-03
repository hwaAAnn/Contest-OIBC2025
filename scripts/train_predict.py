"""CV 로 (k, power, w) 선택 -> test 발전소 일사량 예측 -> 제출 파일 저장.

    python scripts/train_predict.py --data-dir .
    python scripts/train_predict.py --skip-cv --k 5 --power 1 --w 0.2   # 빠른 실행
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data
from src.pipeline import cross_validate, predict_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3, help="ML 학습 시 시간축 솎기 간격")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--powers", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--skip-cv", action="store_true", help="교차검증 생략 (--k/--power/--w 직접 지정)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--w", type=float, default=0.2, help="ML 가중치 (이웃은 1-w)")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    d = load_data(a.data_dir)
    k, p, w = a.k, a.power, a.w
    if not a.skip_cv:
        res = cross_validate(d, a.folds, a.stride, a.ks, a.powers, a.seed)
        res.to_csv(out / "cv_results.csv", index=False)
        print("\n=== CV 결과 (MAE 낮은 순 상위 15) ===")
        print(res.head(15).to_string(float_format=lambda x: f"{x:.3f}"))
        best = res[(res.w_ml > 0) & (res.w_ml < 1)].iloc[0]   # 블렌딩 중 최고 설정 채택
        k, p, w = int(best.k), float(best.power), float(best.w_ml)
        json.dump(dict(k=k, power=p, w_ml=w), open(out / "best_config.json", "w"))
    print(f"\n최종 설정: k={k}, power={p}, ML 가중치={w}")

    sub = predict_test(d, k, p, w, a.stride)
    sub.to_csv(out / "submission.csv", index=False)
    print("saved", out / "submission.csv", sub.shape)


if __name__ == "__main__":
    main()
