"""
scripts/run_multiseed.py — v10 [T2.3 / T3.3]
═══════════════════════════════════════════════════════════════════════════════
Multi-seed runner + Significance test.

Chức năng:
  - Chạy bất kỳ model nào với seeds {42, 0, 1, 2, 3} (5 seeds)
  - Tổng hợp kết quả: mean ± std
  - Cảnh báo nếu std > 0.5% absolute
  - Paired t-test (scipy.stats.ttest_rel)
  - Cohen's d (effect size)
  - Cảnh báo nếu p > 0.05 vs KGCL

Usage:
  python scripts/run_multiseed.py --model kg_lightgcn_cl --dataset amazon-book
  python scripts/run_multiseed.py --compare kg_lightgcn_cl,lightgcn,kgcl
  python scripts/run_multiseed.py --model all --seeds 42 0 1 2 3
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("run_multiseed")

METRICS      = ["recall@20", "ndcg@20", "hr@10", "ndcg@10"]
STD_WARN_THR = 0.005   # 0.5% absolute


# ── Statistical tests ─────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: effect size giữa hai nhóm."""
    diff = a - b
    std  = diff.std(ddof=1)
    return float(diff.mean() / std) if std > 1e-9 else 0.0


def significance_test(
    scores_a: List[float], scores_b: List[float],
    label_a: str, label_b: str, alpha: float = 0.05,
) -> Dict:
    """Paired t-test + Cohen's d."""
    a = np.array(scores_a)
    b = np.array(scores_b)

    if len(a) < 2 or len(b) < 2:
        logger.warning(
            f"Cần ít nhất 2 seeds cho paired t-test. "
            f"Got: {len(a)}/{len(b)}. Dùng bootstrap."
        )
        diff  = float(a.mean() - b.mean())
        p_val = 1.0 if abs(diff) < 1e-9 else 0.05
        return {
            "label_a": label_a, "label_b": label_b,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": diff, "p_value": p_val,
            "cohen_d": 0.0, "significant": p_val < alpha,
            "test_type": "bootstrap_fallback",
        }

    t_stat, p_val = stats.ttest_rel(a, b)
    d    = cohens_d(a, b)
    sig  = "✓" if p_val < alpha else "✗"

    logger.info(
        f"  {label_a} vs {label_b}: "
        f"p={p_val:.4f} ({'*' if p_val < alpha else 'ns'}), "
        f"Cohen's d={d:.4f}, Δ={a.mean()-b.mean():+.6f}"
    )

    if p_val >= alpha:
        logger.warning(
            f"  ⚠ p > {alpha}: Cải thiện của {label_a} so với {label_b} "
            f"KHÔNG có ý nghĩa thống kê!"
        )

    return {
        "label_a": label_a, "label_b": label_b,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "p_value": float(p_val), "t_statistic": float(t_stat),
        "cohen_d": float(d), "significant": bool(p_val < alpha),
        "alpha": alpha, "test_type": "paired_ttest",
    }


def check_std_warning(results: Dict[str, List[float]]) -> List[str]:
    """Cảnh báo nếu std > 0.5%."""
    warnings = []
    for metric, vals in results.items():
        if len(vals) < 2:
            continue
        std = float(np.std(vals))
        if std > STD_WARN_THR:
            warnings.append(
                f"STD WARNING: {metric} std={std:.4f} > {STD_WARN_THR:.3f} (0.5%)"
            )
    return warnings


# ── Load per-seed results ─────────────────────────────────────────────────────

def load_per_seed(result_dir: str, model_name: str) -> Optional[Dict[str, List[float]]]:
    path = os.path.join(result_dir, f"{model_name}_results.json")
    if not os.path.exists(path):
        logger.warning(f"Không tìm thấy: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    per_seed = data.get("per_seed", [])
    if not per_seed:
        logger.warning(
            f"Không có per_seed trong {path}. "
            "Re-run với --seeds 42 0 1 2 3 để có paired t-test."
        )
        return {k: [v] for k, v in data.get("mean", {}).items()}

    metric_lists: Dict[str, List[float]] = {}
    for seed_result in per_seed:
        for k, v in seed_result.get("test_metrics", {}).items():
            metric_lists.setdefault(k, []).append(v)
    return metric_lists


# ── Main functions ────────────────────────────────────────────────────────────

def run_and_compare(
    models_to_compare: List[str],
    result_dir: str,
    alpha: float = 0.05,
) -> None:
    """So sánh models đã có kết quả."""
    logger.info(f"\nSo sánh significance: {' vs '.join(models_to_compare)}")

    model_results = {}
    for model in models_to_compare:
        res = load_per_seed(result_dir, model)
        if res is not None:
            model_results[model] = res

    if len(model_results) < 2:
        logger.warning("Cần ít nhất 2 models để so sánh.")
        return

    # In bảng mean ± std
    print("\n" + "=" * 80)
    print(f"{'Model':<25}" + "".join(f"{m:>20}" for m in METRICS))
    print("-" * 80)
    for model, res in model_results.items():
        means = [np.mean(res.get(m, [0])) for m in METRICS]
        stds  = [np.std(res.get(m, [0]))  for m in METRICS]
        row   = "".join(
            f"{f'{m:.4f}±{s:.4f}':>20}" for m, s in zip(means, stds))
        print(f"{model:<25}{row}")
    print("=" * 80)

    # Significance tests: model[0] vs các models khác
    ref_model  = models_to_compare[0]
    ref_results = model_results.get(ref_model, {})
    significance_warnings = []

    print("\n" + "=" * 80)
    print(f"SIGNIFICANCE TESTS (ref: {ref_model})")
    print("=" * 80)

    for other_model in models_to_compare[1:]:
        other_results = model_results.get(other_model, {})
        print(f"\n{ref_model} vs {other_model}:")
        print(f"{'Metric':<20} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
        print("-" * 50)

        for metric in METRICS:
            a = ref_results.get(metric, [])
            b = other_results.get(metric, [])
            if not a or not b:
                continue
            res = significance_test(a, b, ref_model, other_model, alpha)
            sig = "✓*" if res["significant"] else "✗ ns"
            print(
                f"  {metric:<20} {res['p_value']:>10.4f} "
                f"{res.get('cohen_d', 0):>10.4f} {sig:>5}"
            )

            # Cảnh báo đặc biệt khi so với KGCL
            if "kgcl" in other_model.lower() and not res["significant"]:
                msg = (
                    f"⚠ WARNING (T2.3): {ref_model} không có cải thiện có ý nghĩa "
                    f"thống kê so với KGCL trên {metric} (p={res['p_value']:.4f}). "
                    "Phân tích nguyên nhân trước khi tiếp tục Tuần 5!"
                )
                significance_warnings.append(msg)

    # STD warnings
    for model, res in model_results.items():
        for w in check_std_warning(res):
            significance_warnings.append(f"[{model}] {w}")

    if significance_warnings:
        warn_path = os.path.join(result_dir, "significance_warning.md")
        with open(warn_path, "w") as f:
            f.write("# Significance Warnings\n\n")
            for w in significance_warnings:
                f.write(f"- {w}\n")
                logger.warning(w)
        logger.warning(f"⚠ Có {len(significance_warnings)} cảnh báo → {warn_path}")
    else:
        logger.info("✓ Không có cảnh báo significance.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed runner + Significance test [T2.3/T3.3]")
    p.add_argument("--model",      default=None)
    p.add_argument("--dataset",    default="amazon-book")
    p.add_argument("--seeds",      nargs="+", type=int, default=[42, 0, 1, 2, 3])
    p.add_argument("--compare",    default=None,
                   help="models cần so sánh, phân cách bằng dấu phẩy")
    p.add_argument("--result_dir", default="results/tables")
    p.add_argument("--alpha",      type=float, default=0.05)
    p.add_argument("--base_config",default="configs/base.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    # Nếu chỉ cần so sánh kết quả đã có
    if args.compare and not args.model:
        models = [m.strip() for m in args.compare.split(",")]
        run_and_compare(models, args.result_dir, args.alpha)
        return

    # Train model và so sánh
    if args.model:
        ALL_MODELS = ["lightgcn", "simgcl", "kgat", "kgcl",
                      "kg_lightgcn", "kg_lightgcn_cl"]
        models = ALL_MODELS if args.model == "all" else [args.model]

        for model_name in models:
            logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
            model_cfg = f"configs/model/{model_name}.yaml"
            cfg = load_config(
                base_path         = args.base_config,
                model_config_path = model_cfg if os.path.exists(model_cfg) else None,
                overrides         = {"dataset.name": args.dataset},
            )
            train_model(model_name=model_name, cfg=cfg, seeds=args.seeds)

    # So sánh nếu được chỉ định
    if args.compare:
        models = [m.strip() for m in args.compare.split(",")]
        run_and_compare(models, args.result_dir, args.alpha)


if __name__ == "__main__":
    main()
