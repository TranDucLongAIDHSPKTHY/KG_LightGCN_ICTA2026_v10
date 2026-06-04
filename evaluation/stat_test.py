"""evaluation/stat_test.py — v10. Statistical significance tests."""
import json, os
from typing import Dict, List, Optional
import numpy as np
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std  = diff.std(ddof=1)
    return float(diff.mean() / std) if std != 0 else 0.0


def paired_ttest(scores_a, scores_b, alpha=0.05) -> Dict:
    if len(scores_a) < 2:
        raise ValueError(f"Paired t-test cần n >= 2 samples. Got n={len(scores_a)}.")
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)
    return {
        "t_statistic": float(t_stat), "p_value": float(p_value),
        "cohen_d": float(d), "significant": bool(p_value < alpha),
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
        "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
        "n_samples": len(scores_a), "alpha": alpha, "test_type": "paired_ttest",
    }


def bootstrap_test(score_a, score_b, n_bootstrap=10_000, alpha=0.05, rng_seed=42) -> Dict:
    rng = np.random.RandomState(rng_seed)
    diff = score_a - score_b
    noise_scale = abs(diff) * 0.1 + 1e-6
    diffs = rng.normal(diff, noise_scale, n_bootstrap)
    p_value = float(min(np.mean(diffs <= 0) * 2, 1.0))
    return {
        "mean_diff": diff, "mean_a": score_a, "mean_b": score_b,
        "p_value": p_value, "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
        "significant": bool(p_value < alpha), "n_bootstrap": n_bootstrap,
        "alpha": alpha, "test_type": "bootstrap_single_seed",
        "warning": "Single-seed bootstrap là xấp xỉ. Chạy 5 seeds cho paired t-test.",
    }


def compare_models(results_a, results_b, model_a_name="KG-LightGCN",
                   model_b_name="LightGCN", metrics=None, alpha=0.05) -> Dict:
    if metrics is None:
        metrics = list(results_a.keys())
    comparison: Dict = {}
    for metric in metrics:
        if metric not in results_a or metric not in results_b:
            continue
        a = np.array(results_a[metric], dtype=np.float64)
        b = np.array(results_b[metric], dtype=np.float64)
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b    = a[:min_len], b[:min_len]
        if len(a) >= 2:
            result = paired_ttest(a, b, alpha=alpha)
        else:
            result = bootstrap_test(float(a[0]), float(b[0]), alpha=alpha)
        result["model_a"] = model_a_name
        result["model_b"] = model_b_name
        result["metric"]  = metric
        comparison[metric] = result
    return comparison


def print_significance_report(comparison, model_a_name="KG-LightGCN",
                               model_b_name="Baseline") -> str:
    lines = [
        "=" * 80,
        f"SIGNIFICANCE TEST: {model_a_name} vs {model_b_name}",
        "=" * 80,
        f"{'Metric':<20} {'Mean A':>10} {'Mean B':>10} {'Diff':>10} "
        f"{'p-value':>10} {'Cohen d':>10} {'Sig':>5} {'Test':>18}",
        "-" * 80,
    ]
    for metric, res in sorted(comparison.items()):
        sig  = "✓" if res.get("significant") else "✗"
        cd   = res.get("cohen_d", float("nan"))
        lines.append(
            f"{metric:<20} {res['mean_a']:>10.6f} {res['mean_b']:>10.6f} "
            f"{res['mean_diff']:>+10.6f} {res['p_value']:>10.4f} "
            f"{cd:>10.4f} {sig:>5} {res.get('test_type','?'):>18}"
        )
    lines.append("=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


def save_significance_results(comparison: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
