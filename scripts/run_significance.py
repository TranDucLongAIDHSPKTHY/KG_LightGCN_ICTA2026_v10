"""scripts/run_significance.py — v10. Significance tests giữa các model pairs."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.stat_test import (compare_models, print_significance_report,
                                   save_significance_results)
from utils.logger import get_script_logger

logger = get_script_logger("run_significance")
METRICS = ["recall@20", "ndcg@20", "hr@10", "ndcg@10"]

PAIRS = [
    ("kg_lightgcn",    "lightgcn",    "KG-LightGCN vs LightGCN"),
    ("kg_lightgcn_cl", "lightgcn",    "KG-LightGCN-CL vs LightGCN"),
    ("kg_lightgcn_cl", "simgcl",      "KG-LightGCN-CL vs SimGCL"),
    ("kg_lightgcn_cl", "kgcl",        "KG-LightGCN-CL vs KGCL"),
    ("kg_lightgcn_cl", "kgat",        "KG-LightGCN-CL vs KGAT"),
    ("kg_lightgcn_cl", "kg_lightgcn", "KG-LightGCN-CL vs KG-LightGCN (ablation)"),
]

def load_per_seed(result_path):
    if not os.path.exists(result_path):
        return {}
    with open(result_path) as f:
        data = json.load(f)
    per_seed = data.get("per_seed", [])
    if not per_seed:
        logger.warning(f"Không có per_seed trong {result_path}. Dùng bootstrap.")
        return {k: [v] for k, v in data.get("mean", {}).items()}
    metric_lists = {}
    for seed_result in per_seed:
        for k, v in seed_result.get("test_metrics", {}).items():
            metric_lists.setdefault(k, []).append(v)
    return metric_lists

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="amazon-book")
    p.add_argument("--result_dir", default="results/tables")
    p.add_argument("--alpha",      type=float, default=0.05)
    return p.parse_args()

def main():
    args = parse_args()
    for model_a, model_b, description in PAIRS:
        path_a = os.path.join(args.result_dir, f"{model_a}_results.json")
        path_b = os.path.join(args.result_dir, f"{model_b}_results.json")
        res_a  = load_per_seed(path_a)
        res_b  = load_per_seed(path_b)
        if not res_a or not res_b:
            continue
        available = [m for m in METRICS if m in res_a and m in res_b]
        if not available:
            continue
        logger.info(f"\n{'='*70}\n{description}\n{'='*70}")
        comparison = compare_models(
            results_a=res_a, results_b=res_b,
            model_a_name=model_a, model_b_name=model_b,
            metrics=available, alpha=args.alpha,
        )
        print_significance_report(comparison, model_a, model_b)
        out_path = os.path.join(
            args.result_dir,
            f"significance_{model_a}_vs_{model_b}.json",
        )
        save_significance_results(comparison, out_path)

if __name__ == "__main__":
    main()
