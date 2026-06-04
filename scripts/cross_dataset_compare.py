"""scripts/cross_dataset_compare.py — v10 [T4.1]. Cross-dataset comparison."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("cross_dataset_compare")
MODELS = ["lightgcn","simgcl","kgat","kgcl","kg_lightgcn","kg_lightgcn_cl"]
METRICS = ["recall@20","ndcg@20","hr@10","ndcg@10"]

def load_mean(result_dir, model):
    path = os.path.join(result_dir, f"{model}_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("mean", {})

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir_ab",  default="results/tables/amazon-book")
    p.add_argument("--result_dir_yelp",default="results/tables/yelp2018")
    p.add_argument("--output",         default="results/cross_dataset_comparison.md")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    lines = [
        "# Cross-Dataset Comparison — v10\n\n",
        "| Model | AB Recall@20 | Yelp Recall@20 | Δ% vs LightGCN (AB) | Δ% vs LightGCN (Yelp) |\n",
        "|-------|-------------|----------------|---------------------|----------------------|\n",
    ]

    ab_base   = load_mean(args.result_dir_ab,  "lightgcn").get("recall@20", 0)
    yelp_base = load_mean(args.result_dir_yelp, "lightgcn").get("recall@20", 0)

    for model in MODELS:
        ab_m   = load_mean(args.result_dir_ab,   model)
        yelp_m = load_mean(args.result_dir_yelp,  model)
        ab_r   = ab_m.get("recall@20")
        yelp_r = yelp_m.get("recall@20")
        delta_ab   = (ab_r   - ab_base)   / max(ab_base, 1e-9)   * 100 if ab_r   else None
        delta_yelp = (yelp_r - yelp_base) / max(yelp_base, 1e-9) * 100 if yelp_r else None
        lines.append(
            f"| {model} | {ab_r or 'N/A'} | {yelp_r or 'N/A'} | "
            f"{f'{delta_ab:+.1f}%' if delta_ab is not None else 'N/A'} | "
            f"{f'{delta_yelp:+.1f}%' if delta_yelp is not None else 'N/A'} |\n"
        )

    lines.append(
        "\n> **Ghi chú:** Kết quả này chỉ valid cho CF pipeline.\n"
        "> Yelp2018 có flat POI category (ít phong phú hơn Amazon-Book KG).\n"
    )

    with open(args.output, "w") as f:
        f.writelines(lines)
    logger.info(f"✓ cross_dataset_comparison.md → {args.output}")

if __name__ == "__main__":
    main()
