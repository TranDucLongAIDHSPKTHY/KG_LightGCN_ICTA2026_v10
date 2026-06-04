"""results/aggregate.py — v10. Aggregate JSON results → CSV + LaTeX."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("aggregate")
ALL_MODELS = ["lightgcn","simgcl","kgat","kgcl","kg_lightgcn","kg_lightgcn_cl"]
METRICS = ["recall@20","ndcg@20","hr@10","ndcg@10"]
MODEL_LABELS = {
    "lightgcn":       "LightGCN",
    "simgcl":         "SimGCL",
    "kgat":           "KGAT",
    "kgcl":           "KGCL",
    "kg_lightgcn":    "KG-LightGCN (Ours)",
    "kg_lightgcn_cl": "KG-LightGCN-CL (Ours)",
}

def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def build_main_table(dataset, result_dir):
    rows = []
    for model in ALL_MODELS:
        path = os.path.join(result_dir, f"{model}_results.json")
        data = load_result(path)
        if data is None:
            continue
        row = {"model": model}
        for metric in METRICS:
            m = data.get("mean", {}).get(metric)
            s = data.get("std",  {}).get(metric)
            row[metric] = (m, s)
        row["n_seeds"] = len(data.get("per_seed", []))
        rows.append(row)
    return rows

def rows_to_csv(rows, metrics):
    header = "model,n_seeds," + ",".join(f"{m}_mean,{m}_std" for m in metrics)
    lines  = [header]
    for row in rows:
        parts = [row["model"], str(row.get("n_seeds", "?"))]
        for m in metrics:
            val = row.get(m)
            if val and val[0] is not None:
                parts.extend([f"{val[0]:.6f}", f"{val[1]:.6f}"])
            else:
                parts.extend(["N/A", "N/A"])
        lines.append(",".join(parts))
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="amazon-book")
    p.add_argument("--result_dir", default="results/tables")
    args = p.parse_args()

    rows = build_main_table(args.dataset, args.result_dir)
    if not rows:
        logger.warning(f"Không tìm thấy result files trong {args.result_dir}.")
        return

    csv_str  = rows_to_csv(rows, METRICS)
    csv_path = os.path.join(args.result_dir, f"main_results_{args.dataset}.csv")
    with open(csv_path, "w") as f:
        f.write(csv_str)
    logger.info(f"CSV → {csv_path}")

    print(f"\n{'='*75}\nMAIN RESULTS — {args.dataset}\n{'='*75}")
    print(f"{'Model':<22}" + "".join(f"{m:>26}" for m in METRICS))
    print("-" * 75)
    for row in rows:
        vals = []
        for m in METRICS:
            v = row.get(m)
            vals.append(f"{v[0]:.4f}±{v[1]:.4f}" if v and v[0] is not None else "N/A")
        print(f"{row['model']:<22}" + "".join(f"{v:>26}" for v in vals))
    print("=" * 75)

if __name__ == "__main__":
    main()
