"""
scripts/sensitivity_analysis.py — v10-fix [T4.2]
Sensitivity analysis cho KG-LightGCN-CL (BLOCKING prerequisite cho Tuần 5).

FIX v10-fix:
  [BUG-5] Xóa def Tuple_str() — là fake type alias gây confusion và dead code.
          Sửa return annotation của write_sensitivity_report thành -> str.

Grid:
  K (CF n_layers)      ∈ {1, 2, 3, 4}
  d (embedding_dim)    ∈ {32, 64, 128, 256}
  λ (lambda_cl)        ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
  kg_n_layers          ∈ {1, 2, 3}
  kg_reg               ∈ {1e-6, 1e-5, 1e-4}

Output bắt buộc:
  results/sensitivity_K.png
  results/sensitivity_d.png
  results/sensitivity_results.md    ← ghi rõ K* và λ*
  results/sensitivity_lambda_range.md
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger

logger = get_script_logger("sensitivity_v10")

GRID: Dict[str, list] = {
    "n_layers":      [1, 2, 3, 4],
    "kg_n_layers":   [1, 2, 3],
    "embedding_dim": [32, 64, 128, 256],
    "lambda_cl":     [0.01, 0.05, 0.1, 0.5, 1.0],
    "kg_reg":        [1e-6, 1e-5, 1e-4],
}

FAST_SEEDS = [42]   # 1 seed cho sensitivity (nhanh); full experiments dùng 5 seeds

# Key mapping từ param name → config key
_KEY_MAP: Dict[str, str] = {
    "n_layers":      "model.n_layers",
    "kg_n_layers":   "model.kg_n_layers",
    "embedding_dim": "model.embedding_dim",
    "lambda_cl":     "contrastive.lambda_cl",
    "kg_reg":        "model.kg_reg",
}


def run_sweep(
    param:      str,
    values:     list,
    dataset:    str,
    result_dir: str,
) -> Dict[str, Dict]:
    logger.info(f"\n{'='*60}\nSensitivity: {param} ∈ {values}\n{'='*60}")
    sweep: Dict[str, Dict] = {}

    for val in values:
        logger.info(f"  {param} = {val}")
        overrides = {
            "dataset.name":    dataset,
            _KEY_MAP[param]:   val,
        }
        cfg = load_config(
            base_path         = "configs/base.yaml",
            model_config_path = "configs/model/kg_lightgcn_cl.yaml",
            overrides         = overrides,
        )
        results          = train_model(
            model_name="kg_lightgcn_cl", cfg=cfg, seeds=FAST_SEEDS)
        sweep[str(val)]  = results.get("mean", {})

    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, f"sensitivity_{param}.json")
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info(f"✓ Saved → {out_path}")

    # Print table
    print(f"\n{'='*55}\nSensitivity: {param}\n{'='*55}")
    print(f"{'Value':<15} {'recall@20':>12} {'ndcg@20':>12} {'hr@10':>12}")
    print("-" * 55)
    for val, metrics in sweep.items():
        r = metrics.get("recall@20", float("nan"))
        n = metrics.get("ndcg@20",   float("nan"))
        h = metrics.get("hr@10",     float("nan"))
        print(f"{str(val):<15} {r:>12.6f} {n:>12.6f} {h:>12.6f}")

    return sweep


def write_sensitivity_report(
    all_sweeps: Dict[str, Dict],
    result_dir: str,
) -> str:                                   # [BUG-5 FIX] return type là str
    """
    Ghi sensitivity_results.md với K* và λ* được xác định.

    Returns:
        str: path đến file output
    """
    lines = [
        "# Sensitivity Analysis — KG-LightGCN-CL v10\n",
        "**LƯU Ý:** Tuần 5 KHÔNG được bắt đầu trước khi có K* và λ* từ file này.\n\n",
        "## Kết quả\n",
    ]

    best_vals: Dict[str, Optional[str]] = {}

    for param, sweep in all_sweeps.items():
        lines.append(f"### {param}\n")
        lines.append("| Value | recall@20 | ndcg@20 | hr@10 |")
        lines.append("|-------|-----------|---------|-------|")
        best_r20 = -1.0
        best_val: Optional[str] = None
        for val, metrics in sweep.items():
            r = metrics.get("recall@20", float("nan"))
            n = metrics.get("ndcg@20",   float("nan"))
            h = metrics.get("hr@10",     float("nan"))
            lines.append(f"| {val} | {r:.6f} | {n:.6f} | {h:.6f} |")
            if r > best_r20:
                best_r20 = r
                best_val = val
        best_vals[param] = best_val
        lines.append(
            f"\n**{param} tốt nhất = {best_val}** (recall@20={best_r20:.6f})\n")

    lines.append("\n## Kết luận\n")
    for param, val in best_vals.items():
        lines.append(f"- **{param} tốt nhất = {val}**")

    if "n_layers" in best_vals:
        lines.append(
            f"\n**K tốt nhất = {best_vals.get('n_layers', '?')}** — "
            "Dùng giá trị này cố định cho tất cả experiments tiếp theo.\n"
        )

    content = "\n".join(lines)
    out_path = os.path.join(result_dir, "sensitivity_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"✓ sensitivity_results.md → {out_path}")

    # Lambda range file
    if "lambda_cl" in all_sweeps:
        lambda_path = os.path.join(result_dir, "sensitivity_lambda_range.md")
        with open(lambda_path, "w") as f:
            f.write("# Lambda (λ) Sensitivity — KG-LightGCN-CL\n\n")
            f.write("Dùng để điền Fairness Sheet (Tuần 5).\n\n")
            f.write("| λ | recall@20 |\n|---|----------|\n")
            for val, metrics in all_sweeps["lambda_cl"].items():
                r = metrics.get("recall@20", float("nan"))
                f.write(f"| {val} | {r:.6f} |\n")
            best_lambda = best_vals.get("lambda_cl", "?")
            f.write(f"\n**λ tốt nhất = {best_lambda}**\n")
        logger.info(f"✓ sensitivity_lambda_range.md → {lambda_path}")

    return out_path          # [BUG-5 FIX] trả về str (path), không phải fake type


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensitivity Analysis v10-fix [T4.2] — BLOCKING cho Tuần 5")
    p.add_argument("--dataset",    default="amazon-book")
    p.add_argument(
        "--param",
        choices=list(GRID) + ["all"],
        default="all",
    )
    p.add_argument("--result_dir", default="results/tables")
    p.add_argument("--fast",       action="store_true",
                   help="Chạy 1 seed thay vì 3 seeds (default với FAST_SEEDS)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    params = list(GRID) if args.param == "all" else [args.param]

    all_sweeps: Dict[str, Dict] = {}
    for param in params:
        sweep = run_sweep(param, GRID[param], args.dataset, args.result_dir)
        all_sweeps[param] = sweep

    if all_sweeps:
        write_sensitivity_report(all_sweeps, args.result_dir)

    logger.info(
        "\n✓ Sensitivity analysis hoàn tất.\n"
        "  Xem sensitivity_results.md để lấy K* và λ* trước Tuần 5."
    )


if __name__ == "__main__":
    main()