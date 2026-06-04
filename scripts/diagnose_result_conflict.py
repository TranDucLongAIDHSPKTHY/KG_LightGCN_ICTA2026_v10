"""
scripts/diagnose_result_conflict.py — v10 [T1.4]
═══════════════════════════════════════════════════════════════════════════════
Script phát hiện và log mâu thuẫn kết quả (T1.4).

Kiểm tra: Recall@20 Full Ranking của KG-LightGCN < LightGCN?
Phân tích 3 nguyên nhân:
  (a) Bug trong evaluation (wrong candidate pool, wrong metric formula)
  (b) KG entities gây noise (orphan, low-frequency entities)
  (c) Hyperparameter chưa tune (lr, weight_decay, K)

Output: results/conflict_analysis.md (honest reporting bắt buộc)

Usage:
  python scripts/diagnose_result_conflict.py --dataset amazon-book
  python scripts/diagnose_result_conflict.py --result_dir results/tables
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("diagnose_conflict")

METRICS = ["recall@20", "ndcg@20", "hr@10", "ndcg@10"]


def load_results(result_dir: str, model_name: str) -> Optional[Dict]:
    """Load kết quả từ JSON file."""
    path = os.path.join(result_dir, f"{model_name}_results.json")
    if not os.path.exists(path):
        logger.warning(f"Không tìm thấy: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def check_metric_conflict(
    res_kg:  Dict, res_base: Dict, metric: str = "recall@20"
) -> Optional[bool]:
    """Kiểm tra KG model có thấp hơn baseline không."""
    kg_val   = res_kg.get("mean",  {}).get(metric)
    base_val = res_base.get("mean", {}).get(metric)
    if kg_val is None or base_val is None:
        return None
    return kg_val < base_val


def analyze_cause_a_evaluation(result_dir: str) -> str:
    """(a) Kiểm tra bug trong evaluation."""
    lines = [
        "### Nguyên nhân (a): Bug trong Evaluation\n",
        "**Checklist:**\n",
        "- [ ] Candidate pool: phải là TẤT CẢ items chưa tương tác (full ranking)",
        "- [ ] Metric formula: Recall@K = hits / min(|GT|, K)",
        "- [ ] Training mask: đã mask đúng training items trong evaluation chưa?",
        "- [ ] n_items consistency: KG model và CF model dùng cùng n_items không?",
        "- [ ] eval_protocol: 'full' (không phải 'sampled')\n",
        "**Kết quả kiểm tra:**",
    ]

    # Kiểm tra config
    config_path = "configs/base.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            content = f.read()
        if "eval_protocol: full" in content:
            lines.append("  ✓ eval_protocol: full được cài đặt đúng")
        else:
            lines.append(
                "  ✗ CẢNH BÁO: eval_protocol không rõ ràng — kiểm tra lại configs/base.yaml")
    else:
        lines.append("  ⚠ configs/base.yaml không tìm thấy")

    lines.append(
        "\n**Kết luận:** Nếu eval_protocol đúng và candidate pool = full items, "
        "lỗi evaluation khó xảy ra. Chuyển sang kiểm tra nguyên nhân (b) và (c).\n"
    )
    return "\n".join(lines)


def analyze_cause_b_kg_noise(data_dir: str, dataset: str) -> str:
    """(b) KG entities gây noise — phân tích entity frequency."""
    lines = [
        "### Nguyên nhân (b): KG Entities Gây Noise\n",
        "**Hypothesis:** Orphan entities hoặc low-frequency entities làm giảm "
        "chất lượng embeddings.\n",
        "**Entity frequency analysis:**",
    ]

    kg_path = os.path.join(data_dir, dataset, "kg_final.txt")
    if not os.path.exists(kg_path):
        lines.append(f"  ⚠ Không tìm thấy {kg_path} — bỏ qua phân tích")
        return "\n".join(lines)

    # Đếm frequency của entities
    entity_freq: Dict[int, int] = {}
    n_triples = 0
    with open(kg_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) == 3:
                try:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    entity_freq[h] = entity_freq.get(h, 0) + 1
                    entity_freq[t] = entity_freq.get(t, 0) + 1
                    n_triples += 1
                except ValueError:
                    pass

    if not entity_freq:
        lines.append("  ⚠ KG file rỗng")
        return "\n".join(lines)

    total_entities = len(entity_freq)
    thresholds     = [1, 2, 5, 10, 20]
    lines.append(f"\n  Total triples: {n_triples:,}")
    lines.append(f"  Total entities: {total_entities:,}\n")
    lines.append("  | Threshold freq | Entities bị lọc | % bị lọc |")
    lines.append("  |----------------|-----------------|----------|")

    for th in thresholds:
        filtered = sum(1 for f in entity_freq.values() if f < th)
        pct      = filtered / total_entities * 100
        lines.append(f"  | < {th:<13} | {filtered:<15,} | {pct:>7.1f}% |")

    lines.append(
        "\n**Khuyến nghị:** Thử filter entities với freq < 5 hoặc < 10 "
        "trong kg_dataset.py để giảm noise.\n"
    )
    lines.append(
        "**Kết luận:** Nếu KG có nhiều orphan entities (freq=1), "
        "đây có thể là nguyên nhân chính làm KG model tệ hơn baseline.\n"
    )
    return "\n".join(lines)


def analyze_cause_c_hyperparams() -> str:
    """(c) Hyperparameter analysis."""
    lines = [
        "### Nguyên nhân (c): Hyperparameter Chưa Tune\n",
        "**Checklist:**\n",
        "| Hyperparameter | KG-LightGCN | LightGCN | Ghi chú |",
        "|----------------|-------------|----------|---------|",
        "| lr             | 0.001       | 0.001    | Fairness: phải bằng nhau |",
        "| weight_decay   | 1e-4        | 1e-4     | Fairness: phải bằng nhau |",
        "| embedding_dim  | 64          | 64       | HARD constraint |",
        "| n_layers (CF)  | 3           | 3        | Từ sensitivity analysis |",
        "| kg_n_layers    | 2           | N/A      | Chạy sensitivity? |",
        "| kg_reg (λ_kg)  | 1e-5        | N/A      | Chạy sensitivity? |\n",
        "**Nguyên nhân tiềm năng:**",
        "1. kg_n_layers quá cao → over-smoothing trong KG propagation",
        "2. kg_reg quá cao → KG alignment loss dominates BPR loss",
        "3. alpha (blend ratio) chưa học được giá trị tối ưu\n",
        "**Khuyến nghị:** Chạy sensitivity analysis trước khi kết luận:\n",
        "```bash",
        "python scripts/sensitivity_analysis.py --dataset amazon-book --param kg_n_layers",
        "python scripts/sensitivity_analysis.py --dataset amazon-book --param kg_reg",
        "```\n",
        "**Kết luận:** Nếu kg_n_layers=1 cho kết quả tốt hơn kg_n_layers=2, "
        "đây là nguyên nhân.\n",
    ]
    return "\n".join(lines)


def write_conflict_analysis(
    result_dir: str, data_dir: str, dataset: str,
    res_kg: Optional[Dict], res_base: Optional[Dict],
) -> str:
    """Viết conflict_analysis.md."""
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, "conflict_analysis.md")

    lines = [
        "# Conflict Analysis: KG-LightGCN vs LightGCN\n",
        f"**Dataset:** {dataset}  \n",
        f"**Script:** scripts/diagnose_result_conflict.py  \n",
        f"**Mục đích:** Phân tích trung thực mâu thuẫn kết quả (T1.4)\n",
        "---\n",
        "## 1. Phát hiện mâu thuẫn\n",
    ]

    if res_kg is None or res_base is None:
        lines.append(
            "⚠ Chưa có kết quả để so sánh. Chạy models trước:\n"
            "```bash\n"
            "python main.py --model lightgcn --dataset amazon-book --seeds 42\n"
            "python main.py --model kg_lightgcn --dataset amazon-book --seeds 42\n"
            "```\n"
        )
    else:
        lines.append("| Metric | KG-LightGCN | LightGCN | Mâu thuẫn? |")
        lines.append("|--------|-------------|----------|------------|")
        has_conflict = False
        for metric in METRICS:
            kg_val   = res_kg.get("mean", {}).get(metric, "N/A")
            base_val = res_base.get("mean", {}).get(metric, "N/A")
            if isinstance(kg_val, float) and isinstance(base_val, float):
                conflict = "✗ CÓ" if kg_val < base_val else "✓ Không"
                if kg_val < base_val:
                    has_conflict = True
            else:
                conflict = "?"
            lines.append(f"| {metric} | {kg_val} | {base_val} | {conflict} |")

        if has_conflict:
            lines.append(
                "\n⚠ **PHÁT HIỆN MÂU THUẪN:** KG-LightGCN thấp hơn LightGCN "
                "trên một số metrics!\n"
            )
            lines.append(
                "**Theo yêu cầu T1.4:** Mâu thuẫn này phải được phân tích và "
                "log rõ nguyên nhân — KHÔNG được bỏ qua.\n"
            )
        else:
            lines.append(
                "\n✓ Không phát hiện mâu thuẫn — KG-LightGCN >= LightGCN "
                "trên tất cả metrics.\n"
            )

    lines.append("\n---\n")
    lines.append("## 2. Phân tích nguyên nhân\n")
    lines.append(analyze_cause_a_evaluation(result_dir))
    lines.append("\n---\n")
    lines.append(analyze_cause_b_kg_noise(data_dir, dataset))
    lines.append("\n---\n")
    lines.append(analyze_cause_c_hyperparams())
    lines.append("\n---\n")
    lines.append("## 3. Kết luận và Hành động tiếp theo\n")
    lines.append(
        "Sau khi phân tích 3 nguyên nhân trên, xác định nguyên nhân chính "
        "và thực hiện:\n\n"
        "1. **Nếu (a):** Fix code evaluation → verify lại\n"
        "2. **Nếu (b):** Thêm entity frequency filtering → re-run\n"
        "3. **Nếu (c):** Chạy sensitivity analysis → tune hyperparameters\n\n"
        "> *Ghi chú: Nếu sau khi tune vẫn thấp hơn LightGCN, cần honest reporting "
        "trong paper — không được che giấu kết quả.*\n"
    )

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"✓ conflict_analysis.md đã được tạo → {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Phân tích mâu thuẫn kết quả KG-LightGCN vs LightGCN [T1.4]")
    p.add_argument("--dataset",    default="amazon-book")
    p.add_argument("--result_dir", default="results/tables")
    p.add_argument(
        "--data_dir",
        default="/data/phuongtran/project_v10/unified",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 65)
    logger.info("DIAGNOSE RESULT CONFLICT [T1.4]")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("=" * 65)

    res_kg   = load_results(args.result_dir, "kg_lightgcn")
    res_base = load_results(args.result_dir, "lightgcn")

    if res_kg is not None and res_base is not None:
        logger.info("\n=== So sánh kết quả ===")
        for metric in METRICS:
            kg_v   = res_kg.get("mean", {}).get(metric, "N/A")
            base_v = res_base.get("mean", {}).get(metric, "N/A")
            if isinstance(kg_v, float) and isinstance(base_v, float):
                diff   = kg_v - base_v
                status = "✓" if diff >= 0 else "✗ MÂU THUẪN"
                logger.info(
                    f"  {metric}: KG={kg_v:.6f} vs Base={base_v:.6f} "
                    f"(Δ={diff:+.6f}) {status}"
                )
            else:
                logger.info(f"  {metric}: N/A")

    out_path = write_conflict_analysis(
        result_dir=args.result_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        res_kg=res_kg,
        res_base=res_base,
    )
    logger.info(f"\nXem báo cáo: {out_path}")


if __name__ == "__main__":
    main()
