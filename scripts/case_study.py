"""
scripts/case_study.py — v10 [T4.3].
Phân tích 3 case studies trên Amazon-Book.
Case 1: Warm item đúng
Case 2: Cold item đúng (Cold-20)
Case 3: Failure case (model sai) — phải trung thực
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("case_study")

def write_case_study_template(output_path: str) -> None:
    """Tạo template case_study.md để điền sau khi có kết quả thực."""
    content = """# Case Study — KG-LightGCN-CL v10 [T4.3]

Dataset: Amazon-Book | Model: KG-LightGCN-CL

---

## Case 1: Warm Item Đúng

| user_id | item_id | item_name | supporting_entities | rank |
|---------|---------|-----------|---------------------|------|
| (điền sau khi có model output) | | | | |

**Phân tích:** KG entities nào đã hỗ trợ recommendation đúng?

---

## Case 2: Cold Item Đúng (Cold-20)

| user_id | cold_item_id | item_name | inferred_from | rank |
|---------|-------------|-----------|---------------|------|
| (điền sau khi có model output) | | | | |

**Phân tích:** Model suy luận embedding từ KG entities nào?
(cold item không có trong training — chỉ có KG links)

---

## Case 3: Failure Case (Model Sai)

**Lưu ý:** Failure cases phải được báo cáo TRUNG THỰC — không che giấu.

| user_id | item_id | predicted_rank | true_rank | failure_reason |
|---------|---------|----------------|-----------|----------------|
| (điền sau khi có model output) | | | | |

**Phân tích nguyên nhân:**
- KG noise (orphan entities, low-quality triples)?
- Interaction quá sparse cho user này?
- Positive pair construction không phù hợp?

---

## Chạy script để lấy predictions:

```bash
python main.py --model kg_lightgcn_cl --dataset amazon-book --seeds 42
# Sau đó phân tích checkpoint tại:
# results/checkpoints/amazon-book/kglightgcncl/seed42_best.pt
```
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"✓ Case study template → {output_path}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/case_study.md")
    args = p.parse_args()
    write_case_study_template(args.output)
    logger.info(
        "Template đã tạo. Điền kết quả thực sau khi chạy model và "
        "phân tích predictions.")

if __name__ == "__main__":
    main()
