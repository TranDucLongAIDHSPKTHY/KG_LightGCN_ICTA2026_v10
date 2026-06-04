# CHANGELOG — KG-LightGCN ICTA2026

## v10 (Hiện tại) — Theo Kịch bản 3 / T2.1 Data Unification Runbook

### Thay đổi lớn (Breaking Changes)

| ID | File | Thay đổi | Lý do |
|----|------|----------|-------|
| V10-DS-1 | `scripts/download_data.py` | VIẾT LẠI hoàn toàn — chỉ dùng KGAT repo | T2.1: nguồn DUY NHẤT là KGAT repo |
| V10-DS-2 | `scripts/preprocess.py` | VIẾT LẠI hoàn toàn — gộp KGAT train+test | T2.1: 80/10/10 user-wise, seed=42 |
| V10-DS-3 | `scripts/build_cold_split.py` | `cold_items.txt` → `cold_item_ids.txt` | T3.1: naming convention v10 |
| V10-DS-4 | `scripts/build_cold_split.py` | `test.txt` (cold) → `test_cold.txt` | T3.1: tránh nhầm với warm test |
| V10-DS-5 | Toàn bộ codebase | `val.txt` → `valid.txt` | T2.1: naming convention |
| V10-DS-6 | Toàn bộ codebase | `kg_full.txt` → `kg_final.txt` | KGAT repo format |
| V10-DS-7 | Toàn bộ codebase | Paths → `/data/phuongtran/project_v10/` | T2.1: canonical paths |

### Thay đổi bất biến (Invariants)

| ID | File | Thay đổi | Lý do |
|----|------|----------|-------|
| V10-INV-1 | `datasets/kg_dataset.py` | Thêm `_assert_kgat_invariant()` | Verify `item_id == entity_id` |
| V10-INV-2 | `models/kg_lightgcn.py` | `_get_entity_for_items()` dùng direct indexing | item_id == entity_id, không cần lookup |
| V10-INV-3 | `main.py` | `set_item_entity_map(None)` — no-op | KGAT convention: không cần mapping |
| V10-INV-4 | `scripts/preprocess.py` | Không re-index item IDs | Giữ KGAT original IDs |

### Files mới (v10)

| File | Mô tả | Yêu cầu |
|------|-------|---------|
| `scripts/diagnose_result_conflict.py` | Phân tích mâu thuẫn KG < LightGCN | T1.4 |
| `scripts/run_multiseed.py` | Multi-seed runner + paired t-test + Cohen's d | T2.3 |
| `scripts/log_experiment_setup.py` | Tự động sinh experimental_setup.md | T3.4 |
| `scripts/sensitivity_analysis.py` | Sensitivity K/d/λ (BLOCKING Tuần 5) | T4.2 |
| `scripts/case_study.py` | Template case study 3 loại | T4.3 |
| `scripts/cross_dataset_compare.py` | So sánh Amazon-Book vs Yelp2018 | T4.1 |
| `scripts/run_all.py` | Full pipeline từ download → results | T2.1 |
| `scripts/run_baselines.py` | Chạy 4 baselines bắt buộc | T2.2 |
| `scripts/run_ablation.py` | Entity ablation A1/A2/A3/A4 | T3.2 |
| `scripts/run_side_info_comparison.py` | Settings A/B/C comparison | T5.1-T5.3 |
| `scripts/run_cold_start.py` | Cold-start evaluation | T3.1 |

### Thay đổi config

| File | Thay đổi |
|------|----------|
| `configs/base.yaml` | Thêm `eval_protocol: full`, `valid_file: valid.txt`, `kg_file: kg_final.txt` |
| `configs/base.yaml` | Paths → `/data/phuongtran/project_v10/unified/` |
| `configs/base.yaml` | Thêm `data_source: kgat_repo` |

### Thay đổi dataset & evaluation

| File | Thay đổi |
|------|----------|
| `datasets/base_dataset.py` | Split "valid" thay "val" |
| `datasets/cf_dataset.py` | `valid.txt` thay `val.txt` |
| `datasets/kg_dataset.py` | KGAT format, kg_final.txt, item_id==entity_id |
| `evaluation/evaluator.py` | `valid_user2items` thay `val_user2items` |
| `evaluation/cold_evaluator.py` | `cold_item_ids.txt` + `test_cold.txt` |
| `trainers/trainer.py` | evaluate(split="valid") thay "val" |

### Files không thay đổi về architecture

Các file sau giữ nguyên architecture từ v7 (chỉ cập nhật comment/docstring):
- `models/base_model.py`, `models/lightgcn.py`, `models/simgcl.py`
- `models/kgat.py`, `models/kgcl.py`, `models/kg_lightgcn.py`
- `trainers/kg_trainer.py`, `losses/bpr_loss.py`, `losses/contrastive_loss.py`
- `evaluation/metrics.py`, `evaluation/full_ranking.py`, `evaluation/stat_test.py`
- `utils/seed.py`, `utils/logger.py`, `utils/config.py`

---

## v7 (Trước đó)

- Data source: LightGCN repo (Amazon-Book CF) + KGCL repo (Yelp2018 KG)
- val.txt (không phải valid.txt)
- cold_items.txt (không phải cold_item_ids.txt)
- test.txt cho cold evaluation (không phải test_cold.txt)
- kg_full.txt (không phải kg_final.txt)
- Paths: `/data/phuongtran/processed/`
- Build KG từ meta_Books.json.gz (không dùng KGAT kg_final.txt trực tiếp)
- Có thể re-index item IDs (vi phạm KGAT invariant)

---

## Ghi chú về tương thích

- **v10 KHÔNG tương thích ngược với v7** về tên file và đường dẫn
- Dữ liệu cũ tại `/data/phuongtran/processed/` vẫn có thể dùng nhưng cần:
  - Đổi tên `val.txt` → `valid.txt`
  - Đổi tên `cold_items.txt` → `cold_item_ids.txt`
  - Đổi tên `test.txt` (cold) → `test_cold.txt`
  - Verify KGAT invariant: `assert max(item_ids) < n_items`
- **Khuyến nghị**: Chạy lại `scripts/preprocess.py` từ KGAT raw data để có v10 format chuẩn
