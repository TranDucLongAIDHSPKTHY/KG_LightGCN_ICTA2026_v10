# KG-LightGCN ICTA2026 — v10

> **Research Question:** Trong Collaborative Filtering, khi dùng cùng LightGCN backbone và InfoNCE loss, dạng structured side information nào tạo ra contrastive signal hiệu quả hơn: flat category, brand/publisher, hay full knowledge graph?

> ⚠ **v10 BREAKING CHANGES:** Nguồn dữ liệu, đường dẫn và tên file thay đổi so với v7. Xem [CHANGELOG.md](CHANGELOG.md).

---

## Thay đổi chính v10 so với v7

| Thay đổi | v7 | v10 |
|----------|-----|-----|
| Nguồn dữ liệu | LightGCN + KGCL repos | **KGAT repo DUY NHẤT** |
| KG file | kg_full.txt (build lại) | **kg_final.txt (KGAT gốc)** |
| Split files | val.txt | **valid.txt** |
| Cold split | cold_items.txt, test.txt | **cold_item_ids.txt, test_cold.txt** |
| Paths | /data/phuongtran/processed/ | **/data/phuongtran/project_v10/** |
| Item-Entity | item2entity.json | **item_id == entity_id (KGAT invariant)** |
| KG build | build_kg_from_meta() | **Copy kg_final.txt trực tiếp** |
| Cold evaluator | cold_items.txt | **cold_item_ids.txt + test_cold.txt** |

---

## Cấu trúc Project

```
KG_LightGCN_ICTA2026_v10/
│
├── main.py                        ← Unified entry point
├── requirements.txt
├── CHANGELOG.md                   ← Log thay đổi v7→v10
├── .env.example
│
├── configs/
│   ├── base.yaml                  ← eval_protocol: full, valid.txt, KGAT paths
│   └── model/
│       ├── lightgcn.yaml
│       ├── simgcl.yaml
│       ├── kgat.yaml
│       ├── kgcl.yaml
│       ├── kg_lightgcn.yaml
│       └── kg_lightgcn_cl.yaml
│
├── models/
│   ├── base_model.py
│   ├── lightgcn.py
│   ├── simgcl.py
│   ├── kgat.py                    ← item_id == entity_id (KGAT invariant)
│   ├── kgcl.py
│   └── kg_lightgcn.py             ← Direct indexing (không cần item2entity)
│
├── trainers/
│   ├── trainer.py                 ← split "valid" thay "val"
│   └── kg_trainer.py
│
├── datasets/
│   ├── base_dataset.py            ← split: "train"|"valid"|"test"
│   ├── cf_dataset.py              ← valid.txt
│   ├── kg_dataset.py              ← kg_final.txt, assert item_id==entity_id
│   └── dataloader.py
│
├── losses/
│   ├── bpr_loss.py
│   └── contrastive_loss.py
│
├── evaluation/
│   ├── metrics.py
│   ├── full_ranking.py
│   ├── evaluator.py               ← valid_user2items
│   ├── cold_evaluator.py          ← cold_item_ids.txt + test_cold.txt
│   └── stat_test.py
│
├── utils/
│   ├── config.py                  ← RAW_DIR, DATA_DIR env vars
│   ├── logger.py
│   └── seed.py
│
└── scripts/
    ├── download_data.py           ← KGAT repo DUY NHẤT (xiangwang1223)
    ├── preprocess.py              ← Gộp KGAT train+test → 80/10/10
    ├── build_cold_split.py        ← test_cold.txt, cold_item_ids.txt
    ├── diagnose_result_conflict.py ← [T1.4] Phân tích mâu thuẫn kết quả
    ├── run_multiseed.py           ← [T2.3] paired t-test + Cohen's d
    ├── log_experiment_setup.py    ← [T3.4] Tự động sinh experimental_setup.md
    ├── sensitivity_analysis.py    ← [T4.2] BLOCKING cho Tuần 5
    ├── case_study.py              ← [T4.3] 3 case studies
    ├── cross_dataset_compare.py   ← [T4.1] AB vs Yelp2018
    ├── run_all.py                 ← Full pipeline từ đầu
    ├── run_baselines.py           ← 4 baselines bắt buộc
    ├── run_ablation.py            ← A1/A2/A3/A4 entity ablation
    ├── run_ablation.sh
    ├── run_side_info_comparison.py ← [T5.1-T5.3] Settings A/B/C
    ├── run_cold_start.py          ← Cold-10/20/30 evaluation
    ├── run_significance.py
    └── run_train.py
```

---

## Nguồn dữ liệu

### ⚠ QUAN TRỌNG: Chỉ dùng KGAT repo

```
Repository: KGAT
URL: https://github.com/xiangwang1223/knowledge_graph_attention_network

Files cần thiết:
  Data/amazon-book/
    ├── train.txt
    ├── test.txt
    ├── item_list.txt
    ├── entity_list.txt
    ├── relation_list.txt
    ├── user_list.txt
    └── kg_final.txt.zip  ← giải nén tự động

  Data/yelp2018/
    ├── train.txt
    ├── test.txt
    ├── item_list.txt
    ├── entity_list.txt
    ├── relation_list.txt
    ├── user_list.txt
    └── kg_final.txt.zip  ← giải nén tự động
```

**KHÔNG dùng:**
- ~~LightGCN-PyTorch repo (gusye1234)~~ — v7 cũ
- ~~SimGCL/QRec repo~~ — v7 cũ
- ~~KGCL-SIGIR22 repo~~ — v7 cũ

### Bất biến bắt buộc

```python
# KGAT convention: item_id == entity_id
# n_items thực thể đầu tiên trong KG là item entities
assert max(all_item_ids) < n_items  # KHÔNG được vi phạm!
```

---

## Cài đặt

```bash
# 1. Clone
git clone <repo_url>
cd KG_LightGCN_ICTA2026_v10

# 2. Cài PyTorch (chọn CUDA version)
# GPU CUDA 11.8:
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình paths
cp .env.example .env
# Chỉnh sửa .env nếu cần
```

---

## Pipeline hoàn chỉnh

### Bước 1 — Download dữ liệu (KGAT repo)

```bash
python scripts/download_data.py --dataset all \
    --raw_dir /data/phuongtran/project_v10/raw

# Kiểm tra files
python scripts/download_data.py --dataset all --check_only
```

> Nếu auto-download thất bại:
> ```bash
> git clone https://github.com/xiangwang1223/knowledge_graph_attention_network
> cp -r knowledge_graph_attention_network/Data/amazon-book /data/phuongtran/project_v10/raw/
> cp -r knowledge_graph_attention_network/Data/yelp2018 /data/phuongtran/project_v10/raw/
> ```

### Bước 2 — Preprocessing (Chạy MỘT LẦN — đóng băng)

```bash
python scripts/preprocess.py \
    --dataset all \
    --raw_dir /data/phuongtran/project_v10/raw \
    --out_dir /data/phuongtran/project_v10/unified
```

**Output (v10 format):**
```
/data/phuongtran/project_v10/unified/amazon-book/
  ├── train.txt, valid.txt, test.txt  ← 80/10/10 split (gộp KGAT train+test)
  ├── kg_final.txt                    ← Copy từ KGAT (KHÔNG build lại)
  ├── item_list.txt, entity_list.txt  ← Copy từ KGAT
  ├── item_category.txt               ← Phái sinh (Setting A)
  ├── dataset_stats.md                ← Thống kê (đọc từ file, không dùng số nhớ)
  └── split_checksum.md5             ← MD5 đóng băng
```

### Bước 3 — Cold-start splits

```bash
python scripts/build_cold_split.py \
    --dataset all \
    --data_dir /data/phuongtran/project_v10/unified \
    --ratio 10 20 30 --seed 42
```

**Output (v10 format):**
```
cold_20/
  ├── train.txt, valid.txt  ← cold_items đã xóa
  ├── test_cold.txt         ← CHỈ cold interactions  [v10: không phải test.txt]
  └── cold_item_ids.txt     ← danh sách cold IDs     [v10: không phải cold_items.txt]
```

### Bước 4 — Chạy tất cả baselines (5 seeds)

```bash
python scripts/run_baselines.py \
    --dataset amazon-book --seeds 42 0 1 2 3
```

### Bước 5 — Sensitivity Analysis (BLOCKING cho Tuần 5)

```bash
python scripts/sensitivity_analysis.py \
    --dataset amazon-book --param all
# Xem results/sensitivity_results.md để lấy K* và λ*
```

### Bước 6 — Chạy proposed models (5 seeds)

```bash
python main.py --model kg_lightgcn     --dataset amazon-book --seeds 42 0 1 2 3
python main.py --model kg_lightgcn_cl  --dataset amazon-book --seeds 42 0 1 2 3
```

### Bước 7 — Entity Ablation A1/A2/A3/A4

```bash
python scripts/run_ablation.py \
    --dataset amazon-book --kg_types none,category,brand,full \
    --seeds 42 0 1 2 3
# hoặc:
bash scripts/run_ablation.sh amazon-book
```

### Bước 8 — Yelp2018

```bash
python main.py --model all --dataset yelp2018 --seeds 42 0 1 2 3
```

### Bước 9 — Settings A/B/C (Tuần 5 — cần Fairness Sheet)

```bash
# CHỈ chạy sau khi giáo viên đã duyệt Fairness Sheet
python scripts/run_side_info_comparison.py \
    --settings A,B,C --dataset amazon-book --seeds 42 0 1 2 3 \
    --fairness_approved
```

### Bước 10 — Cold-start Evaluation

```bash
python scripts/run_cold_start.py \
    --dataset amazon-book \
    --levels cold_10 cold_20 cold_30 \
    --models lightgcn kg_lightgcn_cl
```

### Bước 11 — Significance Tests

```bash
python scripts/run_multiseed.py \
    --compare kg_lightgcn_cl,lightgcn,kgcl \
    --result_dir results/tables
# hoặc:
python scripts/run_significance.py --dataset amazon-book
```

### Bước 12 — Aggregate kết quả

```bash
python results/aggregate.py --dataset amazon-book
python results/aggregate.py --dataset yelp2018
```

### Chạy toàn bộ pipeline (tự động)

```bash
python scripts/run_all.py --dataset amazon-book
```

---

## Chẩn đoán mâu thuẫn kết quả [T1.4]

```bash
python scripts/diagnose_result_conflict.py --dataset amazon-book
# Xem: results/conflict_analysis.md
```

---

## Protocol Details

### Split Protocol (80/10/10)
- **Nguồn:** KGAT repo (xiangwang1223)
- **Quy trình:** Gộp train.txt + test.txt của KGAT → full_interactions → chia 80/10/10
- **Seed = 42** cố định cho reproducibility
- **Files output:** train.txt, **valid.txt** (không phải val.txt), test.txt
- KGAT amazon-book đã 10-core filtered → không cần filter thêm

### Cold-Start Protocol (Induced)
- Sample X% items làm cold_items (seed=42)
- Xóa khỏi training → model không bao giờ thấy chúng
- Evaluate trên interactions của cold_items trong test
- **Files:** **cold_item_ids.txt** (không phải cold_items.txt), **test_cold.txt** (không phải test.txt)
- Tương thích TaxPro-CL (Phương) — cùng seed, cùng protocol

### Fairness Protocol
- `embedding_dim = 64` — HARD constraint
- `optimizer = Adam`, `lr = 0.001` (KGAT: 1e-4)
- `batch_size = 2048`, `negative_sampling = 1`
- `full-item ranking` (eval_protocol: full)
- `5 seeds = {42, 0, 1, 2, 3}`, report mean ± std
- `gradient_clipping = 1.0` — tất cả models
- `temperature τ = 0.2` — cố định theo Fairness Sheet

### KGAT Invariant
```python
# BẮT BUỘC giữ trong toàn bộ pipeline:
assert max(all_item_ids) < n_items
# Không bao giờ re-index item IDs
# entity_emb[item_id] = embedding của item_id
```

---

## Thay đổi so với v7

Xem [CHANGELOG.md](CHANGELOG.md) để biết đầy đủ các thay đổi.

---

## Citation

```bibtex
@inproceedings{kg_lightgcn_icta2026,
  title     = {KG-LightGCN: Knowledge Graph Enhanced LightGCN
               with Cross-View Contrastive Learning for Recommendation},
  author    = {Tran, Duc Long},
  booktitle = {ICTA 2026},
  year      = {2026},
}
```

---

*Version: v10 (ICTA2026) | Data source: KGAT repo | Author: TranDucLong — AIDHSPKTTHY*
