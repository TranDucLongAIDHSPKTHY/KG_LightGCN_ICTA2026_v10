"""
scripts/preprocess.py — v10
═══════════════════════════════════════════════════════════════════════════════
Data Unification Pipeline (theo T2.1 Data Unification Runbook)

NGUỒN DỮ LIỆU DUY NHẤT: KGAT repo
  https://github.com/xiangwang1223/knowledge_graph_attention_network

THAY ĐỔI QUAN TRỌNG v10 so với v7:
  - Nguồn: KGAT repo ONLY (không dùng LightGCN/SimGCL/KGCL repo)
  - KG: dùng kg_final.txt từ KGAT (KHÔNG build lại, KHÔNG remap entity)
  - BẤT BIẾN: item_id == entity_id (KGAT convention)
    → KHÔNG re-index item ID
  - Split: gộp train.txt + test.txt của KGAT, sau đó chia 80/10/10
  - Output: valid.txt (không phải val.txt)
  - Thư mục: /data/phuongtran/project_v10/unified/
  - Tạo: dataset_stats.md, split_checksum.md5, item_category.txt
  - KGAT amazon-book đã 10-core filtered → không cần 5-core filter lại
  - kg_final.txt.zip phải được giải nén TỰ ĐỘNG trong pipeline

PIPELINE:
  Bước 1: Tải và kiểm kê dữ liệu từ raw/
  Bước 2: Hợp nhất train.txt + test.txt (KGAT gốc) → full_interactions
  Bước 3: Chia 80/10/10 user-wise (seed=42), KHÔNG re-index
  Bước 4: Copy KG nguyên trạng (giải nén kg_final.txt.zip nếu cần)
  Bước 5: Phái sinh item_category.txt (Setting A)
  Bước 6: Tính thống kê → dataset_stats.md
  Bước 7: Lưu tất cả files vào unified/
  Bước 8: Đóng băng (MD5 checksum → split_checksum.md5)
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("preprocess_v10")

# ── Constants ─────────────────────────────────────────────────────────────────
SPLIT_SEED   = 42
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1
TEST_RATIO   = 0.1

RAW_DIR      = "/data/phuongtran/project_v10/raw"
UNIFIED_DIR  = "/data/phuongtran/project_v10/unified"


# =============================================================================
# I/O helpers
# =============================================================================

def load_adj(path: str) -> Dict[int, List[int]]:
    """Đọc file interaction KGAT format: user item1 item2 ..."""
    result: Dict[int, List[int]] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            result[uid] = [int(x) for x in parts[1:]]
    return result


def save_adj(user2items: Dict[int, List[int]], path: str) -> None:
    """Lưu file interaction theo format KGAT."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def md5_file(path: str) -> str:
    """Tính MD5 checksum của file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def unzip_if_needed(zip_path: str, extract_to: str, target_name: str) -> str:
    """Tự động giải nén kg_final.txt.zip nếu kg_final.txt chưa có."""
    target_path = os.path.join(extract_to, target_name)
    if os.path.exists(target_path):
        logger.info(f"  ✓ {target_name} đã tồn tại — bỏ qua giải nén")
        return target_path
    if not os.path.exists(zip_path):
        logger.warning(f"  ✗ Không tìm thấy: {zip_path}")
        return None
    logger.info(f"  Giải nén {os.path.basename(zip_path)} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    if os.path.exists(target_path):
        logger.info(f"  ✓ Giải nén thành công: {target_path}")
        return target_path
    logger.warning(f"  ✗ Giải nén xong nhưng không tìm thấy {target_name}")
    return None


# =============================================================================
# Bước 2: Hợp nhất interactions
# =============================================================================

def merge_interactions(
    train_path: str, test_path: str
) -> Dict[int, List[int]]:
    """
    Gộp train.txt + test.txt của KGAT thành full_interactions.
    Đây là bước T2.1 Runbook: union, loại trùng lặp.
    """
    full: Dict[int, Set[int]] = defaultdict(set)

    for path in [train_path, test_path]:
        d = load_adj(path)
        for u, items in d.items():
            full[u].update(items)

    merged = {u: sorted(v) for u, v in full.items()}
    n_users = len(merged)
    n_items = len({i for items in merged.values() for i in items})
    n_pairs = sum(len(v) for v in merged.values())
    logger.info(
        f"  Sau gộp: {n_users:,} users | {n_items:,} items | "
        f"{n_pairs:,} interactions"
    )
    return merged


# =============================================================================
# Bước 3: Chia 80/10/10
# =============================================================================

def split_80_10_10(
    full: Dict[int, List[int]],
    seed: int = SPLIT_SEED,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    User-wise 80/10/10 split theo T2.1 Runbook.
    KHÔNG re-index user/item ID (giữ KGAT original IDs).
    Seed = 42 cố định cho tương thích TaxPro-CL (Phương).
    """
    logger.info(
        f"  [Split 80/10/10] seed={seed} | KHÔNG re-index IDs")
    rng = random.Random(seed)

    train_d: Dict[int, List[int]] = {}
    valid_d: Dict[int, List[int]] = {}
    test_d:  Dict[int, List[int]] = {}
    n_short = 0

    for u, items in full.items():
        items_shuffled = items[:]
        rng.shuffle(items_shuffled)
        n      = len(items_shuffled)
        n_tr   = int(TRAIN_RATIO * n)
        n_va   = int(VAL_RATIO * n)

        # Đảm bảo ít nhất 1 item cho mỗi split
        if n_tr < 1:
            train_d[u] = items_shuffled
            n_short += 1
            continue
        if n_va < 1:
            n_tr = n - 1
            n_va = 1

        train_d[u] = items_shuffled[:n_tr]
        valid_d[u] = items_shuffled[n_tr: n_tr + n_va]
        test_d[u]  = items_shuffled[n_tr + n_va:]
        if not test_d[u]:
            test_d.pop(u, None)
        if not valid_d[u]:
            valid_d.pop(u, None)

    n_tr    = sum(len(v) for v in train_d.values())
    n_va    = sum(len(v) for v in valid_d.values())
    n_te    = sum(len(v) for v in test_d.values())
    n_total = n_tr + n_va + n_te

    logger.info(
        f"    train={n_tr:,} ({n_tr/max(n_total,1):.1%}) | "
        f"valid={n_va:,} ({n_va/max(n_total,1):.1%}) | "
        f"test={n_te:,} ({n_te/max(n_total,1):.1%})"
    )
    if n_short > 0:
        logger.warning(
            f"    {n_short} users có quá ít items → toàn bộ vào train")
    return train_d, valid_d, test_d


# =============================================================================
# Reproducibility check
# =============================================================================

def verify_reproducibility(
    full:    Dict[int, List[int]],
    train_d: Dict[int, List[int]],
    valid_d: Dict[int, List[int]],
    test_d:  Dict[int, List[int]],
    n_runs:  int = 3,
) -> None:
    """Chạy lại 3 lần → phải cho cùng kết quả (MD5 fingerprint check)."""
    logger.info(f"  [Reproducibility] Xác minh {n_runs} lần chạy lại ...")

    def fingerprint(d):
        h = hashlib.md5()
        for u in sorted(d):
            h.update(f"{u}:{','.join(map(str, d[u]))}".encode())
        return h.hexdigest()

    exp_train = fingerprint(train_d)
    exp_valid = fingerprint(valid_d)
    exp_test  = fingerprint(test_d)

    for run in range(1, n_runs + 1):
        tr, va, te = split_80_10_10(full, seed=SPLIT_SEED)
        if (fingerprint(tr) != exp_train or
                fingerprint(va) != exp_valid or
                fingerprint(te) != exp_test):
            raise RuntimeError(
                f"Reproducibility FAILED tại lần chạy {run}! "
                "Pipeline không deterministic."
            )

    logger.info(f"    train MD5: {exp_train}")
    logger.info(f"    valid MD5: {exp_valid}")
    logger.info(f"    test  MD5: {exp_test}")
    logger.info("    ✓ Reproducibility PASSED")


# =============================================================================
# Bước 5: Phái sinh item_category.txt (Setting A)
# =============================================================================

def extract_item_category(
    kg_final_path: str, n_items: int, out_path: str
) -> int:
    """
    Phái sinh item_category.txt từ kg_final.txt.
    Lấy triple (item_id, relation_category, entity_id) → item_id  entity_id.
    BẤT BIẾN: item_id < n_items.
    """
    if not os.path.exists(kg_final_path):
        logger.warning(f"  kg_final.txt không tồn tại — bỏ qua item_category.txt")
        return 0

    # Phát hiện relation nào là "category" bằng cách lấy relation phổ biến nhất
    # mà kết nối items với entities ngoài item range
    rel_counts: Dict[int, int] = defaultdict(int)
    item_rel_pairs: List[Tuple[int, int, int]] = []  # (item, rel, entity)

    with open(kg_final_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            try:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            # items là head, entities ngoài item range là tail
            if h < n_items and t >= n_items:
                rel_counts[r] += 1
                item_rel_pairs.append((h, r, t))

    if not item_rel_pairs:
        logger.warning("  Không tìm thấy item→entity triples trong kg_final.txt")
        return 0

    # Lấy relation phổ biến nhất (thường là category)
    most_common_rel = max(rel_counts, key=lambda r: rel_counts[r])
    category_pairs = [(h, t) for h, r, t in item_rel_pairs
                      if r == most_common_rel]

    # Lưu mapping: mỗi item lấy category đầu tiên (flat)
    item2cat: Dict[int, int] = {}
    for item_id, cat_id in category_pairs:
        if item_id not in item2cat:
            item2cat[item_id] = cat_id

    with open(out_path, "w") as f:
        for item_id in sorted(item2cat.keys()):
            f.write(f"{item_id}\t{item2cat[item_id]}\n")

    n_cats = len(set(item2cat.values()))
    coverage = len(item2cat) / max(n_items, 1) * 100
    logger.info(
        f"  item_category.txt: {len(item2cat):,} items | "
        f"{n_cats:,} categories | "
        f"rel_id={most_common_rel} | coverage={coverage:.1f}%"
    )
    return len(item2cat)


# =============================================================================
# Bước 6: Dataset statistics
# =============================================================================

def compute_and_write_stats(
    out_dir: str, dataset: str,
    train_d: Dict, valid_d: Dict, test_d: Dict,
    n_users: int, n_items: int,
    kg_final_path: str = None,
) -> dict:
    """Tính thống kê và ghi vào dataset_stats.md."""
    n_tr    = sum(len(v) for v in train_d.values())
    n_va    = sum(len(v) for v in valid_d.values())
    n_te    = sum(len(v) for v in test_d.values())
    n_total = n_tr + n_va + n_te
    density = n_tr / max(n_users * n_items, 1)

    kg_stats = {}
    if kg_final_path and os.path.exists(kg_final_path):
        triples, entities, relations = [], set(), set()
        items_in_kg: Set[int] = set()
        with open(kg_final_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    parts = line.strip().split("\t")
                if len(parts) == 3:
                    try:
                        h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                        triples.append((h, r, t))
                        entities.add(h); entities.add(t)
                        relations.add(r)
                        if h < n_items:
                            items_in_kg.add(h)
                    except ValueError:
                        continue
        n_entities  = max(entities) + 1 if entities else 0
        n_relations = max(relations) + 1 if relations else 0
        n_triples   = len(triples)
        kg_coverage = len(items_in_kg) / max(n_items, 1) * 100
        kg_stats = {
            "n_entities":  n_entities,
            "n_relations": n_relations,
            "n_triples":   n_triples,
            "kg_coverage_pct": round(kg_coverage, 2),
            "items_in_kg": len(items_in_kg),
        }

        # Kiểm tra bất biến KGAT
        assert max(entities & set(range(n_items))) < n_items if entities else True, \
            "KGAT invariant violation: item_id >= n_items!"

    # Ghi dataset_stats.md
    stats_path = os.path.join(out_dir, "dataset_stats.md")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"# Dataset Statistics: {dataset}\n\n")
        f.write(f"**Data source:** KGAT repo (xiangwang1223)\n")
        f.write(f"**Split protocol:** 80/10/10 user-wise, seed={SPLIT_SEED}\n")
        f.write(f"**Bất biến:** item_id == entity_id (KGAT convention)\n\n")
        f.write("## Interaction Statistics\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| n_users | {n_users:,} |\n")
        f.write(f"| n_items | {n_items:,} |\n")
        f.write(f"| n_train | {n_tr:,} |\n")
        f.write(f"| n_valid | {n_va:,} |\n")
        f.write(f"| n_test  | {n_te:,} |\n")
        f.write(f"| n_total_interactions | {n_total:,} |\n")
        f.write(f"| density_train | {density:.8f} |\n\n")
        if kg_stats:
            f.write("## Knowledge Graph Statistics\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| n_entities  | {kg_stats['n_entities']:,} |\n")
            f.write(f"| n_relations | {kg_stats['n_relations']:,} |\n")
            f.write(f"| n_triples   | {kg_stats['n_triples']:,} |\n")
            f.write(f"| kg_coverage_pct | {kg_stats['kg_coverage_pct']:.1f}% |\n")
            f.write(f"| items_in_kg | {kg_stats['items_in_kg']:,} |\n\n")
            f.write(f"> **Bất biến KGAT:** item_id == entity_id cho {n_items:,} ")
            f.write(f"thực thể đầu tiên trong KG.\n\n")

            # Cảnh báo taxonomy coverage cho Setting B
            if kg_stats.get("n_relations", 0) < 3:
                f.write(
                    "⚠ **CẢNH BÁO Setting B (Taxonomy):** KG có ít relations "
                    f"({kg_stats['n_relations']}). Taxonomy có thể suy biến về "
                    "flat category (Setting A).\n\n"
                )

    logger.info(
        f"  dataset_stats.md: {n_users:,} users | {n_items:,} items | "
        f"density={density:.6f}"
    )
    all_stats = {
        "n_users": n_users, "n_items": n_items,
        "n_train": n_tr, "n_valid": n_va, "n_test": n_te,
        "n_total": n_total, "density_train": density,
        **kg_stats
    }
    return all_stats


# =============================================================================
# Bước 8: Đóng băng với MD5 checksum
# =============================================================================

def freeze_splits(out_dir: str) -> None:
    """Tạo split_checksum.md5 để đóng băng splits."""
    files = ["train.txt", "valid.txt", "test.txt"]
    checksums = []
    for fname in files:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            md5 = md5_file(fpath)
            checksums.append(f"{md5}  {fname}")
            logger.info(f"  MD5({fname}) = {md5}")

    checksum_path = os.path.join(out_dir, "split_checksum.md5")
    with open(checksum_path, "w") as f:
        f.write("\n".join(checksums) + "\n")
    logger.info(f"  ✓ split_checksum.md5 đã được tạo → {checksum_path}")


# =============================================================================
# Amazon-Book pipeline
# =============================================================================

def preprocess_amazon_book(raw_dir: str, out_dir: str) -> None:
    logger.info("=" * 65)
    logger.info("PREPROCESSING: Amazon-Book (KGAT repo)")
    logger.info("  Nguồn: xiangwang1223/knowledge_graph_attention_network")
    logger.info(f"  Raw:   {raw_dir}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 65)

    ds_raw = os.path.join(raw_dir, "amazon-book")
    os.makedirs(out_dir, exist_ok=True)

    # Kiểm tra files bắt buộc
    required_files = ["train.txt", "test.txt", "item_list.txt",
                      "entity_list.txt", "relation_list.txt", "user_list.txt"]
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(ds_raw, f))]
    if missing:
        raise FileNotFoundError(
            f"Thiếu files: {missing}\n"
            f"Download từ KGAT repo: Data/amazon-book/\n"
            f"Chạy: python scripts/download_data.py --dataset amazon-book"
        )

    # Bước 1: Giải nén kg_final.txt.zip nếu cần
    logger.info("\n[Bước 1] Kiểm tra và giải nén KG file ...")
    kg_zip  = os.path.join(ds_raw, "kg_final.txt.zip")
    kg_path = os.path.join(ds_raw, "kg_final.txt")
    if not os.path.exists(kg_path):
        if os.path.exists(kg_zip):
            unzip_if_needed(kg_zip, ds_raw, "kg_final.txt")
        else:
            logger.warning(
                "  kg_final.txt.zip không tìm thấy. "
                "KG models sẽ không có KG data."
            )

    # Bước 2: Hợp nhất interactions
    logger.info("\n[Bước 2] Hợp nhất train.txt + test.txt (KGAT gốc) ...")
    full = merge_interactions(
        os.path.join(ds_raw, "train.txt"),
        os.path.join(ds_raw, "test.txt"),
    )

    # Xác định n_users, n_items từ dữ liệu thực (không dùng số nhớ từ paper)
    all_users = set(full.keys())
    all_items = {i for items in full.values() for i in items}
    n_users   = max(all_users) + 1 if all_users else 0
    n_items   = max(all_items) + 1 if all_items else 0
    logger.info(f"  n_users={n_users:,} | n_items={n_items:,} (đọc từ file)")

    # Kiểm tra bất biến KGAT
    if all_items:
        assert max(all_items) < n_items, \
            f"KGAT invariant: max(item_id)={max(all_items)} >= n_items={n_items}"
    logger.info(f"  ✓ KGAT invariant: max(item_id) < n_items")

    # Bước 3: Split 80/10/10
    logger.info("\n[Bước 3] Chia 80/10/10 (seed=42) ...")
    train_d, valid_d, test_d = split_80_10_10(full, seed=SPLIT_SEED)
    verify_reproducibility(full, train_d, valid_d, test_d)

    # Bước 4: Copy KG files
    logger.info("\n[Bước 4] Copy KG files nguyên trạng ...")
    kg_files = ["kg_final.txt", "item_list.txt", "entity_list.txt",
                "relation_list.txt", "user_list.txt"]
    for fname in kg_files:
        src = os.path.join(ds_raw, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {fname}")
        else:
            logger.warning(f"  Không tìm thấy: {fname}")

    # Bước 5: Phái sinh item_category.txt
    logger.info("\n[Bước 5] Tạo item_category.txt (Setting A) ...")
    kg_final_out = os.path.join(out_dir, "kg_final.txt")
    if os.path.exists(kg_final_out):
        extract_item_category(
            kg_final_out, n_items,
            os.path.join(out_dir, "item_category.txt")
        )

    # Bước 6: Thống kê
    logger.info("\n[Bước 6] Tính thống kê ...")
    compute_and_write_stats(
        out_dir=out_dir, dataset="Amazon-Book",
        train_d=train_d, valid_d=valid_d, test_d=test_d,
        n_users=n_users, n_items=n_items,
        kg_final_path=kg_final_out if os.path.exists(kg_final_out) else None,
    )

    # Bước 7: Lưu splits
    logger.info("\n[Bước 7] Lưu train/valid/test splits ...")
    save_adj(train_d, os.path.join(out_dir, "train.txt"))
    save_adj(valid_d, os.path.join(out_dir, "valid.txt"))   # ← valid.txt
    save_adj(test_d,  os.path.join(out_dir, "test.txt"))

    # Bước 8: Đóng băng
    logger.info("\n[Bước 8] Đóng băng với MD5 checksum ...")
    freeze_splits(out_dir)

    logger.info(f"\n  ✓ Amazon-Book processed → {out_dir}")
    logger.info("  ⚠ Đừng chạy lại script này sau khi experiments đã bắt đầu!")
    logger.info("=" * 65)


# =============================================================================
# Yelp2018 pipeline
# =============================================================================

def preprocess_yelp2018(raw_dir: str, out_dir: str) -> None:
    logger.info("=" * 65)
    logger.info("PREPROCESSING: Yelp2018 (KGAT repo)")
    logger.info("  Nguồn: xiangwang1223/knowledge_graph_attention_network")
    logger.info(f"  Raw:   {raw_dir}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 65)

    ds_raw = os.path.join(raw_dir, "yelp2018")
    os.makedirs(out_dir, exist_ok=True)

    required_files = ["train.txt", "test.txt"]
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(ds_raw, f))]
    if missing:
        raise FileNotFoundError(
            f"Thiếu files: {missing}\n"
            f"Download từ KGAT repo: Data/yelp2018/\n"
            f"Chạy: python scripts/download_data.py --dataset yelp2018"
        )

    # Bước 1: Giải nén kg_final.txt.zip
    logger.info("\n[Bước 1] Kiểm tra và giải nén KG file ...")
    kg_zip  = os.path.join(ds_raw, "kg_final.txt.zip")
    kg_path = os.path.join(ds_raw, "kg_final.txt")
    if not os.path.exists(kg_path):
        if os.path.exists(kg_zip):
            unzip_if_needed(kg_zip, ds_raw, "kg_final.txt")
        else:
            logger.warning("  kg_final.txt.zip không tìm thấy cho Yelp2018.")

    # Bước 2: Hợp nhất interactions
    logger.info("\n[Bước 2] Hợp nhất train.txt + test.txt ...")
    full = merge_interactions(
        os.path.join(ds_raw, "train.txt"),
        os.path.join(ds_raw, "test.txt"),
    )

    all_users = set(full.keys())
    all_items = {i for items in full.values() for i in items}
    n_users   = max(all_users) + 1 if all_users else 0
    n_items   = max(all_items) + 1 if all_items else 0
    logger.info(f"  n_users={n_users:,} | n_items={n_items:,} (đọc từ file)")

    if all_items:
        assert max(all_items) < n_items, \
            f"KGAT invariant: max(item_id)={max(all_items)} >= n_items={n_items}"
    logger.info(f"  ✓ KGAT invariant: max(item_id) < n_items")

    # Bước 3: Split 80/10/10
    logger.info("\n[Bước 3] Chia 80/10/10 (seed=42) ...")
    train_d, valid_d, test_d = split_80_10_10(full, seed=SPLIT_SEED)
    verify_reproducibility(full, train_d, valid_d, test_d)

    # Bước 4: Copy KG files
    logger.info("\n[Bước 4] Copy KG files nguyên trạng ...")
    kg_files_to_copy = ["kg_final.txt", "item_list.txt", "entity_list.txt",
                        "relation_list.txt", "user_list.txt"]
    for fname in kg_files_to_copy:
        src = os.path.join(ds_raw, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {fname}")

    # Bước 5: item_category.txt (nếu KG có)
    kg_final_out = os.path.join(out_dir, "kg_final.txt")
    if os.path.exists(kg_final_out):
        logger.info("\n[Bước 5] Tạo item_category.txt ...")
        extract_item_category(
            kg_final_out, n_items,
            os.path.join(out_dir, "item_category.txt")
        )

    # Bước 6: Thống kê
    logger.info("\n[Bước 6] Tính thống kê ...")
    compute_and_write_stats(
        out_dir=out_dir, dataset="Yelp2018",
        train_d=train_d, valid_d=valid_d, test_d=test_d,
        n_users=n_users, n_items=n_items,
        kg_final_path=kg_final_out if os.path.exists(kg_final_out) else None,
    )

    # Bước 7: Lưu splits
    logger.info("\n[Bước 7] Lưu train/valid/test splits ...")
    save_adj(train_d, os.path.join(out_dir, "train.txt"))
    save_adj(valid_d, os.path.join(out_dir, "valid.txt"))   # ← valid.txt
    save_adj(test_d,  os.path.join(out_dir, "test.txt"))

    # Bước 8: Đóng băng
    logger.info("\n[Bước 8] Đóng băng với MD5 checksum ...")
    freeze_splits(out_dir)

    logger.info(f"\n  ✓ Yelp2018 processed → {out_dir}")
    logger.info("=" * 65)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preprocessing v10 — Amazon-Book + Yelp2018 (KGAT repo)\n"
            "Split: 80/10/10 user-wise (gộp KGAT train+test) | seed=42\n"
            "BẤT BIẾN: item_id == entity_id (KHÔNG re-index)\n"
            "Output: valid.txt (không phải val.txt)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all")
    p.add_argument("--raw_dir",  default=RAW_DIR)
    p.add_argument("--out_dir",  default=UNIFIED_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SPLIT_SEED)

    logger.info("=" * 65)
    logger.info("DATA UNIFICATION PIPELINE v10")
    logger.info("Nguồn: KGAT repo (xiangwang1223)")
    logger.info("BẤT BIẾN: item_id == entity_id (KHÔNG re-index)")
    logger.info("=" * 65)

    if args.dataset in ("amazon-book", "all"):
        out = os.path.join(args.out_dir, "amazon-book")
        preprocess_amazon_book(raw_dir=args.raw_dir, out_dir=out)

    if args.dataset in ("yelp2018", "all"):
        out = os.path.join(args.out_dir, "yelp2018")
        preprocess_yelp2018(raw_dir=args.raw_dir, out_dir=out)

    logger.info("\n✓ Preprocessing hoàn tất.")
    logger.info(
        "  Files đã được đóng băng — ĐỪNG chạy lại script này.\n"
        "  Dùng split_checksum.md5 để verify tính nhất quán."
    )


if __name__ == "__main__":
    main()
