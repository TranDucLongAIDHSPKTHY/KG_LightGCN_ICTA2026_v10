"""
scripts/build_cold_split.py — v10
Xây dựng cold-start splits (induced cold-start protocol T3.1).

THAY ĐỔI v10 so với v7:
  - cold_items.txt       → cold_item_ids.txt   (tương thích TaxPro-CL)
  - test.txt (cold)      → test_cold.txt        (tránh nhầm với warm test)
  - valid.txt            thay cho val.txt
  - Paths: /data/phuongtran/project_v10/unified/amazon-book/cold_XX/
  - Seed = 42 cố định (tương thích Phương/TaxPro-CL)

Protocol:
  1. Pool items từ train + valid + test của unified/
  2. Sample X% làm cold_items (seed=42)
  3. train_cold.txt  = train.txt gốc, xóa cold_items
  4. valid_cold.txt  = valid.txt gốc, xóa cold_items  
  5. test_cold.txt   = CHỈ giữ interactions trong test.txt có item ∈ cold_items
  6. cold_item_ids.txt = danh sách cold item IDs
  7. Validate: cold_items ∉ train_cold AND cold_items ⊆ test_cold
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("build_cold_split_v10")

COLD_SEED   = 42
COLD_RATIOS = [10, 20, 30]
UNIFIED_DIR = "/data/phuongtran/project_v10/unified"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read(path: str) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                result[int(parts[0])] = [int(x) for x in parts[1:]]
            except ValueError:
                continue
    return result


def _write(path: str, user2items: Dict[int, List[int]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def _read_kg(path: str) -> List[Tuple[int, int, int]]:
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) == 3:
                try:
                    triples.append(
                        (int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    return triples


def _write_kg(path: str, triples: List[Tuple]) -> None:
    with open(path, "w") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


# ── Sample cold items ─────────────────────────────────────────────────────────

def sample_cold_items(
    all_items: Set[int], ratio: int, seed: int = COLD_SEED,
) -> Tuple[Set[int], dict]:
    """Sample X% items làm cold items (seed=42 cố định)."""
    sorted_items = sorted(all_items)
    n_total = len(sorted_items)
    n_cold  = max(1, int(n_total * ratio / 100))

    rng      = random.Random(seed)
    shuffled = sorted_items[:]
    rng.shuffle(shuffled)
    cold_items = set(shuffled[:n_cold])

    actual_ratio = n_cold / n_total * 100
    info = {
        "n_total_items":     n_total,
        "n_cold_items":      n_cold,
        "n_warm_items":      n_total - n_cold,
        "target_ratio_pct":  ratio,
        "actual_ratio_pct":  round(actual_ratio, 2),
        "seed":              seed,
        "method":            "induced_cold_start",
    }
    logger.info(
        f"    Total items: {n_total:,} | "
        f"Cold: {n_cold:,} ({actual_ratio:.1f}%) | "
        f"Warm: {n_total - n_cold:,}"
    )
    return cold_items, info


# ── Make cold splits ──────────────────────────────────────────────────────────

def make_cold_train_valid(
    user2items: Dict[int, List[int]], cold_items: Set[int]
) -> Dict[int, List[int]]:
    """Xóa cold_items khỏi train/valid."""
    result = {}
    for uid, items in user2items.items():
        warm = [i for i in items if i not in cold_items]
        if warm:
            result[uid] = warm
    return result


def make_test_cold(
    user2items: Dict[int, List[int]], cold_items: Set[int]
) -> Dict[int, List[int]]:
    """[v10] Tạo test_cold.txt — CHỈ giữ interactions của cold_items."""
    result = {}
    for uid, items in user2items.items():
        cold_gt = [i for i in items if i in cold_items]
        if cold_gt:
            result[uid] = cold_gt
    return result


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_cold_split(
    cold_items:  Set[int],
    train_cold:  Dict[int, List[int]],
    valid_cold:  Dict[int, List[int]],
    test_cold:   Dict[int, List[int]],
) -> None:
    """Kiểm tra: không có cold_items trong train/valid, có cold_items trong test_cold."""
    train_items  = {i for items in train_cold.values() for i in items}
    valid_items  = {i for items in valid_cold.values() for i in items}
    test_items   = {i for items in test_cold.values()  for i in items}

    leaked_train = cold_items & train_items
    leaked_valid = cold_items & valid_items

    if leaked_train:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(leaked_train):,} cold items vẫn còn trong train_cold!")
    if leaked_valid:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(leaked_valid):,} cold items vẫn còn trong valid_cold!")

    cold_in_test = cold_items & test_items
    if not cold_in_test:
        raise RuntimeError(
            "Không có cold item nào có ground truth trong test_cold! "
            "Cold evaluation sẽ vô nghĩa.")

    coverage = len(cold_in_test) / len(cold_items) * 100
    logger.info(
        f"    ✓ Validation OK: không leakage | "
        f"{len(cold_in_test):,}/{len(cold_items):,} "
        f"({coverage:.1f}%) cold items có test ground truth"
    )


# ── Build one cold split ──────────────────────────────────────────────────────

def build_cold_split(
    processed_dir: str, dataset: str, ratio: int, seed: int = COLD_SEED,
) -> None:
    """
    Xây dựng cold_{ratio}/ split cho một dataset.

    Output (v10):
      train.txt          — interactions với cold_items đã xóa
      valid.txt          — interactions với cold_items đã xóa  ← valid, không phải val
      test_cold.txt      — CHỈ interactions của cold_items     ← test_cold, không phải test
      cold_item_ids.txt  — danh sách cold item IDs             ← cold_item_ids, không phải cold_items
      cold_stats.json    — statistics
      cold_protocol.json — reproduction instructions
    """
    logger.info(
        f"\n{'─'*65}\n"
        f"  Cold-{ratio} | {dataset} | seed={seed}\n"
        f"  Phương pháp: Induced cold-start\n"
        f"{'─'*65}"
    )

    train_path = os.path.join(processed_dir, "train.txt")
    valid_path = os.path.join(processed_dir, "valid.txt")   # [v10] valid.txt
    test_path  = os.path.join(processed_dir, "test.txt")

    for p in [train_path, valid_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Thiếu: {p}\n"
                "Chạy scripts/preprocess.py trước."
            )

    user2train = _read(train_path)
    user2valid = _read(valid_path)
    user2test  = _read(test_path)

    # Pool items từ tất cả splits
    all_items: Set[int] = set()
    for d in [user2train, user2valid, user2test]:
        for items in d.values():
            all_items.update(items)

    # Sample cold items
    cold_items, cold_info = sample_cold_items(all_items, ratio, seed)

    # Tạo cold splits
    train_cold = make_cold_train_valid(user2train, cold_items)
    valid_cold = make_cold_train_valid(user2valid, cold_items)
    test_cold  = make_test_cold(user2test, cold_items)          # [v10] test_cold

    # Statistics
    n_tr_before       = sum(len(v) for v in user2train.values())
    n_tr_after        = sum(len(v) for v in train_cold.values())
    n_va_before       = sum(len(v) for v in user2valid.values())
    n_va_after        = sum(len(v) for v in valid_cold.values())
    n_cold_test_users = len(test_cold)
    n_cold_test_pairs = sum(len(v) for v in test_cold.values())
    n_total_test      = sum(len(v) for v in user2test.values())

    logger.info(f"    Train: {n_tr_before:,} → {n_tr_after:,} (-{n_tr_before-n_tr_after:,})")
    logger.info(f"    Valid: {n_va_before:,} → {n_va_after:,} (-{n_va_before-n_va_after:,})")
    logger.info(f"    test_cold: {n_cold_test_pairs:,} pairs | {n_cold_test_users:,} users")

    # Validate
    validate_cold_split(cold_items, train_cold, valid_cold, test_cold)

    # KG (lọc cold-item triples)
    kg_cold: Optional[List[Tuple[int, int, int]]] = None
    for kg_fname in ["kg_final.txt", "kg.txt"]:
        kg_path = os.path.join(processed_dir, kg_fname)
        if os.path.exists(kg_path):
            all_triples = _read_kg(kg_path)
            kg_cold     = [(h, r, t) for h, r, t in all_triples
                           if h in cold_items or t in cold_items]
            logger.info(
                f"    KG ({kg_fname}): {len(all_triples):,} → "
                f"{len(kg_cold):,} cold-related triples"
            )
            break

    # Lưu output
    out_dir = os.path.join(processed_dir, f"cold_{ratio}")
    os.makedirs(out_dir, exist_ok=True)

    _write(os.path.join(out_dir, "train.txt"), train_cold)
    _write(os.path.join(out_dir, "valid.txt"), valid_cold)          # [v10] valid.txt

    # [v10] test_cold.txt (không phải test.txt)
    _write(os.path.join(out_dir, "test_cold.txt"), test_cold)

    # [v10] cold_item_ids.txt (không phải cold_items.txt)
    with open(os.path.join(out_dir, "cold_item_ids.txt"), "w") as f:
        for iid in sorted(cold_items):
            f.write(f"{iid}\n")

    if kg_cold is not None:
        _write_kg(os.path.join(out_dir, "kg_final.txt"), kg_cold)

    # Stats JSON
    stats = {
        "dataset":              dataset,
        "ratio_pct":            ratio,
        "seed":                 seed,
        "method":               "induced_cold_start",
        **cold_info,
        "n_train_before":       n_tr_before,
        "n_train_after":        n_tr_after,
        "n_valid_before":       n_va_before,
        "n_valid_after":        n_va_after,
        "n_cold_test_users":    n_cold_test_users,
        "n_cold_test_pairs":    n_cold_test_pairs,
        "n_total_test_pairs":   n_total_test,
        "cold_test_coverage_pct": round(
            n_cold_test_pairs / max(n_total_test, 1) * 100, 2),
        "kg_triples_cold":      len(kg_cold) if kg_cold else 0,
        "eval_metrics":         [
            "HR@10_cold", "NDCG@10_cold", "Recall@10_cold",
            "HR@20_cold", "NDCG@20_cold", "Recall@20_cold",
        ],
        "v10_file_names": {
            "cold_ids":     "cold_item_ids.txt",   # [v10] thay cold_items.txt
            "test_cold":    "test_cold.txt",         # [v10] thay test.txt
            "valid_cold":   "valid.txt",             # [v10] thay val.txt
        },
        "validation": {
            "no_leakage_to_train":  True,
            "no_leakage_to_valid":  True,
            "has_test_groundtruth": True,
        },
    }
    with open(os.path.join(out_dir, "cold_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Protocol file
    protocol = {
        "cold_split_name":     f"Cold-{ratio}",
        "dataset":             dataset,
        "method":              "induced_cold_start",
        "cold_ratio_pct":      ratio,
        "n_cold_items":        cold_info["n_cold_items"],
        "seed":                seed,
        "v10_changes": {
            "cold_item_ids": "cold_item_ids.txt (v7: cold_items.txt)",
            "test_split":    "test_cold.txt (v7: test.txt)",
            "valid_split":   "valid.txt (v7: val.txt)",
        },
        "phương_compatibility": (
            "TaxPro-CL dùng induced cold-start với seed=42. "
            "cold_item_ids.txt có thể chia sẻ để so sánh trực tiếp."
        ),
        "eval_metrics": [
            "HR@10_cold", "NDCG@10_cold", "Recall@10_cold",
            "HR@20_cold", "NDCG@20_cold", "Recall@20_cold",
        ],
        "how_to_reproduce": [
            "1. python scripts/preprocess.py --dataset <ds>",
            "2. python scripts/build_cold_split.py --dataset <ds> --ratio 10 20 30",
            "3. Dùng cold_item_ids.txt + cold_XX/test_cold.txt để evaluate",
        ],
    }
    with open(os.path.join(out_dir, "cold_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    logger.info(f"    ✓ Saved → {out_dir}")
    logger.info(f"      Files: train.txt, valid.txt, test_cold.txt, cold_item_ids.txt")


def build_all_cold_splits(
    processed_dir: str, dataset: str,
    ratios: List[int] = None, seed: int = COLD_SEED,
) -> None:
    if ratios is None:
        ratios = COLD_RATIOS

    logger.info(f"\n{'='*65}")
    logger.info(f"  COLD SPLITS (v10): {dataset}")
    logger.info(f"  Levels: {ratios} | Seed: {seed}")
    logger.info(f"{'='*65}")

    for fname in ["train.txt", "valid.txt", "test.txt"]:
        p = os.path.join(processed_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Thiếu: {p}\nChạy scripts/preprocess.py trước.")

    for ratio in ratios:
        build_cold_split(processed_dir, dataset, ratio, seed)

    logger.info(f"\n  ✓ Tất cả cold splits cho {dataset}:")
    for ratio in ratios:
        sp = os.path.join(processed_dir, f"cold_{ratio}", "cold_stats.json")
        if os.path.exists(sp):
            with open(sp) as f:
                s = json.load(f)
            logger.info(
                f"    cold_{ratio:>2}: "
                f"{s['n_cold_items']:,} cold items | "
                f"{s['n_cold_test_pairs']:,} cold test pairs | "
                f"coverage={s['cold_test_coverage_pct']:.1f}%"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build cold-start splits v10 [CHK-T3.1]\n"
            "Output: test_cold.txt, cold_item_ids.txt, valid.txt\n"
            "seed=42 để tương thích TaxPro-CL (Phương)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all")
    p.add_argument("--ratio", nargs="+", type=int, default=COLD_RATIOS)
    p.add_argument("--data_dir", default=UNIFIED_DIR)
    p.add_argument("--seed", type=int, default=COLD_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    datasets = (["amazon-book", "yelp2018"] if args.dataset == "all"
                else [args.dataset])

    for ds in datasets:
        processed_dir = os.path.join(args.data_dir, ds)
        if not os.path.isdir(processed_dir):
            logger.error(
                f"Thư mục không tồn tại: {processed_dir}\n"
                "Chạy scripts/preprocess.py trước.")
            continue
        build_all_cold_splits(processed_dir, ds, args.ratio, args.seed)

    logger.info(
        "\n✓ Tất cả cold splits đã tạo xong.\n"
        "  [v10] Files: test_cold.txt, cold_item_ids.txt, valid.txt\n"
        "  Tương thích TaxPro-CL (Phương) — cùng seed=42."
    )


if __name__ == "__main__":
    main()
