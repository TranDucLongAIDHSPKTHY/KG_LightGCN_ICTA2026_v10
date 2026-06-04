"""
scripts/download_data.py — v10
═══════════════════════════════════════════════════════════════════════════════
Download raw data từ KGAT repo (nguồn DUY NHẤT).

NGUỒN: KGAT repo — xiangwang1223/knowledge_graph_attention_network
  https://github.com/xiangwang1223/knowledge_graph_attention_network

THAY ĐỔI v10:
  - Không còn dùng LightGCN/SimGCL/KGCL repo
  - Tự động giải nén kg_final.txt.zip
  - Paths: /data/phuongtran/project_v10/raw/

Files cần thiết (từ KGAT repo):
  Amazon-Book: Data/amazon-book/
    train.txt, test.txt, item_list.txt, entity_list.txt,
    relation_list.txt, user_list.txt, kg_final.txt.zip
  Yelp2018: Data/yelp2018/
    train.txt, test.txt, item_list.txt, entity_list.txt,
    relation_list.txt, user_list.txt, kg_final.txt.zip

LƯU Ý: kg_final.txt.zip được tự động giải nén trong preprocess.py.
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("download_data_v10")

# ── KGAT repo base URL ────────────────────────────────────────────────────────
KGAT_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "xiangwang1223/knowledge_graph_attention_network/master/Data"
)
KGAT_ZIP_BASE = (
    "https://github.com/xiangwang1223/"
    "knowledge_graph_attention_network/raw/master/Data"
)

# ── File registry — KGAT repo ─────────────────────────────────────────────────
AMAZON_BOOK_FILES = {
    "train.txt":          f"{KGAT_RAW_BASE}/amazon-book/train.txt",
    "test.txt":           f"{KGAT_RAW_BASE}/amazon-book/test.txt",
    "item_list.txt":      f"{KGAT_RAW_BASE}/amazon-book/item_list.txt",
    "entity_list.txt":    f"{KGAT_RAW_BASE}/amazon-book/entity_list.txt",
    "relation_list.txt":  f"{KGAT_RAW_BASE}/amazon-book/relation_list.txt",
    "user_list.txt":      f"{KGAT_RAW_BASE}/amazon-book/user_list.txt",
    "kg_final.txt.zip":   f"{KGAT_ZIP_BASE}/amazon-book/kg_final.txt.zip",
}

YELP2018_FILES = {
    "train.txt":          f"{KGAT_RAW_BASE}/yelp2018/train.txt",
    "test.txt":           f"{KGAT_RAW_BASE}/yelp2018/test.txt",
    "item_list.txt":      f"{KGAT_RAW_BASE}/yelp2018/item_list.txt",
    "entity_list.txt":    f"{KGAT_RAW_BASE}/yelp2018/entity_list.txt",
    "relation_list.txt":  f"{KGAT_RAW_BASE}/yelp2018/relation_list.txt",
    "user_list.txt":      f"{KGAT_RAW_BASE}/yelp2018/user_list.txt",
    "kg_final.txt.zip":   f"{KGAT_ZIP_BASE}/yelp2018/kg_final.txt.zip",
}


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_file(url: str, dest_path: str, desc: str = "") -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    label = desc or os.path.basename(dest_path)
    logger.info(f"  Đang tải: {label}")
    logger.info(f"    URL : {url}")
    logger.info(f"    Dest: {dest_path}")

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct  = min(downloaded / total_size * 100, 100)
                mb_d = downloaded  / 1024 / 1024
                mb_t = total_size  / 1024 / 1024
                print(f"\r    Progress: {pct:.1f}% ({mb_d:.1f}/{mb_t:.1f} MB)",
                      end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, _progress)
        print()
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        logger.info(f"    ✓ Tải thành công ({size_mb:.1f} MB)")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print()
        if os.path.exists(dest_path):
            os.remove(dest_path)
        logger.warning(f"    ✗ Thất bại: {e}")
        return False


def _check_file(path: str, min_bytes: int = 100) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def _auto_unzip(zip_path: str, extract_to: str) -> bool:
    """Tự động giải nén .zip file."""
    if not os.path.exists(zip_path):
        return False
    logger.info(f"  Giải nén: {os.path.basename(zip_path)}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logger.info(f"    ✓ Giải nén thành công")
        return True
    except Exception as e:
        logger.warning(f"    ✗ Lỗi giải nén: {e}")
        return False


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_amazon_book(raw_dir: str, check_only: bool = False) -> bool:
    ds_dir = os.path.join(raw_dir, "amazon-book")
    os.makedirs(ds_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Amazon-Book — KGAT repo")
    logger.info("  Source: xiangwang1223/knowledge_graph_attention_network")
    logger.info("=" * 65)

    all_ok = True
    for filename, url in AMAZON_BOOK_FILES.items():
        dest = os.path.join(ds_dir, filename)
        if _check_file(dest):
            logger.info(f"  ✓ {filename} đã tồn tại — bỏ qua")
            continue
        if check_only:
            logger.warning(f"  ✗ {filename} THIẾU")
            all_ok = False
            continue
        ok = _download_file(url, dest, filename)
        if not ok:
            all_ok = False

    # Tự động giải nén kg_final.txt.zip
    zip_path = os.path.join(ds_dir, "kg_final.txt.zip")
    kg_path  = os.path.join(ds_dir, "kg_final.txt")
    if not _check_file(kg_path) and _check_file(zip_path):
        _auto_unzip(zip_path, ds_dir)

    if _check_file(kg_path):
        logger.info(f"  ✓ kg_final.txt sẵn sàng")
    else:
        logger.warning(
            "\n  kg_final.txt không có. Hướng dẫn thủ công:\n"
            "    git clone https://github.com/xiangwang1223/"
            "knowledge_graph_attention_network\n"
            f"    cp -r knowledge_graph_attention_network/Data/amazon-book {ds_dir}\n"
        )
        all_ok = False

    return all_ok


def download_yelp2018(raw_dir: str, check_only: bool = False) -> bool:
    ds_dir = os.path.join(raw_dir, "yelp2018")
    os.makedirs(ds_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Yelp2018 — KGAT repo")
    logger.info("  Source: xiangwang1223/knowledge_graph_attention_network")
    logger.info("  LƯU Ý: Dùng KGAT repo, KHÔNG phải KGCL/SimGCL/LightGCN repo")
    logger.info("=" * 65)

    all_ok = True
    for filename, url in YELP2018_FILES.items():
        dest = os.path.join(ds_dir, filename)
        if _check_file(dest):
            logger.info(f"  ✓ {filename} đã tồn tại — bỏ qua")
            continue
        if check_only:
            logger.warning(f"  ✗ {filename} THIẾU")
            all_ok = False
            continue
        ok = _download_file(url, dest, filename)
        if not ok:
            all_ok = False

    # Tự động giải nén
    zip_path = os.path.join(ds_dir, "kg_final.txt.zip")
    kg_path  = os.path.join(ds_dir, "kg_final.txt")
    if not _check_file(kg_path) and _check_file(zip_path):
        _auto_unzip(zip_path, ds_dir)

    if not _check_file(kg_path):
        logger.warning(
            "\n  kg_final.txt.zip không tải được tự động. Hướng dẫn thủ công:\n"
            "    git clone https://github.com/xiangwang1223/"
            "knowledge_graph_attention_network\n"
            f"    cp -r knowledge_graph_attention_network/Data/yelp2018 {ds_dir}\n"
        )

    return all_ok


def verify_downloads(raw_dir: str) -> None:
    logger.info("\n" + "=" * 65)
    logger.info("KIỂM TRA FILES ĐÃ TẢI")
    logger.info("=" * 65)

    checks = {
        "amazon-book": {
            "train.txt":          (100_000,  "CF interactions train"),
            "test.txt":           (50_000,   "CF interactions test"),
            "item_list.txt":      (10_000,   "Item ID mapping"),
            "entity_list.txt":    (10_000,   "Entity ID mapping"),
            "relation_list.txt":  (100,      "Relation mapping"),
            "user_list.txt":      (10_000,   "User ID mapping"),
            "kg_final.txt":       (100_000,  "KG triples (đã giải nén)"),
        },
        "yelp2018": {
            "train.txt":          (100_000,  "CF interactions train"),
            "test.txt":           (50_000,   "CF interactions test"),
            "item_list.txt":      (10_000,   "Item ID mapping"),
            "entity_list.txt":    (10_000,   "Entity ID mapping"),
            "kg_final.txt":       (100_000,  "KG triples (đã giải nén)"),
        },
    }

    all_ready = True
    for dataset, files in checks.items():
        logger.info(f"\n  {dataset}:")
        for fname, (min_bytes, desc) in files.items():
            fpath = os.path.join(raw_dir, dataset, fname)
            if _check_file(fpath, min_bytes):
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                logger.info(f"    ✓ {fname:<30} ({size_mb:.1f} MB) — {desc}")
            else:
                logger.warning(f"    ✗ {fname:<30} THIẾU — {desc}")
                all_ready = False

    logger.info("\n" + "=" * 65)
    if all_ready:
        logger.info("  ✓ Tất cả files đã sẵn sàng. Chạy preprocess.py tiếp theo.")
    else:
        logger.warning("  ✗ Một số files còn thiếu.")
    logger.info("=" * 65)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download dữ liệu cho KG-LightGCN v10\n"
            "Nguồn DUY NHẤT: KGAT repo (xiangwang1223)\n"
            "KHÔNG dùng LightGCN/SimGCL/KGCL repo"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["amazon-book", "yelp2018", "all"],
        default="all",
    )
    p.add_argument(
        "--raw_dir",
        default="/data/phuongtran/project_v10/raw",
        help="Thư mục lưu raw data",
    )
    p.add_argument(
        "--check_only",
        action="store_true",
        help="Chỉ kiểm tra, không tải",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    action = "Kiểm tra" if args.check_only else "Tải về"
    logger.info(f"{action} dữ liệu vào: {args.raw_dir}")
    logger.info("NGUỒN DUY NHẤT: KGAT repo (xiangwang1223)")
    logger.info("KHÔNG dùng LightGCN/SimGCL/KGCL repo — v10 requirement")

    if args.dataset in ("amazon-book", "all"):
        download_amazon_book(args.raw_dir, check_only=args.check_only)

    if args.dataset in ("yelp2018", "all"):
        download_yelp2018(args.raw_dir, check_only=args.check_only)

    verify_downloads(args.raw_dir)

    if not args.check_only:
        logger.info(
            f"\nBước tiếp theo: python scripts/preprocess.py "
            f"--raw_dir {args.raw_dir} "
            f"--out_dir {args.raw_dir.replace('raw', 'unified')}"
        )


if __name__ == "__main__":
    main()
