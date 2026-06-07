"""
evaluation/metrics.py — v10-fix
Vectorised metrics: Recall@K, NDCG@K, HR@K.

FIX v10-fix:
  [WARNING-6] RECALL DENOMINATOR — làm rõ convention và đặt default đúng.

  Hai convention phổ biến:
    (A) Recall@K = hits / |GT|           → RECALL_DENOM_GT_ONLY = True
        Dùng bởi: LightGCN paper, KGAT paper, hầu hết RecSys papers gốc.
        Ý nghĩa: "trong tất cả GT items, bao nhiêu % được recall trong top-K"

    (B) Recall@K = hits / min(|GT|, K)   → RECALL_DENOM_GT_ONLY = False
        Dùng bởi: một số implementations (gần với Precision@K hơn)
        Ý nghĩa: "trong tất cả GT items CÓ THỂ recall được, bao nhiêu % được recall"

  QUYẾT ĐỊNH v10-fix: đổi sang RECALL_DENOM_GT_ONLY = True (convention chuẩn)
  để khớp với LightGCN paper (He et al., 2020), KGAT paper (Wang et al., 2019),
  KGCL paper (Yang et al., 2022).

  LƯU Ý: Thay đổi này ảnh hưởng đến số liệu tuyệt đối (absolute values),
  nhưng KHÔNG ảnh hưởng đến ranking tương đối giữa các model.
  Cần chạy lại tất cả experiments sau khi đổi convention.
"""
import numpy as np
from typing import Dict, List

# [WARNING-6 FIX] Đổi thành True để dùng |GT| làm denominator
# Khớp với LightGCN (SIGIR 2020), KGAT (KDD 2019), KGCL (SIGIR 2022)
RECALL_DENOM_GT_ONLY: bool = True


def recall_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """
    Recall@K = hits(K) / denominator

    denominator:
      RECALL_DENOM_GT_ONLY=True  → |GT|           (standard paper convention)
      RECALL_DENOM_GT_ONLY=False → min(|GT|, K)   (alternative convention)
    """
    if not ground_truth:
        return 0.0
    top_k = set(ranked_items[:k].tolist())
    gt    = set(ground_truth)
    hits  = len(top_k & gt)
    denom = len(gt) if RECALL_DENOM_GT_ONLY else min(len(gt), k)
    return hits / denom if denom > 0 else 0.0


def ndcg_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """NDCG@K — chuẩn theo paper."""
    if not ground_truth:
        return 0.0
    gt    = set(ground_truth)
    top_k = ranked_items[:k].tolist()
    dcg   = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in gt
    )
    ideal_k = min(len(gt), k)
    idcg    = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """HR@K = 1 nếu có ít nhất 1 GT item trong top-K."""
    if not ground_truth:
        return 0.0
    return 1.0 if set(ranked_items[:k].tolist()) & set(ground_truth) else 0.0


# ── Batch versions (vectorized) ───────────────────────────────────────────────

def batch_recall_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    recalls = np.zeros(len(ground_truths), dtype=np.float32)
    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set   = set(gt)
        actual_k = min(k, len(ranked_matrix[i]))
        hits     = sum(1 for item in ranked_matrix[i][:actual_k] if item in gt_set)
        denom    = len(gt_set) if RECALL_DENOM_GT_ONLY else min(len(gt_set), k)
        recalls[i] = hits / denom if denom > 0 else 0.0
    return recalls


def batch_ndcg_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    ndcgs      = np.zeros(len(ground_truths), dtype=np.float32)
    log2_table = np.log2(np.arange(2, k + 2))
    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set   = set(gt)
        actual_k = min(k, len(ranked_matrix[i]))
        hits     = np.array(
            [1.0 if item in gt_set else 0.0
             for item in ranked_matrix[i][:actual_k]],
            dtype=np.float32,
        )
        dcg      = (hits / log2_table[:actual_k]).sum()
        ideal_k  = min(len(gt_set), actual_k)
        idcg     = (1.0 / log2_table[:ideal_k]).sum()
        ndcgs[i] = dcg / idcg if idcg > 0 else 0.0
    return ndcgs


def batch_hr_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    hrs = np.zeros(len(ground_truths), dtype=np.float32)
    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        actual_k = min(k, len(ranked_matrix[i]))
        hrs[i]   = 1.0 if any(
            item in set(gt) for item in ranked_matrix[i][:actual_k]) else 0.0
    return hrs


def compute_all_metrics(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    top_k_list:    List[int] = None,
) -> Dict[str, float]:
    if top_k_list is None:
        top_k_list = [10, 20]
    results: Dict[str, float] = {}
    for k in top_k_list:
        recalls = batch_recall_at_k(ranked_matrix, ground_truths, k)
        ndcgs   = batch_ndcg_at_k(ranked_matrix, ground_truths, k)
        hrs     = batch_hr_at_k(ranked_matrix, ground_truths, k)
        results[f"recall@{k}"] = float(recalls.mean())
        results[f"ndcg@{k}"]   = float(ndcgs.mean())
        results[f"hr@{k}"]     = float(hrs.mean())
    return results