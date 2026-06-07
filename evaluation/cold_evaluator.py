"""
evaluation/cold_evaluator.py — v10-fix
Cold-start evaluator theo induced cold-start protocol (T3.1).

FIX v10-fix:
  [PERF-9] Vectorize mask loop trong cold_start_eval.
           Trước: Python loop O(B × n_cold) để set -inf cho seen items.
           Sau:   Vectorized scatter bằng index_fill_ → O(B + seen_items).
"""
import os
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from evaluation.metrics import compute_all_metrics
from utils.logger import get_logger

logger = get_logger("cold_evaluator")


def load_cold_item_ids(cold_dir: str) -> Set[int]:
    """Load cold item IDs từ cold_item_ids.txt."""
    path = os.path.join(cold_dir, "cold_item_ids.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"cold_item_ids.txt không tìm thấy trong {cold_dir}. "
            "Chạy scripts/build_cold_split.py trước."
        )
    with open(path, "r") as f:
        items = {int(line.strip()) for line in f if line.strip().isdigit()}
    logger.info(f"Loaded {len(items):,} cold items từ {cold_dir}")
    return items


def _read_interaction_file(path: str) -> Dict[int, List[int]]:
    user2items: Dict[int, List[int]] = {}
    if not os.path.exists(path):
        return user2items
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                user2items[int(parts[0])] = [int(x) for x in parts[1:]]
            except ValueError:
                continue
    return user2items


@torch.no_grad()
def cold_start_eval(
    model,
    train_user2items: Dict[int, List[int]],
    test_user2items:  Dict[int, List[int]],
    cold_items:       Set[int],
    n_items:          int,
    device:           torch.device,
    batch_size:       int = 512,
    top_k_list:       Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Cold-start evaluation: candidate pool = cold_items ONLY.
    Metrics trả về với hậu tố '_cold'.

    [PERF-9 FIX] Mask train-seen items bằng vectorized scatter thay vì
    Python loop O(B × n_cold).
    """
    if top_k_list is None:
        top_k_list = [10, 20]
    model.eval()

    # Filter test data để chỉ giữ cold interactions
    cold_test: Dict[int, List[int]] = {}
    for uid, items in test_user2items.items():
        cold_gt = [i for i in items if i in cold_items]
        if cold_gt:
            cold_test[uid] = cold_gt

    if not cold_test:
        logger.warning(
            "Không tìm thấy cold test pairs. "
            "Kiểm tra cold split đã được build đúng chưa."
        )
        return {}

    n_users_cold   = len(cold_test)
    n_pairs_cold   = sum(len(v) for v in cold_test.values())
    cold_item_list = sorted(cold_items)       # list cố định để index
    n_cold         = len(cold_item_list)
    max_k          = max(top_k_list)

    logger.info(
        f"Cold eval: {n_users_cold:,} users | "
        f"{n_pairs_cold:,} cold pairs | "
        f"{n_cold:,} cold items (candidate pool)"
    )

    if n_cold <= max_k:
        logger.warning(
            f"TRIVIAL RANKING: n_cold={n_cold} <= max_k={max_k}. "
            "Metrics có thể quá cao."
        )

    # Build reverse lookup: cold_item_id → local index trong cold_item_list
    cold_item_to_local: Dict[int, int] = {
        item_id: local_idx
        for local_idx, item_id in enumerate(cold_item_list)
    }

    user_emb, item_emb = model.get_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    # Pre-build cold item embedding matrix (fixed, reuse across batches)
    cold_item_tensor = torch.tensor(cold_item_list, dtype=torch.long, device=device)
    cold_emb         = item_emb[cold_item_tensor]   # (n_cold, d)

    eval_users   = sorted(cold_test.keys())
    all_ranked:  List[np.ndarray] = []
    all_gt:      List[List[int]]  = []

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start: start + batch_size]
        B           = len(batch_users)

        u_emb  = user_emb[batch_users]                 # (B, d)
        scores = torch.matmul(u_emb, cold_emb.T)       # (B, n_cold)

        # [PERF-9 FIX] Vectorized mask — O(B + total_seen) thay vì O(B × n_cold)
        #
        # Bước 1: Với mỗi user trong batch, tìm cold items đã seen trong train
        # Bước 2: Dùng advanced indexing để set -inf một lần
        #
        row_indices = []   # list of local user indices (0..B-1)
        col_indices = []   # list of local cold item indices

        for local_i, uid in enumerate(batch_users):
            train_seen = train_user2items.get(uid, [])
            for item_id in train_seen:
                local_j = cold_item_to_local.get(item_id)
                if local_j is not None:
                    row_indices.append(local_i)
                    col_indices.append(local_j)

        if row_indices:
            ri = torch.tensor(row_indices, dtype=torch.long, device=device)
            ci = torch.tensor(col_indices, dtype=torch.long, device=device)
            scores[ri, ci] = float("-inf")

        # Top-K ranking
        k_eff = min(max_k, n_cold)
        _, ranked_local = torch.topk(scores, k=k_eff, dim=-1)
        # Map local indices back to global item IDs
        ranked_global = cold_item_tensor[ranked_local.cpu()].numpy()

        for local_i, uid in enumerate(batch_users):
            all_ranked.append(ranked_global[local_i])
            all_gt.append(cold_test[uid])

    ranked_matrix = np.vstack(all_ranked)
    metrics_raw   = compute_all_metrics(ranked_matrix, all_gt, top_k_list)

    # Thêm hậu tố '_cold' để phân biệt với warm metrics
    cold_metrics = {f"{k}_cold": v for k, v in metrics_raw.items()}

    logger.info("Cold metrics: " + " | ".join(
        f"{k}={v:.4f}" for k, v in sorted(cold_metrics.items())
    ))
    return cold_metrics


class ColdEvaluator:
    """
    Wrapper cho cold-start evaluation (v10 format).

    Args:
        cold_dir:       path đến cold_XX/ directory
                        Chứa: cold_item_ids.txt, test_cold.txt
        train_data_dir: path đến unified/ directory (dùng train.txt để mask)
        n_items:        tổng số items
        device:         torch device
    """

    def __init__(
        self,
        cold_dir:       str,
        train_data_dir: str,
        n_items:        int,
        device:         torch.device,
        batch_size:     int = 512,
        top_k_list:     Optional[List[int]] = None,
    ) -> None:
        self.cold_items = load_cold_item_ids(cold_dir)

        self.train_user2items = _read_interaction_file(
            os.path.join(train_data_dir, "train.txt"))

        test_cold_path = os.path.join(cold_dir, "test_cold.txt")
        if not os.path.exists(test_cold_path):
            raise FileNotFoundError(
                f"test_cold.txt không tìm thấy trong {cold_dir}. "
                "Chạy scripts/build_cold_split.py trước."
            )
        self.test_user2items = _read_interaction_file(test_cold_path)

        self.n_items    = n_items
        self.device     = device
        self.batch_size = batch_size
        self.top_k_list = top_k_list or [10, 20]

        logger.info(
            f"ColdEvaluator (v10-fix): {len(self.cold_items):,} cold items | "
            f"{len(self.test_user2items):,} test users"
        )

    def evaluate(self, model) -> Dict[str, float]:
        return cold_start_eval(
            model            = model,
            train_user2items = self.train_user2items,
            test_user2items  = self.test_user2items,
            cold_items       = self.cold_items,
            n_items          = self.n_items,
            device           = self.device,
            batch_size       = self.batch_size,
            top_k_list       = self.top_k_list,
        )