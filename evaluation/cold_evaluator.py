"""
evaluation/cold_evaluator.py — v10
Cold-start evaluator theo induced cold-start protocol (T3.1).

THAY ĐỔI v10 so với v7:
  - cold_items.txt    → cold_item_ids.txt
  - test.txt (cold)   → test_cold.txt
  - valid.txt         thay cho val.txt
  - Paths: /data/phuongtran/project_v10/unified/amazon-book/cold_XX/

Protocol:
  cold_item_ids.txt   = danh sách cold item IDs (seed=42, tương thích TaxPro-CL)
  test_cold.txt       = chỉ chứa interactions của cold_items (từ original test.txt)
  train.txt (unified) = dùng để mask training interactions

Evaluation: candidate pool = cold_items ONLY.
Metrics: HR@K_cold, NDCG@K_cold, Recall@K_cold.
"""
import os
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from evaluation.metrics import compute_all_metrics
from utils.logger import get_logger

logger = get_logger("cold_evaluator")


def load_cold_item_ids(cold_dir: str) -> Set[int]:
    """[v10] Load cold item IDs từ cold_item_ids.txt (không phải cold_items.txt)."""
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
    Cold-start evaluation: chỉ score cold_items làm candidates.
    Trả về metrics với hậu tố '_cold'.
    """
    if top_k_list is None:
        top_k_list = [10, 20]
    model.eval()

    # test_cold.txt đã filter cold interactions
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
    cold_item_list = sorted(cold_items)
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

    user_emb, item_emb = model.get_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    cold_item_tensor = torch.tensor(cold_item_list, dtype=torch.long, device=device)
    eval_users       = sorted(cold_test.keys())

    all_ranked: List[np.ndarray] = []
    all_gt:     List[List[int]]  = []

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start: start + batch_size]
        u_emb       = user_emb[batch_users]
        cold_emb    = item_emb[cold_item_tensor]
        scores      = torch.matmul(u_emb, cold_emb.T)

        # Mask cold items đã seen trong train
        for local_i, uid in enumerate(batch_users):
            train_seen = set(train_user2items.get(uid, []))
            for j, cold_iid in enumerate(cold_item_list):
                if cold_iid in train_seen:
                    scores[local_i, j] = float("-inf")

        k_eff = min(max_k, n_cold)
        _, ranked_local = torch.topk(scores, k=k_eff, dim=-1)
        ranked_global   = cold_item_tensor.cpu()[ranked_local.cpu()].numpy()

        for local_i, uid in enumerate(batch_users):
            all_ranked.append(ranked_global[local_i])
            all_gt.append(cold_test[uid])

    ranked_matrix = np.vstack(all_ranked)
    metrics_raw   = compute_all_metrics(ranked_matrix, all_gt, top_k_list)

    # [v10] Thêm hậu tố '_cold' để phân biệt với warm metrics
    cold_metrics = {f"{k}_cold": v for k, v in metrics_raw.items()}

    logger.info("Cold metrics: " + " | ".join(
        f"{k}={v:.4f}" for k, v in sorted(cold_metrics.items())
    ))
    return cold_metrics


class ColdEvaluator:
    """
    [v10] Wrapper cho cold-start evaluation.

    Args:
        cold_dir:       path đến cold_XX/ directory
                        Chứa: cold_item_ids.txt, test_cold.txt
        train_data_dir: path đến unified/ directory
                        Dùng train.txt để mask training interactions
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
        # [v10] cold_item_ids.txt (không phải cold_items.txt)
        self.cold_items = load_cold_item_ids(cold_dir)

        # Train gốc để mask
        self.train_user2items = _read_interaction_file(
            os.path.join(train_data_dir, "train.txt"))

        # [v10] test_cold.txt (không phải test.txt)
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
            f"ColdEvaluator (v10): {len(self.cold_items):,} cold items | "
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
