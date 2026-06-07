"""
evaluation/evaluator.py — v10-fix
Unified Evaluator.

FIX v10-fix:
  [BUG-3] Khi evaluate test split, mask CẢ train VÀ valid items (đã seen).
           Trước đây chỉ mask train → valid items có thể xuất hiện trong test
           ranking → test metrics bị inflate nhẹ.
           Fix: truyền combined_exclude = train ∪ valid vào full_ranking_eval.
"""
from typing import Dict, List, Optional

import torch

from evaluation.full_ranking import full_ranking_eval
from utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    def __init__(
        self,
        train_user2items: Dict[int, List[int]],
        valid_user2items: Dict[int, List[int]],
        test_user2items:  Dict[int, List[int]],
        n_items:          int,
        device:           torch.device,
        batch_size:       int = 2048,
        top_k_list:       Optional[List[int]] = None,
    ):
        self.train_user2items = train_user2items
        self.valid_user2items = valid_user2items
        self.test_user2items  = test_user2items
        self.n_items          = n_items
        self.device           = device
        self.batch_size       = batch_size
        self.top_k_list       = top_k_list or [10, 20]

        # [BUG-3 FIX] Pre-compute combined exclude set cho test evaluation:
        # mask = train ∪ valid (tất cả items đã seen trước test)
        self._test_exclude = self._merge_exclude(
            train_user2items, valid_user2items)

    @staticmethod
    def _merge_exclude(
        d1: Dict[int, List[int]],
        d2: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        """
        Gộp 2 user→items dicts thành 1 để dùng làm mask.
        Kết quả: {uid: sorted(d1[uid] ∪ d2[uid])}
        """
        all_users = set(d1) | set(d2)
        merged: Dict[int, List[int]] = {}
        for uid in all_users:
            s = set(d1.get(uid, [])) | set(d2.get(uid, []))
            if s:
                merged[uid] = sorted(s)
        return merged

    def evaluate(self, model, split: str = "valid") -> Dict[str, float]:
        """
        Full-item ranking evaluation.

        Args:
            split: "valid" — mask train items
                   "test"  — mask train ∪ valid items [BUG-3 FIX]
        """
        assert split in ("valid", "test"), (
            f"split phải là 'valid' hoặc 'test' (nhận được '{split}')")

        if split == "valid":
            eval_map     = self.valid_user2items
            exclude_map  = self.train_user2items   # chỉ mask train
        else:
            eval_map     = self.test_user2items
            exclude_map  = self._test_exclude      # [BUG-3 FIX] mask train ∪ valid

        if not eval_map:
            logger.warning(f"Không có eval data cho split='{split}'.")
            return {}

        return full_ranking_eval(
            model            = model,
            train_user2items = exclude_map,        # tên param giữ nguyên để tương thích
            eval_user2items  = eval_map,
            n_items          = self.n_items,
            device           = self.device,
            batch_size       = self.batch_size,
            top_k_list       = self.top_k_list,
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        split:   str,
        epoch:   Optional[int] = None,
    ) -> None:
        prefix = f"[{split.upper()}]"
        if epoch is not None:
            prefix += f" epoch={epoch}"
        msg = prefix + "  " + "  ".join(
            f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
        logger.info(msg)