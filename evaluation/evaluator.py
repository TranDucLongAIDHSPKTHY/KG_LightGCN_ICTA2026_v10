"""evaluation/evaluator.py — v10. Unified Evaluator.
THAY ĐỔI v10: dùng valid.txt thay vì val.txt.
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
        valid_user2items: Dict[int, List[int]],   # [v10] valid
        test_user2items:  Dict[int, List[int]],
        n_items:          int,
        device:           torch.device,
        batch_size:       int = 2048,
        top_k_list:       Optional[List[int]] = None,
    ):
        self.train_user2items = train_user2items
        self.valid_user2items = valid_user2items  # [v10]
        self.test_user2items  = test_user2items
        self.n_items          = n_items
        self.device           = device
        self.batch_size       = batch_size
        self.top_k_list       = top_k_list or [10, 20]

    def evaluate(self, model, split: str = "valid") -> Dict[str, float]:
        # [v10] split: "valid" thay cho "val"
        assert split in ("valid", "test"), \
            f"split phải là 'valid' hoặc 'test' (không phải '{split}')"
        eval_map = (self.valid_user2items
                    if split == "valid" else self.test_user2items)
        if not eval_map:
            logger.warning(f"Không có eval data cho split '{split}'.")
            return {}
        return full_ranking_eval(
            model            = model,
            train_user2items = self.train_user2items,
            eval_user2items  = eval_map,
            n_items          = self.n_items,
            device           = self.device,
            batch_size       = self.batch_size,
            top_k_list       = self.top_k_list,
        )

    def log_metrics(self, metrics, split, epoch=None):
        prefix = f"[{split.upper()}]" + (
            f" epoch={epoch}" if epoch is not None else "")
        msg = prefix + "  " + "  ".join(
            f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
        logger.info(msg)
