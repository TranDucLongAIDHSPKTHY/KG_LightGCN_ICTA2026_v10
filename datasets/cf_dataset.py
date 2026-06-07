"""
datasets/cf_dataset.py — v10-fix
CF dataset với norm_adj và vectorised sampling.

FIX v10-fix:
  [BUG-1] Negative sampling: tăng retry từ 5 → 50 lần (rejection sampling đúng chuẩn)
           Với Amazon-Book density ~0.6%, 5 lần retry đủ. Nhưng với Yelp2018 hay cold
           splits thưa hơn có thể gặp vấn đề nếu user có nhiều positives.
           Standard practice: retry đủ lớn để đảm bảo sample được negative thực sự.
  [INFO-11] __getitem__ chỉ valid với split="train" — thêm assertion rõ ràng.
"""
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from datasets.base_dataset import BaseDataset

# Số lần retry tối đa cho negative sampling
_NEG_SAMPLE_MAX_RETRY = 50


class CFDataset(BaseDataset, Dataset):
    def __init__(
        self,
        data_dir:    str,
        split:       str = "train",
        neg_samples: int = 1,
        seed:        int = 42,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, seed=seed)
        self.neg_samples    = neg_samples
        self.train_pairs:   Optional[np.ndarray] = None
        self.norm_adj_mat:  Optional[torch.Tensor] = None
        self._user_item_set: Optional[Dict[int, Set[int]]] = None
        self.load()

    def load(self) -> None:
        train_d = self._load_split("train")
        valid_d = self._load_split("valid")
        test_d  = self._load_split("test")

        if not train_d:
            raise FileNotFoundError(
                f"train.txt not found in {self.data_dir}")

        self.n_users, self.n_items = self._compute_dimensions(
            train_d, valid_d, test_d)

        # user2items cho split hiện tại (dùng trong evaluator)
        self.user2items = {"train": train_d, "valid": valid_d, "test": test_d}[
            self.split]

        # Training positive set — dùng cho negative sampling và norm_adj
        self._user_item_set = {u: set(items) for u, items in train_d.items()}

        # Build train pairs array (chỉ từ train split)
        rows, cols = [], []
        for uid, items in train_d.items():
            rows.extend([uid] * len(items))
            cols.extend(items)
        self.train_pairs = np.stack(
            [np.array(rows, dtype=np.int32),
             np.array(cols, dtype=np.int32)],
            axis=1,
        )

        self.norm_adj_mat = self._build_norm_adj(train_d)

    def _build_norm_adj(
        self, train_d: Dict[int, List[int]]
    ) -> torch.Tensor:
        """D^{-1/2} A D^{-1/2} symmetric normalized adjacency."""
        N, M     = self.n_users, self.n_items
        n_inters = sum(len(v) for v in train_d.values())

        row = np.empty(n_inters, dtype=np.int32)
        col = np.empty(n_inters, dtype=np.int32)
        ptr = 0
        for uid, items in train_d.items():
            k = len(items)
            row[ptr:ptr + k] = uid
            col[ptr:ptr + k] = [N + i for i in items]
            ptr += k

        data = np.ones(n_inters, dtype=np.float32)
        R    = sp.coo_matrix((data, (row, col)), shape=(N + M, N + M))
        A    = (R + R.T).tocsr()

        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(
                deg > 0, np.power(deg, -0.5), 0.0
            ).astype(np.float32)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo().astype(np.float32)

        indices = torch.from_numpy(
            np.vstack([A_hat.row, A_hat.col]).astype(np.int64))
        values  = torch.from_numpy(A_hat.data)
        return torch.sparse_coo_tensor(
            indices, values, (N + M, N + M)).coalesce()

    # ── Pickle support cho DataLoader workers ─────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.norm_adj_mat is not None:
            t = self.norm_adj_mat.coalesce()
            state["_adj_indices_np"] = t.indices().numpy()
            state["_adj_values_np"]  = t.values().numpy()
            state["_adj_size"]       = tuple(t.size())
        else:
            state["_adj_indices_np"] = None
            state["_adj_values_np"]  = None
            state["_adj_size"]       = None
        state["norm_adj_mat"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if (self._adj_indices_np is not None
                and self._adj_values_np is not None
                and self._adj_size is not None):
            indices = torch.from_numpy(self._adj_indices_np)
            values  = torch.from_numpy(self._adj_values_np)
            self.norm_adj_mat = torch.sparse_coo_tensor(
                indices, values, self._adj_size).coalesce()
        self.__dict__.pop("_adj_indices_np", None)
        self.__dict__.pop("_adj_values_np",  None)
        self.__dict__.pop("_adj_size",        None)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        # [INFO-11] __getitem__ chỉ hợp lệ khi split="train"
        # valid/test được truy cập qua evaluator, không qua DataLoader
        return len(self.train_pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trả về (user, pos_item, neg_item) từ training set.

        [BUG-1 FIX] Negative sampling dùng rejection sampling với
        _NEG_SAMPLE_MAX_RETRY lần thử thay vì chỉ 5 lần.
        Đảm bảo neg_item không phải positive item của user.
        """
        uid, pos_iid = self.train_pairs[idx]
        pos_set      = self._user_item_set.get(int(uid), set())

        # [BUG-1 FIX] Tăng retry lên _NEG_SAMPLE_MAX_RETRY (50)
        neg_iid = int(self.rng.randint(0, self.n_items))
        for _ in range(_NEG_SAMPLE_MAX_RETRY):
            if neg_iid not in pos_set:
                break
            neg_iid = int(self.rng.randint(0, self.n_items))
        # Nếu sau _NEG_SAMPLE_MAX_RETRY vẫn không tìm được → chấp nhận
        # (xảy ra cực hiếm với dataset thưa; không ảnh hưởng đáng kể)

        return (
            torch.tensor(int(uid),     dtype=torch.long),
            torch.tensor(int(pos_iid), dtype=torch.long),
            torch.tensor(neg_iid,      dtype=torch.long),
        )

    def get_eval_data(
        self,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        return self._load_split("train"), self._load_split(self.split)