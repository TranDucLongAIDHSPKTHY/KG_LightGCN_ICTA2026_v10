"""
datasets/cf_dataset.py — v10
CF dataset với norm_adj và vectorised sampling.

THAY ĐỔI v10:
  - Dùng valid.txt thay vì val.txt
  - Đọc từ unified/ (KGAT repo format)
  - Split: "valid" thay vì "val"
"""
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset


class CFDataset(BaseDataset, Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 neg_samples: int = 1, seed: int = 42) -> None:
        # [v10] split có thể là "train", "valid", "test"
        super().__init__(data_dir=data_dir, split=split, seed=seed)
        self.neg_samples    = neg_samples
        self.train_pairs:   Optional[np.ndarray] = None
        self.norm_adj_mat:  Optional[torch.Tensor] = None
        self._user_item_set: Optional[Dict[int, Set[int]]] = None
        self.load()

    def load(self) -> None:
        train_d = self._load_split("train")
        valid_d = self._load_split("valid")   # [v10] valid.txt
        test_d  = self._load_split("test")
        if not train_d:
            raise FileNotFoundError(
                f"train.txt not found in {self.data_dir}")
        self.n_users, self.n_items = self._compute_dimensions(
            train_d, valid_d, test_d)
        self.user2items = {
            "train": train_d, "valid": valid_d, "test": test_d,
        }[self.split]
        self._user_item_set = {
            u: set(items) for u, items in train_d.items()}
        rows, cols = [], []
        for uid, items in train_d.items():
            rows.extend([uid] * len(items))
            cols.extend(items)
        self.train_pairs = np.stack(
            [np.array(rows, dtype=np.int32),
             np.array(cols, dtype=np.int32)], axis=1)
        self.norm_adj_mat = self._build_norm_adj(train_d)

    def _build_norm_adj(
        self, train_d: Dict[int, List[int]]
    ) -> torch.Tensor:
        """Xây dựng D^{-1/2}AD^{-1/2} norm adj cho CF propagation."""
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
        deg  = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(
                deg > 0, np.power(deg, -0.5), 0.0).astype(np.float32)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack([A_hat.row, A_hat.col]).astype(np.int64))
        values  = torch.from_numpy(A_hat.data)
        return torch.sparse_coo_tensor(
            indices, values, (N + M, N + M)).coalesce()

    # Pickle support cho DataLoader workers
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
        if (self._adj_indices_np is not None and
                self._adj_values_np is not None and
                self._adj_size is not None):
            indices = torch.from_numpy(self._adj_indices_np)
            values  = torch.from_numpy(self._adj_values_np)
            self.norm_adj_mat = torch.sparse_coo_tensor(
                indices, values, self._adj_size).coalesce()
        self.__dict__.pop("_adj_indices_np", None)
        self.__dict__.pop("_adj_values_np",  None)
        self.__dict__.pop("_adj_size",        None)

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_pairs)
        return sum(len(v) for v in self.user2items.values())

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid, pos_iid = self.train_pairs[idx]
        pos_set = self._user_item_set.get(int(uid), set())
        neg_iid = int(self.rng.randint(0, self.n_items))
        for _ in range(5):
            if neg_iid not in pos_set:
                break
            neg_iid = int(self.rng.randint(0, self.n_items))
        return (torch.tensor(int(uid),     dtype=torch.long),
                torch.tensor(int(pos_iid), dtype=torch.long),
                torch.tensor(neg_iid,      dtype=torch.long))

    def get_eval_data(
        self,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        return self._load_split("train"), self._load_split(self.split)
