"""
datasets/base_dataset.py — v10
Abstract base dataset.

THAY ĐỔI v10:
  - Split file: valid.txt (không còn val.txt)
  - Đọc từ unified/ (KGAT repo format)
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC):
    def __init__(self, data_dir: str, split: str = "train",
                 seed: int = 42) -> None:
        # [v10] valid thay cho val
        assert split in ("train", "valid", "test"), \
            f"split phải là 'train', 'valid' hoặc 'test' (không phải '{split}')"
        self.data_dir  = data_dir
        self.split     = split
        self.seed      = seed
        self.rng       = np.random.RandomState(seed)
        self.n_users:  int = 0
        self.n_items:  int = 0
        self.user2items: Dict[int, List[int]] = {}
        self.all_train_items: Set[int] = set()

    @staticmethod
    def read_interaction_file(path: str) -> Dict[int, List[int]]:
        """Đọc file tương tác (KGAT format: user item1 item2 ...)."""
        user2items: Dict[int, List[int]] = {}
        if not os.path.exists(path):
            return user2items
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                user2items[uid] = [int(x) for x in parts[1:]]
        return user2items

    def _load_split(self, split_name: str) -> Dict[int, List[int]]:
        # [v10] Ánh xạ 'valid' → 'valid.txt'
        fname = f"{split_name}.txt"
        return self.read_interaction_file(
            os.path.join(self.data_dir, fname))

    @property
    def n_interactions(self) -> int:
        return sum(len(v) for v in self.user2items.values())

    def _compute_dimensions(self, *splits) -> Tuple[int, int]:
        all_users, all_items = set(), set()
        for d in splits:
            for u, items in d.items():
                all_users.add(u)
                all_items.update(items)
        if not all_users or not all_items:
            return 0, 0
        return max(all_users) + 1, max(all_items) + 1

    def sample_negative(self, uid: int, n_neg: int = 1) -> List[int]:
        user_positives = set(self.user2items.get(uid, []))
        negatives: List[int] = []
        while len(negatives) < n_neg:
            neg = self.rng.randint(0, self.n_items)
            if neg not in user_positives:
                negatives.append(neg)
        return negatives

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int): ...
