"""datasets/dataloader.py — DataLoader factory functions (v10).

THAY ĐỔI v10: split "valid" thay cho "val".
"""
import torch
from torch.utils.data import DataLoader
from datasets.cf_dataset import CFDataset
from datasets.kg_dataset import KGDataset


def worker_init_fn(worker_id: int) -> None:
    import numpy as np
    np.random.seed(torch.initial_seed() % 2**32)


def get_cf_dataloader(
    data_dir:    str,
    split:       str = "train",
    batch_size:  int = 2048,
    neg_samples: int = 1,
    seed:        int = 42,
    num_workers: int = 0,
    shuffle:     bool = True,
) -> DataLoader:
    dataset = CFDataset(
        data_dir=data_dir, split=split,
        neg_samples=neg_samples, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (shuffle and split == "train"),
        num_workers = num_workers,
        pin_memory  = (torch.cuda.is_available() and num_workers > 0),
        worker_init_fn    = worker_init_fn if num_workers > 0 else None,
        persistent_workers = (num_workers > 0),
        drop_last   = False,
    )


def get_kg_dataloader(
    data_dir:    str,
    split:       str = "train",
    batch_size:  int = 2048,
    neg_samples: int = 1,
    kg_type:     str = "full",
    seed:        int = 42,
    num_workers: int = 0,
    shuffle:     bool = True,
) -> DataLoader:
    dataset = KGDataset(
        data_dir=data_dir, split=split,
        neg_samples=neg_samples, kg_type=kg_type, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (shuffle and split == "train"),
        num_workers = num_workers,
        pin_memory  = (torch.cuda.is_available() and num_workers > 0),
        worker_init_fn    = worker_init_fn if num_workers > 0 else None,
        persistent_workers = (num_workers > 0),
        drop_last   = False,
    )
