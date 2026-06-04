"""models/base_model.py — Abstract base for all recommender models (v10)."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim
        self.device        = device or torch.device("cpu")

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def predict(self, users: torch.Tensor,
                items: Optional[torch.Tensor] = None) -> torch.Tensor:
        user_emb, item_emb = self.get_embeddings()
        u = user_emb[users]
        if items is None:
            return torch.matmul(u, item_emb.T)
        elif items.dim() == 1:
            return torch.matmul(u, item_emb[items].T)
        else:
            return torch.bmm(
                u.unsqueeze(1), item_emb[items].transpose(1, 2)).squeeze(1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n_users={self.n_users}, "
                f"n_items={self.n_items}, emb_dim={self.embedding_dim}, "
                f"params={self.parameter_count():,})")
