"""
models/lightgcn.py — v10
LightGCN: He et al., SIGIR 2020 — https://arxiv.org/abs/2002.02126
Đọc dữ liệu từ unified/ (KGAT repo format).
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from models.base_model import BaseModel


class LightGCN(BaseModel):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers: int = 3, norm_adj: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_layers       = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None
        self._init_weights()

    def _graph_propagation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        E0   = torch.cat([self.user_embedding.weight,
                          self.item_embedding.weight], dim=0)
        E_k  = E0
        acc  = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, pos_items, neg_items):
        user_final, item_final = self._graph_propagation()
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self):
        return self._graph_propagation()

    def l2_loss(self, users, pos_items, neg_items):
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
                n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        self.register_buffer("norm_adj", norm_adj)
