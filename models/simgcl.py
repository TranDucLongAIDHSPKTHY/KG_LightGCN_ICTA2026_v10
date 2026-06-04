"""
models/simgcl.py — v10
SimGCL: Yu et al., SIGIR 2022 — https://arxiv.org/abs/2112.08679
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class SimGCL(BaseModel):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers: int = 3, eps: float = 0.1, temperature: float = 0.2,
                 lambda_cl: float = 0.5, apply_item_cl: bool = True,
                 norm_adj: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_layers      = n_layers
        self.eps           = eps
        self.temperature   = temperature
        self.lambda_cl     = lambda_cl
        self.apply_item_cl = apply_item_cl
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None
        self._init_weights()

    def _propagate(self, perturb: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        E0   = torch.cat([self.user_embedding.weight,
                          self.item_embedding.weight], dim=0)
        E_k  = E0
        acc  = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            if perturb:
                noise = (torch.rand_like(E_k) * 2 - 1) * self.eps
                E_k   = E_k + noise
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, pos_items, neg_items) -> Tuple[torch.Tensor, ...]:
        user_emb, item_emb = self._propagate(perturb=False)
        u1, i1 = self._propagate(perturb=True)
        u2, i2 = self._propagate(perturb=True)
        if self.apply_item_cl:
            return (user_emb[users], item_emb[pos_items], item_emb[neg_items],
                    u1[users], u2[users], i1[pos_items], i2[pos_items])
        return (user_emb[users], item_emb[pos_items], item_emb[neg_items],
                u1[users], u2[users])

    def get_embeddings(self):
        return self._propagate(perturb=False)

    def contrastive_loss(self, view1, view2) -> torch.Tensor:
        v1  = F.normalize(view1, dim=-1)
        v2  = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.temperature
        labels = torch.arange(len(v1), device=v1.device)
        return (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)) / 2

    def l2_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
                n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        self.register_buffer("norm_adj", norm_adj)
