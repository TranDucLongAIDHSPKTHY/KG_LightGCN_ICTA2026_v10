"""
models/kgcl.py — v10
KGCL: Knowledge Graph Contrastive Learning
Yang et al., SIGIR 2022 — https://arxiv.org/abs/2205.00976
Dữ liệu từ KGAT repo (kg_final.txt). Bất biến item_id==entity_id.
"""
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class KGCL(BaseModel):
    def __init__(
        self, n_users, n_items, n_entities, n_relations,
        embedding_dim=64, n_layers=3, kg_n_layers=2,
        temp=0.2, lambda_kg=0.1, kg_p_drop=0.5, ui_p_drop=0.05,
        norm_adj=None, kg_triples=None, device=None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities  = n_entities
        self.n_relations = n_relations
        self.n_layers    = n_layers
        self.kg_n_layers = kg_n_layers
        self.temp        = temp
        self.lambda_kg   = lambda_kg
        self.kg_p_drop   = kg_p_drop
        self.ui_p_drop   = ui_p_drop

        self.user_embedding     = nn.Embedding(n_users,    embedding_dim)
        self.entity_embedding   = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None
        self.register_buffer("kg_norm_adj", None)

        self.item_kg_degree = None
        if kg_triples is not None:
            self._build_kg_degree(kg_triples, n_entities)

        self._kg_heads = self._kg_tails = self._kg_rels = None
        if kg_triples is not None:
            self._build_kg_edge_tensors(kg_triples)

        self._aug_adj1 = self._aug_adj2 = None
        self._init_weights()

    def _build_kg_degree(self, kg_triples, n_entities):
        degree = np.zeros(n_entities, dtype=np.float32)
        for h, _, t in kg_triples:
            if h < n_entities: degree[h] += 1
            if t < n_entities: degree[t] += 1
        max_deg = degree.max()
        if max_deg > 0:
            degree /= max_deg
        self.item_kg_degree = torch.tensor(degree[:self.n_items], dtype=torch.float32)

    def _build_kg_edge_tensors(self, kg_triples):
        self._kg_heads = torch.tensor(kg_triples[:, 0], dtype=torch.long)
        self._kg_tails = torch.tensor(kg_triples[:, 2], dtype=torch.long)
        self._kg_rels  = torch.tensor(kg_triples[:, 1], dtype=torch.long)

    def _kg_propagation(self):
        E = self.entity_embedding.weight
        if self._kg_heads is not None:
            device = E.device
            heads = self._kg_heads.to(device)
            tails = self._kg_tails.to(device)
            rels  = self._kg_rels.to(device)
            E_k   = E
            for _ in range(self.kg_n_layers):
                t_emb = E_k[tails]
                r_emb = self.relation_embedding(rels)
                msgs  = t_emb * r_emb
                agg   = torch.zeros_like(E_k)
                cnt   = torch.zeros(self.n_entities, device=device)
                agg.scatter_add_(0, heads.unsqueeze(1).expand_as(msgs), msgs)
                cnt.scatter_add_(0, heads, torch.ones(len(heads), device=device))
                cnt = cnt.clamp(min=1).unsqueeze(1)
                E_k = F.normalize(E_k + agg / cnt, dim=-1)
            return E_k
        if self.kg_norm_adj is not None:
            _dev = E.device
            adj  = self.kg_norm_adj.to(_dev)
            E_k  = E; acc = E.clone()
            for _ in range(self.kg_n_layers):
                E_k = torch.sparse.mm(adj, E_k)
                acc = acc + E_k
            return acc / (self.kg_n_layers + 1)
        return E

    def _cf_propagation(self, adj, entity_emb):
        item_e = entity_emb[:self.n_items]
        E0     = torch.cat([self.user_embedding.weight, item_e], dim=0)
        E_k    = E0; acc = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def _augment_adj(self, adj):
        if self.item_kg_degree is None:
            return self._random_edge_drop(adj, drop_prob=1 - self.ui_p_drop)
        adj_coo  = adj.coalesce()
        indices  = adj_coo.indices()
        values   = adj_coo.values()
        device   = adj.device
        n_u      = self.n_users
        item_keep = self.item_kg_degree.to(device)
        rows, cols = indices[0], indices[1]
        row_is_item = rows >= n_u; col_is_item = cols >= n_u
        has_item    = row_is_item | col_is_item
        item_node   = torch.where(row_is_item, rows - n_u, cols - n_u)
        item_node   = item_node.clamp(0, len(item_keep) - 1)
        p_keep = self.ui_p_drop + (1.0 - self.ui_p_drop) * item_keep[item_node]
        p_keep = torch.where(has_item, p_keep, torch.ones_like(p_keep)).clamp(0.0, 1.0)
        keep_mask   = torch.rand_like(p_keep) < p_keep
        new_indices = indices[:, keep_mask]
        new_values  = values[keep_mask]
        return torch.sparse_coo_tensor(
            new_indices, new_values, adj.shape, device=device).coalesce()

    @staticmethod
    def _random_edge_drop(adj, drop_prob):
        adj_coo = adj.coalesce()
        values  = adj_coo.values()
        mask    = torch.rand_like(values) > (1 - drop_prob)
        return torch.sparse_coo_tensor(
            adj_coo.indices()[:, mask], values[mask],
            adj.shape, device=adj.device).coalesce()

    def refresh_augmented_views(self):
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        self._aug_adj1 = self._augment_adj(adj)
        self._aug_adj2 = self._augment_adj(adj)

    def contrastive_loss(self, view1, view2):
        v1  = F.normalize(view1, dim=-1)
        v2  = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.temp
        labels = torch.arange(len(v1), device=v1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def forward(self, users, pos_items, neg_items):
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        entity_emb = self._kg_propagation()
        user_emb, item_emb = self._cf_propagation(adj, entity_emb)
        adj1 = self._aug_adj1 if self._aug_adj1 is not None else self._augment_adj(adj)
        adj2 = self._aug_adj2 if self._aug_adj2 is not None else self._augment_adj(adj)
        u1, i1 = self._cf_propagation(adj1, entity_emb)
        u2, i2 = self._cf_propagation(adj2, entity_emb)
        return (user_emb[users], item_emb[pos_items], item_emb[neg_items],
                u1[users], u2[users], i1[pos_items], i2[pos_items])

    def get_embeddings(self):
        with torch.no_grad():
            _dev = self.user_embedding.weight.device
            adj  = self.norm_adj.to(_dev)
            entity_emb = self._kg_propagation()
            return self._cf_propagation(adj, entity_emb)

    def l2_loss(self, users, pos_items, neg_items):
        u0 = self.user_embedding(users)
        p0 = self.entity_embedding(pos_items)
        n0 = self.entity_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
                n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj):
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_norm_adj(self, kg_norm_adj):
        self.register_buffer("kg_norm_adj", kg_norm_adj)

    def set_item_entity_map(self, m):
        pass  # [v10] item_id == entity_id theo KGAT
