"""
models/kg_lightgcn.py — v10
KGLightGCN  (Variant 1 — Base):  BPR + KG enrichment
KGLightGCNCL (Variant 2 — Enhanced): BPR + Cross-view CL + KG alignment

[v10] BẤT BIẾN KGAT: item_id == entity_id cho n_items thực thể đầu tiên.
      item2entity là identity mapping — KHÔNG cần lookup table riêng.
      Dùng kg_final.txt từ KGAT repo trực tiếp.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class _KGEnrichMixin:
    """Shared KG entity propagation và item enrichment."""

    def _propagate_entity_embeddings(self) -> torch.Tensor:
        E: torch.Tensor = self.entity_embedding.weight
        kg_adj   = self.kg_norm_adj
        kg_n_layers: int = self.kg_n_layers
        if kg_adj is None or kg_n_layers == 0:
            return E
        _dev = self.entity_embedding.weight.device
        adj  = kg_adj.to(_dev)
        E_k  = E
        acc  = E.clone()
        for _ in range(kg_n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        return acc / (kg_n_layers + 1)

    def _get_entity_for_items(self) -> torch.Tensor:
        """
        [v10] BẤT BIẾN KGAT: item_id == entity_id.
        Lấy trực tiếp entity_emb[0:n_items] — không cần mapping.
        """
        n_items: int = self.n_items
        emb_dim: int = self.embedding_dim
        dev = self.item_embedding.weight.device
        entity_emb = self._propagate_entity_embeddings()
        n = min(n_items, entity_emb.shape[0])
        entity_for_items = torch.zeros(n_items, emb_dim, device=dev)
        entity_for_items[:n] = entity_emb[:n]
        return entity_for_items

    def _enrich_item_embeddings(self) -> torch.Tensor:
        item_emb: torch.Tensor = self.item_embedding.weight
        if not self.has_kg:
            return item_emb
        entity_for_items = self._get_entity_for_items()
        alpha = torch.sigmoid(self.alpha)
        return alpha * item_emb + (1.0 - alpha) * entity_for_items

    def kg_alignment_loss(self) -> torch.Tensor:
        if not self.has_kg:
            return torch.tensor(0.0, device=self.item_embedding.weight.device)
        item_emb:   torch.Tensor = self.item_embedding.weight
        entity_emb: torch.Tensor = self.entity_embedding.weight
        n = min(self.n_items, entity_emb.shape[0])
        cos_sim = F.cosine_similarity(
            item_emb[:n], entity_emb[:n].detach(), dim=-1)
        return (1.0 - cos_sim).mean()


class KGLightGCN(_KGEnrichMixin, BaseModel):
    """KG-LightGCN Variant 1 (Base): KG enrichment + BPR only."""

    def __init__(
        self,
        n_users:      int,
        n_items:      int,
        n_entities:   int = 0,
        n_relations:  int = 0,
        embedding_dim: int = 64,
        n_layers:     int = 3,
        kg_n_layers:  int = 2,
        kg_type:      str = "full",
        entity_agg:   str = "mean",
        kg_reg:       float = 1e-5,
        norm_adj:     Optional[torch.Tensor] = None,
        device:       Optional[torch.device] = None,
    ) -> None:
        BaseModel.__init__(self, n_users, n_items, embedding_dim, device)
        self.n_entities  = max(n_entities, n_items)
        self.n_relations = n_relations
        self.n_layers    = n_layers
        self.kg_n_layers = kg_n_layers
        self.kg_type     = kg_type
        self.entity_agg  = entity_agg
        self.kg_reg      = kg_reg
        self.has_kg      = (kg_type != "none" and n_entities > 0)
        self.user_embedding = nn.Embedding(n_users,  embedding_dim)
        self.item_embedding = nn.Embedding(n_items,  embedding_dim)
        if self.has_kg:
            self.entity_embedding = nn.Embedding(self.n_entities, embedding_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("norm_adj",    norm_adj)
        self.register_buffer("kg_norm_adj", None)
        self._init_weights()

    def _cf_propagation(
        self, item_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        E0   = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        E_k  = E0
        acc  = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, pos_items, neg_items):
        item_enriched = self._enrich_item_embeddings()
        user_final, item_final = self._cf_propagation(item_enriched)
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self):
        with torch.no_grad():
            return self._cf_propagation(self._enrich_item_embeddings())

    def l2_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        u0  = self.user_embedding(users)
        p0  = self.item_embedding(pos_items)
        n0  = self.item_embedding(neg_items)
        reg = (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
               n0.norm(2).pow(2)) / (2 * len(users))
        if self.has_kg:
            e0  = self.entity_embedding(
                pos_items.clamp(0, self.n_entities - 1))
            reg = reg + self.kg_reg * e0.norm(2).pow(2).mean()
        return reg

    def set_adj(self, norm_adj):
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_norm_adj(self, adj):
        self.register_buffer("kg_norm_adj", adj)

    def set_item_entity_map(self, m):
        # [v10] KGAT convention: item_id == entity_id, không cần mapping
        pass


class KGLightGCNCL(_KGEnrichMixin, BaseModel):
    """KG-LightGCN-CL Variant 2 (Enhanced): BPR + Cross-view CL + KG alignment."""

    def __init__(
        self,
        n_users:      int,
        n_items:      int,
        n_entities:   int = 0,
        n_relations:  int = 0,
        embedding_dim: int = 64,
        n_layers:     int = 3,
        kg_n_layers:  int = 2,
        kg_type:      str = "full",
        entity_agg:   str = "mean",
        kg_reg:       float = 1e-5,
        cl_temp:      float = 0.2,
        lambda_cl:    float = 0.5,
        eps:          float = 0.1,
        norm_adj:     Optional[torch.Tensor] = None,
        device:       Optional[torch.device] = None,
    ) -> None:
        BaseModel.__init__(self, n_users, n_items, embedding_dim, device)
        self.n_entities  = max(n_entities, n_items)
        self.n_relations = n_relations
        self.n_layers    = n_layers
        self.kg_n_layers = kg_n_layers
        self.kg_type     = kg_type
        self.entity_agg  = entity_agg
        self.kg_reg      = kg_reg
        self.cl_temp     = cl_temp
        self.lambda_cl   = lambda_cl
        self.eps         = eps
        self.has_kg      = (kg_type != "none" and n_entities > 0)
        self.user_embedding = nn.Embedding(n_users,  embedding_dim)
        self.item_embedding = nn.Embedding(n_items,  embedding_dim)
        if self.has_kg:
            self.entity_embedding = nn.Embedding(self.n_entities, embedding_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("norm_adj",    norm_adj)
        self.register_buffer("kg_norm_adj", None)
        self._init_weights()

    def _cf_propagation(
        self, item_emb: torch.Tensor, perturb: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        E0   = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        E_k  = E0
        acc  = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            if perturb:
                noise = (torch.rand_like(E_k) * 2.0 - 1.0) * self.eps
                E_k   = E_k + noise
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[:self.n_users], E_final[self.n_users:]

    def _get_cf_view(self):
        return self._cf_propagation(self.item_embedding.weight, perturb=True)

    def _get_kg_view(self):
        if self.has_kg:
            return self._cf_propagation(
                self._enrich_item_embeddings(), perturb=False)
        return self._cf_propagation(self.item_embedding.weight, perturb=True)

    def contrastive_loss(self, view1, view2) -> torch.Tensor:
        v1  = F.normalize(view1, dim=-1)
        v2  = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.cl_temp
        labels = torch.arange(len(v1), device=v1.device)
        return (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)) / 2.0

    def forward(self, users, pos_items, neg_items):
        item_enriched = self._enrich_item_embeddings()
        user_main, item_main = self._cf_propagation(item_enriched, perturb=False)
        user_cf_all, item_cf_all = self._get_cf_view()
        user_kg_all, item_kg_all = self._get_kg_view()
        return (
            user_main[users], item_main[pos_items], item_main[neg_items],
            user_cf_all[users], user_kg_all[users],
            item_cf_all[pos_items], item_kg_all[pos_items],
        )

    def get_embeddings(self):
        with torch.no_grad():
            return self._cf_propagation(
                self._enrich_item_embeddings(), perturb=False)

    def l2_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        u0  = self.user_embedding(users)
        p0  = self.item_embedding(pos_items)
        n0  = self.item_embedding(neg_items)
        reg = (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
               n0.norm(2).pow(2)) / (2 * len(users))
        if self.has_kg:
            e0  = self.entity_embedding(
                pos_items.clamp(0, self.n_entities - 1))
            reg = reg + self.kg_reg * e0.norm(2).pow(2).mean()
        return reg

    def set_adj(self, adj):
        self.register_buffer("norm_adj", adj)

    def set_kg_norm_adj(self, adj):
        self.register_buffer("kg_norm_adj", adj)

    def set_item_entity_map(self, m):
        # [v10] KGAT convention: item_id == entity_id, không cần mapping
        pass
