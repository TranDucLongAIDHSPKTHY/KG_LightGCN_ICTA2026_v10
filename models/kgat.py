"""
models/kgat.py — v10
KGAT: Knowledge Graph Attention Network
Wang et al., KDD 2019 — https://arxiv.org/abs/1905.07854
Dữ liệu từ KGAT repo (kg_final.txt). Bất biến item_id==entity_id.
"""
import gc
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel

_DEFAULT_CHUNK = 32_768


class KGAT(BaseModel):
    def __init__(
        self,
        n_users: int, n_items: int, n_entities: int, n_relations: int,
        embedding_dim: int = 64, relation_dim: int = 64,
        n_layers: int = 3, kg_n_layers: int = 2,
        agg_type: str = "bi-interaction",
        norm_adj: Optional[torch.Tensor] = None,
        node_dropout: float = 0.0, mess_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities  = n_entities
        self.n_relations = n_relations
        self.relation_dim = relation_dim
        self.n_layers    = n_layers
        self.kg_n_layers = kg_n_layers
        self.agg_type    = agg_type
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.chunk_size  = chunk_size

        self.user_embedding     = nn.Embedding(n_users,    embedding_dim)
        self.entity_embedding   = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        self.trans_w = nn.Embedding(n_relations, embedding_dim * relation_dim)

        self.W_kg  = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                    for _ in range(kg_n_layers)])
        self.attn_V = nn.Linear(relation_dim, 1, bias=False)

        if agg_type == "bi-interaction":
            self.W_gc = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                       for _ in range(n_layers)])
            self.W_bi = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                       for _ in range(n_layers)])
        else:
            self.W_gc = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                       for _ in range(n_layers)])

        self.W_out = nn.Linear((n_layers + 1) * embedding_dim, embedding_dim, bias=False)
        self.node_drop = nn.Dropout(p=node_dropout)
        self.mess_drop = nn.Dropout(p=mess_dropout)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self.kg_adj: Dict[int, List[Tuple[int, int]]] = {}
        self._kg_heads_cpu = self._kg_tails_cpu = self._kg_rels_cpu = None
        self._init_weights()
        nn.init.xavier_uniform_(self.trans_w.weight)

    def _project(self, entity_emb, relation_id):
        W = self.trans_w(relation_id).view(-1, self.embedding_dim, self.relation_dim)
        return torch.bmm(entity_emb.unsqueeze(1), W).squeeze(1)

    def kg_forward(self, heads, relations, pos_tails, neg_tails):
        h  = self.entity_embedding(heads)
        r  = self.relation_embedding(relations)
        tp = self.entity_embedding(pos_tails)
        tn = self.entity_embedding(neg_tails)
        h_r  = self._project(h,  relations)
        tp_r = self._project(tp, relations)
        tn_r = self._project(tn, relations)
        pos_score = ((h_r + r - tp_r) ** 2).sum(dim=-1)
        neg_score = ((h_r + r - tn_r) ** 2).sum(dim=-1)
        return pos_score, neg_score

    def _build_kg_flat_tensors(self):
        if not self.kg_adj:
            return
        heads_l, tails_l, rels_l = [], [], []
        for h, nbrs in self.kg_adj.items():
            for t, r in nbrs:
                heads_l.append(h); tails_l.append(t); rels_l.append(r)
        if not heads_l:
            return
        self._kg_heads_cpu = torch.tensor(heads_l, dtype=torch.long)
        self._kg_tails_cpu = torch.tensor(tails_l, dtype=torch.long)
        self._kg_rels_cpu  = torch.tensor(rels_l,  dtype=torch.long)

    def _single_hop_attention(self, E, hop, device):
        if self._kg_heads_cpu is None:
            self._build_kg_flat_tensors()
        if self._kg_heads_cpu is None:
            return E
        heads_cpu = self._kg_heads_cpu
        tails_cpu = self._kg_tails_cpu
        rels_cpu  = self._kg_rels_cpu
        E_total   = heads_cpu.size(0)
        E_cpu     = E.detach().cpu()
        chunk     = self.chunk_size
        attn_exp  = torch.zeros(E_total)
        attn_sum  = torch.zeros(self.n_entities)
        for start in range(0, E_total, chunk):
            end   = min(start + chunk, E_total)
            h_idx = heads_cpu[start:end]
            t_idx = tails_cpu[start:end]
            r_idx = rels_cpu[start:end].to(device)
            h_emb = E_cpu[h_idx].to(device)
            t_emb = E_cpu[t_idx].to(device)
            r_emb = self.relation_embedding(r_idx)
            h_proj = self._project(h_emb, r_idx)
            t_proj = self._project(t_emb, r_idx)
            gate  = torch.tanh(h_proj + r_emb)
            raw   = self.attn_V(t_proj * gate).squeeze(-1)
            exp_v = torch.exp(raw - raw.max()).cpu()
            attn_exp[start:end] = exp_v
            attn_sum.scatter_add_(0, h_idx, exp_v)
            del h_emb, t_emb, r_emb, h_proj, t_proj, gate, raw, exp_v, r_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()
        aggregated = torch.zeros_like(E_cpu)
        for start in range(0, E_total, chunk):
            end    = min(start + chunk, E_total)
            h_idx  = heads_cpu[start:end]
            t_idx  = tails_cpu[start:end]
            denom  = (attn_sum[h_idx] + 1e-8).unsqueeze(1)
            attn_w = (attn_exp[start:end] / denom.squeeze(1)).unsqueeze(1)
            weighted = attn_w * E_cpu[t_idx]
            aggregated.scatter_add_(
                0, h_idx.unsqueeze(1).expand(-1, self.embedding_dim), weighted)
            del weighted, attn_w, denom
        agg_gpu = aggregated.to(device)
        E_new   = F.leaky_relu(self.W_kg[hop](E + agg_gpu))
        E_new   = F.normalize(E_new, dim=-1)
        del E_cpu, aggregated, agg_gpu; gc.collect()
        return E_new

    def _compute_entity_embeddings(self):
        if not self.kg_adj:
            return self.entity_embedding.weight
        device = self.entity_embedding.weight.device
        E = self.entity_embedding.weight
        for hop in range(self.kg_n_layers):
            E = self._single_hop_attention(E, hop, device)
        return E

    def _cf_propagation(self, entity_emb):
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        item_e = entity_emb[:self.n_items]
        E0     = torch.cat([self.user_embedding.weight, item_e], dim=0)
        _dev   = self.user_embedding.weight.device
        adj    = self.norm_adj.to(_dev)
        if self.training and self.node_dropout > 0:
            E0 = self.node_drop(E0)
        E_k = E0
        layer_outputs = [E0]
        for l in range(self.n_layers):
            nb = torch.sparse.mm(adj, E_k)
            if self.training and self.mess_dropout > 0:
                nb = self.mess_drop(nb)
            if self.agg_type == "bi-interaction":
                E_k = (F.leaky_relu(self.W_gc[l](E_k + nb))
                       + F.leaky_relu(self.W_bi[l](E_k * nb)))
            else:
                E_k = F.leaky_relu(self.W_gc[l](nb))
            E_k = F.normalize(E_k, dim=-1)
            layer_outputs.append(E_k)
        E_concat = torch.cat(layer_outputs, dim=-1)
        E_final  = self.W_out(E_concat)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, pos_items, neg_items, precomputed_entity_emb=None):
        entity_emb = (precomputed_entity_emb if precomputed_entity_emb is not None
                      else self._compute_entity_embeddings())
        user_final, item_final = self._cf_propagation(entity_emb)
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self):
        with torch.no_grad():
            entity_emb = self._compute_entity_embeddings()
            return self._cf_propagation(entity_emb)

    def l2_loss(self, users, pos_items, neg_items):
        u0 = self.user_embedding(users)
        p0 = self.entity_embedding(pos_items)
        n0 = self.entity_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) +
                n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj):
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_adj(self, kg_adj):
        self.kg_adj = kg_adj
        self._kg_heads_cpu = self._kg_tails_cpu = self._kg_rels_cpu = None
        self._build_kg_flat_tensors()

    def set_kg_norm_adj(self, adj):
        pass  # KGAT dùng adj list, không dùng norm matrix

    def set_item_entity_map(self, m):
        pass  # [v10] item_id == entity_id theo KGAT

    def to(self, *args, **kwargs):
        self._kg_heads_cpu = self._kg_tails_cpu = self._kg_rels_cpu = None
        return super().to(*args, **kwargs)
