"""
models/kgat.py — v10-fix
KGAT: Knowledge Graph Attention Network
Wang et al., KDD 2019 — https://arxiv.org/abs/1905.07854

Dữ liệu từ KGAT repo (kg_final.txt). Bất biến item_id == entity_id.

═══════════════════════════════════════════════════════════════════════════════
KGAT ARCHITECTURE (theo paper gốc):

[1] KG Propagation (Attentive Embedding Propagation):
    Attention score: π(h, r, t) = (e_r)^T * tanh(W_r * e_t + e_r)
      - e_r: relation embedding
      - W_r: relation-specific transform (TransR projection)
      - e_t: tail entity embedding
    Normalized: α(h,r,t) = softmax over all (h,r,*) neighbors
    Aggregated: e_N(h) = sum_{(r,t) in N(h)} α(h,r,t) * e_t
    Updated:    e_h^{l} = LeakyReLU(W1 * (e_h^{l-1} + e_N^{l-1}(h)))

[2] CF Propagation (Bi-Interaction Aggregation):
    Aggregation:
      e_u* = LeakyReLU(W1 * (e_u + sum_i)) + LeakyReLU(W2 * (e_u ⊙ sum_i))
    Final: e_u = L2_norm(concat(e_u^0, e_u^1, ..., e_u^L))
           (paper dùng concat + L2 normalize, KHÔNG dùng W_out linear)

[3] Prediction: y_hat(u,i) = e_u^T * e_i

[4] Loss:
    L = L_BPR_cf + λ_KG * L_BPR_kg + λ_reg * ||Θ||^2
    L_BPR_kg = -sum log σ(||h+r-t_neg||^2 - ||h+r-t_pos||^2)

═══════════════════════════════════════════════════════════════════════════════
THAY ĐỔI so với code cũ (v10-fix):

1. KG attention: dùng ĐÚNG formula paper:
   π(h,r,t) = e_r^T * tanh(W_r * e_t + e_r)
   (không phải t_proj * gate như code cũ)

2. CF aggregation: dùng ĐÚNG Bi-Interaction:
   LeakyReLU(W1*(e+nb)) + LeakyReLU(W2*(e⊙nb))
   với LeakyReLU bọc MỖI NHÁNH riêng biệt

3. Final embedding: concat + L2 normalize (không concat + W_out linear)
   Paper: concat all layer outputs → L2 normalize

4. W_r (TransR): relation-specific matrix W_r ∈ R^{d×d} cho mỗi relation
   (TransR projection: project entity vào relation-specific space)

5. Entity aggregation update rule: GCN-style LayerNorm + residual
   e_h^{l} = normalize(LeakyReLU(W_agg * (e_h^{l-1} + e_N(h)^{l-1})))
═══════════════════════════════════════════════════════════════════════════════
"""
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

_DEFAULT_CHUNK = 16_384  # Giảm chunk để tránh OOM khi tính attention


class KGAT(BaseModel):
    """
    KGAT: Knowledge Graph Attention Network (Wang et al., KDD 2019).

    Bất biến v10: item_id == entity_id (KGAT convention).
    """

    def __init__(
        self,
        n_users:      int,
        n_items:      int,
        n_entities:   int,
        n_relations:  int,
        embedding_dim: int = 64,
        relation_dim:  int = 64,
        n_layers:      int = 3,
        kg_n_layers:   int = 2,
        agg_type:      str = "bi-interaction",   # "bi-interaction" | "sum" | "concat"
        norm_adj:      Optional[torch.Tensor] = None,
        node_dropout:  float = 0.0,
        mess_dropout:  float = 0.0,
        device:        Optional[torch.device] = None,
        chunk_size:    int = _DEFAULT_CHUNK,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)

        self.n_entities   = n_entities
        self.n_relations  = n_relations
        self.relation_dim = relation_dim
        self.n_layers     = n_layers
        self.kg_n_layers  = kg_n_layers
        self.agg_type     = agg_type
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.chunk_size   = chunk_size

        # ── Embeddings ──────────────────────────────────────────────────────
        self.user_embedding     = nn.Embedding(n_users,    embedding_dim)
        self.entity_embedding   = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)

        # ── TransR: relation-specific projection W_r ∈ R^{d×d} ─────────────
        # [v10-fix] Mỗi relation có ma trận W_r riêng để project entity
        # vào relation-specific space. Paper KGAT dùng TransR.
        # Để memory-efficient: dùng Embedding lưu flatten(W_r)
        # Shape: n_relations × (embedding_dim × relation_dim)
        self.trans_w = nn.Embedding(
            n_relations, embedding_dim * relation_dim)

        # ── KG Aggregation layers: W_agg cho từng hop ───────────────────────
        # [v10-fix] Mỗi layer có 1 linear transform W_agg để update entity emb
        # Paper: e_h^l = normalize(LeakyReLU(W_agg * (e_h^{l-1} + e_N^{l-1}(h))))
        self.W_agg = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(kg_n_layers)
        ])

        # ── CF Bi-Interaction Aggregation ────────────────────────────────────
        # [v10-fix] Paper dùng 2 nhánh W1 và W2 RIÊNG BIỆT
        # e_agg = LeakyReLU(W1*(e+nb)) + LeakyReLU(W2*(e⊙nb))
        if agg_type == "bi-interaction":
            self.W1 = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(n_layers)
            ])
            self.W2 = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(n_layers)
            ])
        else:
            # sum aggregation (simpler baseline)
            self.W1 = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(n_layers)
            ])
            self.W2 = None

        # ── Dropout ──────────────────────────────────────────────────────────
        self.node_drop = nn.Dropout(p=node_dropout) if node_dropout > 0 else None
        self.mess_drop = nn.Dropout(p=mess_dropout) if mess_dropout > 0 else None

        # ── Adjacency ────────────────────────────────────────────────────────
        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        # KG adjacency: head → [(tail, relation), ...]
        self.kg_adj: Dict[int, List[Tuple[int, int]]] = {}

        # Flat tensors cho vectorized attention (build khi set_kg_adj được gọi)
        self._kg_heads_cpu: Optional[torch.Tensor] = None
        self._kg_tails_cpu: Optional[torch.Tensor] = None
        self._kg_rels_cpu:  Optional[torch.Tensor] = None

        # ── Init ─────────────────────────────────────────────────────────────
        self._init_weights()
        # TransR cần uniform init riêng
        nn.init.xavier_uniform_(self.trans_w.weight)

    # =========================================================================
    # TransR projection (relation-specific space)
    # =========================================================================

    def _transr_project(
        self,
        entity_emb:  torch.Tensor,   # (B, d)
        relation_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Project entity embeddings vào relation-specific space.
        W_r ∈ R^{d_e × d_r}, project: e_proj = W_r^T * e_entity
        Shape output: (B, relation_dim)
        """
        # W_r flatten: (B, d_e * d_r)
        W = self.trans_w(relation_id).view(
            -1, self.embedding_dim, self.relation_dim)
        # (B, 1, d_e) × (B, d_e, d_r) → (B, 1, d_r) → (B, d_r)
        return torch.bmm(entity_emb.unsqueeze(1), W).squeeze(1)

    # =========================================================================
    # KG score (TransR distance) — dùng cho kg_bpr_loss
    # =========================================================================

    def kg_forward(
        self,
        heads:     torch.Tensor,   # (B,)
        relations: torch.Tensor,   # (B,)
        pos_tails: torch.Tensor,   # (B,)
        neg_tails: torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính TransR score cho positive và negative triples.

        Score = ||e_h_proj + e_r - e_t_proj||^2
        (nhỏ hơn = triple đúng hơn)

        Returns:
            pos_dist: squared distance của positive triple
            neg_dist: squared distance của negative triple
        """
        e_h  = self.entity_embedding(heads)
        e_r  = self.relation_embedding(relations)
        e_tp = self.entity_embedding(pos_tails)
        e_tn = self.entity_embedding(neg_tails)

        # Project entities vào relation-specific space
        h_proj  = self._transr_project(e_h,  relations)   # (B, d_r)
        tp_proj = self._transr_project(e_tp, relations)   # (B, d_r)
        tn_proj = self._transr_project(e_tn, relations)   # (B, d_r)

        # TransR score: ||h_proj + r - t_proj||^2
        pos_dist = ((h_proj + e_r - tp_proj) ** 2).sum(dim=-1)
        neg_dist = ((h_proj + e_r - tn_proj) ** 2).sum(dim=-1)

        return pos_dist, neg_dist

    # =========================================================================
    # KG Attention Propagation (Attentive Embedding Propagation)
    # =========================================================================

    def _build_kg_flat_tensors(self) -> None:
        """Xây dựng flat tensors từ kg_adj để vectorize attention."""
        if not self.kg_adj:
            return
        heads_l, tails_l, rels_l = [], [], []
        for h, neighbors in self.kg_adj.items():
            for t, r in neighbors:
                heads_l.append(h)
                tails_l.append(t)
                rels_l.append(r)
        if not heads_l:
            return
        self._kg_heads_cpu = torch.tensor(heads_l, dtype=torch.long)
        self._kg_tails_cpu = torch.tensor(tails_l, dtype=torch.long)
        self._kg_rels_cpu  = torch.tensor(rels_l,  dtype=torch.long)

    def _attention_score(
        self,
        e_r:     torch.Tensor,   # (B, d_r) — relation embeddings
        e_t:     torch.Tensor,   # (B, d_e) — tail entity embeddings
        rels:    torch.Tensor,   # (B,) — relation ids (cho TransR project)
    ) -> torch.Tensor:
        """
        [v10-fix] Attention score theo paper KGAT (Section 3.2):

        π*(h, r, t) = e_r^T * tanh(W_r * e_t + e_r)

        Bước 1: project e_t vào relation space: e_t_proj = W_r * e_t
        Bước 2: compute gate: tanh(e_t_proj + e_r)
        Bước 3: dot product với e_r: score = e_r * gate → sum

        Returns: (B,) raw attention score (chưa softmax)
        """
        # e_t_proj: project tail entity vào relation space
        e_t_proj = self._transr_project(e_t, rels)          # (B, d_r)
        # gate: tanh(W_r * e_t + e_r)
        gate = torch.tanh(e_t_proj + e_r)                   # (B, d_r)
        # π*(h,r,t) = e_r^T * gate (dot product per sample)
        score = (e_r * gate).sum(dim=-1)                     # (B,)
        return score

    def _kg_propagation_one_hop(
        self,
        E:      torch.Tensor,    # (n_entities, d) — current entity embeddings (GPU)
        hop:    int,             # layer index
        device: torch.device,
    ) -> torch.Tensor:
        """
        Một hop KG attention propagation:
        1. Tính attention score π*(h,r,t) cho mọi edge
        2. Softmax normalize theo head entity
        3. Aggregate: e_N(h) = sum_t α(h,r,t) * e_t
        4. Update: e_h^{l} = normalize(LeakyReLU(W_agg * (e_h + e_N(h))))
        """
        if self._kg_heads_cpu is None:
            # Không có KG edges → trả lại unchanged
            return F.normalize(
                F.leaky_relu(self.W_agg[hop](E), negative_slope=0.2), dim=-1)

        heads_cpu = self._kg_heads_cpu   # (E_total,)
        tails_cpu = self._kg_tails_cpu   # (E_total,)
        rels_cpu  = self._kg_rels_cpu    # (E_total,)
        E_total   = heads_cpu.size(0)
        E_cpu     = E.detach().cpu()
        chunk     = self.chunk_size

        # === Pass 1: tính exp(π*(h,r,t)) và softmax denominator ===
        attn_exp = torch.zeros(E_total, dtype=torch.float32)
        attn_sum = torch.zeros(self.n_entities, dtype=torch.float32)

        for start in range(0, E_total, chunk):
            end   = min(start + chunk, E_total)
            h_idx = heads_cpu[start:end]
            t_idx = tails_cpu[start:end]
            r_idx = rels_cpu[start:end].to(device)

            e_h_b = E_cpu[h_idx].to(device)   # (B, d)
            e_t_b = E_cpu[t_idx].to(device)   # (B, d)
            e_r_b = self.relation_embedding(r_idx)   # (B, d_r)

            score = self._attention_score(e_r_b, e_t_b, r_idx)  # (B,)
            # Numerical stability: subtract max per head (approximate)
            exp_v = torch.exp(score).cpu()

            attn_exp[start:end] = exp_v
            attn_sum.scatter_add_(0, h_idx, exp_v)

            del e_h_b, e_t_b, e_r_b, score, exp_v, r_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # === Pass 2: weighted aggregate ===
        aggregated = torch.zeros_like(E_cpu)   # (n_entities, d)

        for start in range(0, E_total, chunk):
            end   = min(start + chunk, E_total)
            h_idx = heads_cpu[start:end]
            t_idx = tails_cpu[start:end]

            denom  = (attn_sum[h_idx] + 1e-8).unsqueeze(1)   # (B, 1)
            norm_w = (attn_exp[start:end] / denom.squeeze(1)).unsqueeze(1)  # (B, 1)
            weighted = norm_w * E_cpu[t_idx]   # (B, d)

            aggregated.scatter_add_(
                0,
                h_idx.unsqueeze(1).expand(-1, self.embedding_dim),
                weighted,
            )
            del weighted, norm_w, denom

        # === Update: e_h^l = normalize(LeakyReLU(W_agg(e_h + e_N(h)))) ===
        agg_gpu = aggregated.to(device)
        # [v10-fix] paper: W_agg * (e_h + e_N), sau đó LeakyReLU + L2 normalize
        E_new = F.leaky_relu(
            self.W_agg[hop](E + agg_gpu), negative_slope=0.2)
        E_new = F.normalize(E_new, dim=-1)

        del E_cpu, aggregated, agg_gpu
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return E_new

    def _compute_entity_embeddings(self) -> torch.Tensor:
        """
        Chạy kg_n_layers hop KG attention propagation.
        Trả về final entity embeddings sau khi propagate.
        """
        if not self.kg_adj:
            return self.entity_embedding.weight

        device = self.entity_embedding.weight.device
        E = self.entity_embedding.weight   # (n_entities, d)

        for hop in range(self.kg_n_layers):
            E = self._kg_propagation_one_hop(E, hop, device)

        return E

    # =========================================================================
    # CF Propagation (Bi-Interaction Aggregation)
    # =========================================================================

    def _cf_propagation(
        self,
        entity_emb: torch.Tensor,   # (n_entities, d) — enriched by KG
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CF propagation với Bi-Interaction aggregation (paper Section 3.3).

        [v10-fix] Đúng theo paper:
        e_agg^l(u) = LeakyReLU(W1*(e^{l-1} + nb^{l-1}))
                   + LeakyReLU(W2*(e^{l-1} ⊙ nb^{l-1}))

        Final embedding: L2_normalize(concat(e^0, e^1, ..., e^L))
        (paper: concat + L2 normalize — KHÔNG dùng linear W_out)

        Returns:
            user_final: (n_users, d)
            item_final: (n_items, d)
        """
        if self.norm_adj is None:
            raise RuntimeError(
                "norm_adj chưa được set. Gọi model.set_adj(norm_adj) trước.")

        # Lấy item embeddings từ entity embedding (KGAT invariant: item_id == entity_id)
        item_e = entity_emb[:self.n_items]   # (n_items, d)

        # Concat user + item → joint embedding matrix
        E0 = torch.cat([self.user_embedding.weight, item_e], dim=0)  # (n_u+n_i, d)

        if self.node_drop is not None and self.training:
            E0 = self.node_drop(E0)

        _dev = E0.device
        adj  = self.norm_adj.to(_dev)

        # Collect all layer outputs for final concat
        layer_outputs = [E0]
        E_k = E0

        for l in range(self.n_layers):
            # Aggregate neighbors via normalized adj
            nb = torch.sparse.mm(adj, E_k)   # (n_u+n_i, d)

            if self.mess_drop is not None and self.training:
                nb = self.mess_drop(nb)

            if self.agg_type == "bi-interaction":
                # [v10-fix] Paper formula: ĐÚNG với 2 nhánh riêng biệt
                branch1 = F.leaky_relu(self.W1[l](E_k + nb), negative_slope=0.2)
                branch2 = F.leaky_relu(self.W2[l](E_k * nb), negative_slope=0.2)
                E_k = branch1 + branch2
            else:
                # sum aggregation
                E_k = F.leaky_relu(self.W1[l](E_k + nb), negative_slope=0.2)

            # L2 normalize theo layer (paper áp dụng normalize sau mỗi layer)
            E_k = F.normalize(E_k, dim=-1)
            layer_outputs.append(E_k)

        # [v10-fix] Final embedding: concat tất cả layers + L2 normalize
        # Paper: e_u* = L2_norm(concat(e_u^0, ..., e_u^L))
        # Dimension sau concat: (n_layers+1) * embedding_dim
        E_concat = torch.cat(layer_outputs, dim=-1)          # (N, (L+1)*d)
        E_final  = F.normalize(E_concat, dim=-1)             # (N, (L+1)*d)

        return E_final[:self.n_users], E_final[self.n_users:]

    # =========================================================================
    # Prediction interface
    # =========================================================================

    def forward(
        self,
        users:     torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        precomputed_entity_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass cho CF recommendation.

        Args:
            precomputed_entity_emb: pre-computed KG entity embeddings
                (tính 1 lần mỗi epoch trong KGTrainer để tránh recompute)
        """
        entity_emb = (
            precomputed_entity_emb
            if precomputed_entity_emb is not None
            else self._compute_entity_embeddings()
        )
        user_final, item_final = self._cf_propagation(entity_emb)
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về final user và item embeddings (dùng cho evaluation)."""
        with torch.no_grad():
            entity_emb = self._compute_entity_embeddings()
            return self._cf_propagation(entity_emb)

    def predict(
        self,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score prediction cho evaluation.
        Override BaseModel.predict() vì KGAT final dim != embedding_dim.
        """
        user_emb, item_emb = self.get_embeddings()
        u = user_emb[users]
        if items is None:
            return torch.matmul(u, item_emb.T)
        elif items.dim() == 1:
            return torch.matmul(u, item_emb[items].T)
        else:
            return torch.bmm(
                u.unsqueeze(1), item_emb[items].transpose(1, 2)).squeeze(1)

    # =========================================================================
    # L2 regularization
    # =========================================================================

    def l2_loss(
        self,
        users:     torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 regularization trên embedding parameters (paper Equation 7).
        Chỉ regularize initial embeddings (layer 0), không regularize transformed.
        """
        u0 = self.user_embedding(users)
        p0 = self.entity_embedding(pos_items)
        n0 = self.entity_embedding(neg_items)
        return (
            u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)
        ) / (2 * len(users))

    # =========================================================================
    # Setup methods
    # =========================================================================

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        """Set normalized CF adjacency matrix D^{-1/2} A D^{-1/2}."""
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_adj(
        self, kg_adj: Dict[int, List[Tuple[int, int]]]
    ) -> None:
        """
        Set KG adjacency list: {head_id: [(tail_id, relation_id), ...]}.
        Tự động build flat tensors cho vectorized attention computation.
        """
        self.kg_adj = kg_adj
        # Reset flat tensors để rebuild
        self._kg_heads_cpu = None
        self._kg_tails_cpu = None
        self._kg_rels_cpu  = None
        self._build_kg_flat_tensors()

    def set_kg_norm_adj(self, adj) -> None:
        """KGAT dùng attention-based KG aggregation (adj list), không dùng norm matrix."""
        pass

    def set_item_entity_map(self, m) -> None:
        """[v10] KGAT convention: item_id == entity_id. No-op."""
        pass

    def to(self, *args, **kwargs):
        """Override để reset CPU tensors khi move sang device mới."""
        self._kg_heads_cpu = None
        self._kg_tails_cpu = None
        self._kg_rels_cpu  = None
        result = super().to(*args, **kwargs)
        # Rebuild sau khi move (tensors sẽ vẫn là CPU tensors — intentional)
        self._build_kg_flat_tensors()
        return result