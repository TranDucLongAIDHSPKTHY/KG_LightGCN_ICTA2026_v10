"""
trainers/kg_trainer.py — v10-fix
KGTrainer cho các mô hình KG (KGAT, KGCL, KG-LightGCN, KG-LightGCN-CL).

THAY ĐỔI v10-fix so với v10:
  - KGAT: _kgat_step cập nhật cho kg_bpr_loss signature mới
    (pos_dist, neg_dist) — KHÔNG phải (neg, pos) nữa
  - KGAT: Final embedding dim = (n_layers+1) * embedding_dim sau concat
    → evaluator cần nhận đúng dim này (BaseModel.predict() đã handle)
  - split "valid" thay cho "val" (giữ nguyên từ v10)
"""
import gc
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from datasets.kg_dataset import KGDataset
from evaluation.evaluator import Evaluator
from losses.bpr_loss import bpr_loss, kg_bpr_loss
from losses.contrastive_loss import infonce_loss
from trainers.trainer import Trainer, _save_multiseed_summary
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("kg_trainer")


class KGTrainer(Trainer):
    """
    Trainer cho các KG-based models: KGAT, KGCL, KG-LightGCN, KG-LightGCN-CL.

    Tự động detect model type qua duck-typing (hasattr checks).
    """

    def __init__(
        self,
        model,
        train_loader,
        kg_dataset:     KGDataset,
        evaluator:      Evaluator,
        cfg:            dict,
        device:         torch.device,
        checkpoint_dir: str = "results/checkpoints",
        log_dir:        str = "results/logs",
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        self.kg_dataset = kg_dataset

        # ── Model type detection ─────────────────────────────────────────────
        # Thứ tự detect quan trọng: KG-LightGCN-CL trước KGCL và KG-LightGCN
        self._is_kgat = hasattr(model, "kg_forward") and hasattr(model, "set_kg_adj")

        self._is_kg_lightgcn_cl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "kg_alignment_loss")
            and hasattr(model, "cl_temp")
            and not self._is_kgat
        )

        self._is_kgcl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "set_kg_norm_adj")
            and not self._is_kg_lightgcn_cl
            and not self._is_kgat
        )

        self._is_kg_lightgcn = (
            hasattr(model, "kg_alignment_loss")
            and not self._is_kg_lightgcn_cl
            and not self._is_kgat
        )

        logger.info(
            f"KGTrainer model type: "
            f"kgat={self._is_kgat}, "
            f"kgcl={self._is_kgcl}, "
            f"kg_lightgcn_cl={self._is_kg_lightgcn_cl}, "
            f"kg_lightgcn={self._is_kg_lightgcn}"
        )

        # ── Loss weights ─────────────────────────────────────────────────────
        model_cfg = cfg.get("model", {})
        self.lambda_kg = float(model_cfg.get("kg_reg", 1e-5))

        cl_cfg = cfg.get("contrastive", {})
        self.lambda_cl = float(cl_cfg.get("lambda_cl", 0.1))

        # Cache cho KGAT pre-computed entity embeddings
        self._cached_entity_emb: Optional[torch.Tensor] = None

    # =========================================================================
    # Main training loop
    # =========================================================================

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        # KGAT: pre-compute entity embeddings 1 lần/epoch (expensive)
        if self._is_kgat:
            self._free_entity_emb_cache()
            with torch.no_grad():
                self._cached_entity_emb = (
                    self.model._compute_entity_embeddings().detach()
                )

        # KGCL: refresh augmented views 1 lần/epoch
        if self._is_kgcl and hasattr(self.model, "refresh_augmented_views"):
            with torch.no_grad():
                self.model.refresh_augmented_views()

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            if self._is_kgat:
                loss = self._kgat_step(users, pos_items, neg_items)
            elif self._is_kg_lightgcn_cl:
                loss = self._kg_lightgcn_cl_step(users, pos_items, neg_items)
            elif self._is_kgcl:
                loss = self._kgcl_step(users, pos_items, neg_items)
            elif self._is_kg_lightgcn:
                loss = self._kg_lightgcn_step(users, pos_items, neg_items)
            else:
                # Fallback CF model
                u, p, n = self.model(users, pos_items, neg_items)
                loss = (
                    bpr_loss(u, p, n)
                    + self.weight_decay * self.model.l2_loss(
                        users, pos_items, neg_items)
                )

            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # Cleanup sau epoch
        if self._is_kgat:
            self._free_entity_emb_cache()

        return total_loss / max(n_batches, 1)

    # =========================================================================
    # Per-model step functions
    # =========================================================================

    def _kgat_step(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> torch.Tensor:
        """
        KGAT training step.

        Loss = L_BPR_cf + λ_KG * L_BPR_kg + λ_reg * L2

        [v10-fix] kg_bpr_loss(pos_dist, neg_dist):
          - pos_dist = TransR distance của positive triple
          - neg_dist = TransR distance của negative triple
          Mục tiêu: neg_dist > pos_dist (negative triple xa hơn positive)
        """
        # CF prediction với pre-computed entity embeddings
        u, p, n = self.model(
            users, pos_items, neg_items,
            precomputed_entity_emb=self._cached_entity_emb,
        )
        cf_loss = bpr_loss(u, p, n)
        l2_loss = self.model.l2_loss(users, pos_items, neg_items)

        # KG TransR loss
        kg_loss = torch.tensor(0.0, device=self.device)
        if self.kg_dataset.kg_triples is not None:
            triples = self.kg_dataset.sample_kg_triples(len(users))
            if triples is not None:
                h, r, t_pos, t_neg = triples
                h_t   = torch.tensor(h,     dtype=torch.long, device=self.device)
                r_t   = torch.tensor(r,     dtype=torch.long, device=self.device)
                tp_t  = torch.tensor(t_pos, dtype=torch.long, device=self.device)
                tn_t  = torch.tensor(t_neg, dtype=torch.long, device=self.device)

                # [v10-fix] kg_forward trả về (pos_dist, neg_dist)
                pos_dist, neg_dist = self.model.kg_forward(h_t, r_t, tp_t, tn_t)
                # [v10-fix] kg_bpr_loss(pos_dist, neg_dist) — đúng chiều
                kg_loss = kg_bpr_loss(pos_dist, neg_dist)

        return cf_loss + self.lambda_kg * kg_loss + self.weight_decay * l2_loss

    def _kgcl_step(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> torch.Tensor:
        """KGCL training step: BPR + CL (user view + item view)."""
        u, p, n, u1, u2, i1, i2 = self.model(users, pos_items, neg_items)
        cf_loss  = bpr_loss(u, p, n)
        l2_loss  = self.model.l2_loss(users, pos_items, neg_items)
        user_cl  = self.model.contrastive_loss(u1, u2)
        item_cl  = self.model.contrastive_loss(i1, i2)
        cl_loss  = user_cl + item_cl
        return cf_loss + self.lambda_cl * cl_loss + self.weight_decay * l2_loss

    def _kg_lightgcn_cl_step(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> torch.Tensor:
        """KG-LightGCN-CL step: BPR + Cross-view CL + KG alignment."""
        u, p, n, u_cf, u_kg, i_cf, i_kg = self.model(users, pos_items, neg_items)
        cf_loss    = bpr_loss(u, p, n)
        user_cl    = self.model.contrastive_loss(u_cf, u_kg)
        item_cl    = self.model.contrastive_loss(i_cf, i_kg)
        cl_loss    = (user_cl + item_cl) / 2.0
        align_loss = self.model.kg_alignment_loss()
        l2_loss    = self.model.l2_loss(users, pos_items, neg_items)
        return (
            cf_loss
            + self.lambda_cl  * cl_loss
            + self.lambda_kg  * align_loss
            + self.weight_decay * l2_loss
        )

    def _kg_lightgcn_step(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> torch.Tensor:
        """KG-LightGCN step: BPR + KG alignment (no CL)."""
        u, p, n    = self.model(users, pos_items, neg_items)
        cf_loss    = bpr_loss(u, p, n)
        align_loss = self.model.kg_alignment_loss()
        l2_loss    = self.model.l2_loss(users, pos_items, neg_items)
        return cf_loss + self.lambda_kg * align_loss + self.weight_decay * l2_loss

    # =========================================================================
    # Helpers
    # =========================================================================

    def _free_entity_emb_cache(self) -> None:
        """Giải phóng cached entity embeddings."""
        if self._cached_entity_emb is not None:
            del self._cached_entity_emb
            self._cached_entity_emb = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# =============================================================================
# Multi-seed runner cho KG models
# =============================================================================

def run_kg_multi_seed(
    model_factory,
    train_loader_factory,
    kg_dataset_factory,
    evaluator:      Evaluator,
    cfg:            dict,
    device:         torch.device,
    seeds:          Optional[List[int]] = None,
    checkpoint_dir: str = "results/checkpoints",
    log_dir:        str = "results/logs",
) -> Dict[str, Any]:
    """
    Chạy KG model với nhiều seeds, tổng hợp mean ± std.

    Args:
        model_factory:        callable → model instance
        train_loader_factory: callable(seed) → DataLoader
        kg_dataset_factory:   callable → KGDataset instance
        evaluator:            Evaluator instance
        cfg:                  config dict
        device:               torch device
        seeds:                list of random seeds
    """
    if seeds is None:
        seeds = [42, 0, 1, 2, 3]

    per_seed_results: List[Dict] = []

    for seed in seeds:
        logger.info(f"\n{'='*60}\nKG Seed {seed}\n{'='*60}")
        set_seed(seed)

        model      = model_factory()
        loader     = train_loader_factory(seed)
        kg_dataset = kg_dataset_factory()

        trainer = KGTrainer(
            model=model,
            train_loader=loader,
            kg_dataset=kg_dataset,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        result = trainer.train(seed=seed)
        per_seed_results.append(result)

        del trainer, model, loader, kg_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate metrics
    all_metrics: Dict[str, List[float]] = {}
    for result in per_seed_results:
        for k, v in result["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    mean_m = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_m  = {k: float(np.std(v))  for k, v in all_metrics.items()}

    # Save summary
    _cls_name    = model_factory().__class__.__name__.lower()
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    _save_multiseed_summary(
        model_name=_cls_name,
        dataset_name=dataset_name,
        seeds=seeds,
        mean_m=mean_m,
        std_m=std_m,
        per_seed_results=per_seed_results,
        log_dir=log_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("KG MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_m):
        logger.info(f"  {k}: {mean_m[k]:.6f} ± {std_m[k]:.6f}")
    logger.info("=" * 60)

    return {
        "per_seed": per_seed_results,
        "mean":     mean_m,
        "std":      std_m,
    }