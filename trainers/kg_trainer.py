"""
trainers/kg_trainer.py — v10
KGTrainer cho các mô hình KG (KGAT, KGCL, KG-LightGCN).
THAY ĐỔI v10: split "valid" thay cho "val".
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
            model=model, train_loader=train_loader, evaluator=evaluator,
            cfg=cfg, device=device,
            checkpoint_dir=checkpoint_dir, log_dir=log_dir,
        )
        self._log_kg_info = True

        self.kg_dataset = kg_dataset

        self._is_kgat = hasattr(model, "kg_forward")
        self._is_kg_lightgcn_cl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "kg_alignment_loss")
            and hasattr(model, "cl_temp")
        )
        self._is_kgcl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "set_kg_norm_adj")
            and not self._is_kg_lightgcn_cl
        )
        self._is_kg_lightgcn = (
            hasattr(model, "kg_alignment_loss")
            and not self._is_kg_lightgcn_cl
        )

        model_cfg = cfg.get("model", {})
        self.lambda_kg = float(model_cfg.get("kg_reg", 1e-5))

        # lambda_cl từ contrastive config
        cl_cfg = cfg.get("contrastive", {})
        self.lambda_cl = float(cl_cfg.get("lambda_cl", 0.1))

        self._cached_entity_emb: Optional[torch.Tensor] = None

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = n_batches = 0.0

        if self._is_kgat:
            self._free_entity_emb_cache()
            with torch.no_grad():
                self._cached_entity_emb = (
                    self.model._compute_entity_embeddings().detach())

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
                u, p, n = self.model(users, pos_items, neg_items)
                loss = (bpr_loss(u, p, n)
                        + self.weight_decay * self.model.l2_loss(
                            users, pos_items, neg_items))

            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if self._is_kgat:
            self._free_entity_emb_cache()
        return total_loss / max(n_batches, 1)

    def _free_entity_emb_cache(self):
        if self._cached_entity_emb is not None:
            del self._cached_entity_emb
            self._cached_entity_emb = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _kgat_step(self, users, pos_items, neg_items):
        u, p, n = self.model(
            users, pos_items, neg_items,
            precomputed_entity_emb=self._cached_entity_emb)
        cf_loss = bpr_loss(u, p, n)
        l2      = self.model.l2_loss(users, pos_items, neg_items)
        kg_loss = torch.tensor(0.0, device=self.device)
        if self.kg_dataset.kg_triples is not None:
            triples = self.kg_dataset.sample_kg_triples(len(users))
            if triples is not None:
                h, r, tp, tn = triples
                h  = torch.tensor(h,  dtype=torch.long, device=self.device)
                r  = torch.tensor(r,  dtype=torch.long, device=self.device)
                tp = torch.tensor(tp, dtype=torch.long, device=self.device)
                tn = torch.tensor(tn, dtype=torch.long, device=self.device)
                pos_score, neg_score = self.model.kg_forward(h, r, tp, tn)
                kg_loss = kg_bpr_loss(pos_score, neg_score)
        return cf_loss + self.lambda_kg * kg_loss + self.weight_decay * l2

    def _kgcl_step(self, users, pos_items, neg_items):
        u, p, n, u1, u2, i1, i2 = self.model(users, pos_items, neg_items)
        cf_loss = bpr_loss(u, p, n)
        l2      = self.model.l2_loss(users, pos_items, neg_items)
        user_cl = self.model.contrastive_loss(u1, u2)
        item_cl = self.model.contrastive_loss(i1, i2)
        cl_loss = user_cl + item_cl
        return cf_loss + self.lambda_cl * cl_loss + self.weight_decay * l2

    def _kg_lightgcn_cl_step(self, users, pos_items, neg_items):
        u, p, n, u_cf, u_kg, i_cf, i_kg = self.model(users, pos_items, neg_items)
        cf_loss    = bpr_loss(u, p, n)
        user_cl    = self.model.contrastive_loss(u_cf, u_kg)
        item_cl    = self.model.contrastive_loss(i_cf, i_kg)
        cl_loss    = (user_cl + item_cl) / 2.0
        align_loss = self.model.kg_alignment_loss()
        l2         = self.model.l2_loss(users, pos_items, neg_items)
        return (cf_loss
                + self.lambda_cl * cl_loss
                + self.lambda_kg * align_loss
                + self.weight_decay * l2)

    def _kg_lightgcn_step(self, users, pos_items, neg_items):
        u, p, n    = self.model(users, pos_items, neg_items)
        cf_loss    = bpr_loss(u, p, n)
        align_loss = self.model.kg_alignment_loss()
        l2         = self.model.l2_loss(users, pos_items, neg_items)
        return cf_loss + self.lambda_kg * align_loss + self.weight_decay * l2


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
    if seeds is None:
        seeds = [42, 0, 1, 2, 3]

    per_seed_results = []
    for seed in seeds:
        logger.info(f"\n{'='*60}\nKG Seed {seed}\n{'='*60}")
        set_seed(seed)
        model      = model_factory()
        loader     = train_loader_factory(seed)
        kg_dataset = kg_dataset_factory()

        trainer = KGTrainer(
            model=model, train_loader=loader, kg_dataset=kg_dataset,
            evaluator=evaluator, cfg=cfg, device=device,
            checkpoint_dir=checkpoint_dir, log_dir=log_dir,
        )
        result = trainer.train(seed=seed)
        per_seed_results.append(result)
        del trainer, model, loader, kg_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics: Dict[str, List[float]] = {}
    for result in per_seed_results:
        for k, v in result["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    mean_m = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_m  = {k: float(np.std(v))  for k, v in all_metrics.items()}

    _cls_name    = model_factory().__class__.__name__.lower()
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    _save_multiseed_summary(
        model_name=_cls_name, dataset_name=dataset_name,
        seeds=seeds, mean_m=mean_m, std_m=std_m,
        per_seed_results=per_seed_results, log_dir=log_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("KG MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_m):
        logger.info(f"  {k}: {mean_m[k]:.6f} ± {std_m[k]:.6f}")
    logger.info("=" * 60)
    return {"per_seed": per_seed_results, "mean": mean_m, "std": std_m}
