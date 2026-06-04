"""
trainers/trainer.py — v10
Trainer cơ sở cho tất cả models.

THAY ĐỔI v10:
  - split "valid" thay cho "val"
  - eval protocol: full-item ranking (eval_protocol: full)
  - Gradient clipping cho TẤT CẢ models
  - per_seed results lưu trong JSON cho significance test
"""
import gc
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from evaluation.evaluator import Evaluator
from losses.bpr_loss import bpr_loss
from losses.contrastive_loss import infonce_loss
from utils.logger import (get_logger, get_run_logger,
                           EpochLogger, RunSummaryLogger)
from utils.seed import set_seed

logger = get_logger("trainer")


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        evaluator:      Evaluator,
        cfg:            dict,
        device:         torch.device,
        checkpoint_dir: str = "results/checkpoints",
        log_dir:        str = "results/logs",
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.evaluator      = evaluator
        self.cfg            = cfg
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir        = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir,        exist_ok=True)

        train_cfg = cfg.get("train",       {})
        eval_cfg  = cfg.get("eval",        {})
        log_cfg   = cfg.get("logging",     {})
        cl_cfg    = cfg.get("contrastive", {})

        self.lr             = float(train_cfg.get("learning_rate",          1e-3))
        self.weight_decay   = float(train_cfg.get("weight_decay",           1e-4))
        self.epochs         = int(train_cfg.get("epochs",                  1000))
        self.patience       = int(train_cfg.get("early_stopping_patience",   10))
        self.monitor_metric = train_cfg.get("early_stopping_metric", "recall@20")
        self.max_grad_norm  = float(train_cfg.get("max_grad_norm",           1.0))
        self.manual_l2_reg  = float(train_cfg.get("manual_l2_reg",          0.0))
        self.eval_interval  = int(eval_cfg.get("eval_interval",              5))
        self.log_interval   = int(log_cfg.get("log_interval",               1))
        self.temperature    = float(cl_cfg.get("temperature",               0.2))
        self.lambda_cl      = float(cl_cfg.get("lambda_cl",                 0.5))
        self.num_workers     = int(train_cfg.get("num_workers",             0))

        self.model_name   = self.model.__class__.__name__.lower()
        self.dataset_name = cfg.get("dataset", {}).get("name", "unknown")

        # weight_decay=0 trong Adam; manual L2 thêm vào loss nếu cần
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        self._is_simgcl = (
            hasattr(model, "contrastive_loss")
            and not hasattr(model, "kg_forward")
            and not hasattr(model, "kg_alignment_loss")
        )

    def train(self, seed: int = 42) -> Dict[str, Any]:
        set_seed(seed)
        t_start = time.time()

        run_logger     = get_run_logger(
            self.model_name, self.dataset_name, seed,
            base_log_dir=self.log_dir)
        epoch_logger   = EpochLogger(
            run_logger, self.model_name, self.dataset_name,
            seed, base_log_dir=self.log_dir)
        summary_logger = RunSummaryLogger(
            self.model_name, self.dataset_name, seed,
            base_log_dir=self.log_dir)

        patience_window = self.patience * self.eval_interval
        run_logger.info("=" * 65)
        run_logger.info(
            f"  MODEL   : {self.model_name} | DATASET: {self.dataset_name}")
        
        run_logger.info(
            f"  SEED    : {seed} | DEVICE: {self.device}")
        
        run_logger.info(
            f"  EPOCHS  : {self.epochs} | PATIENCE: {self.patience} "
            f"(window={patience_window})")
        
        run_logger.info(f"  MONITOR : {self.monitor_metric}")

        run_logger.info(
            f"  LR      : {self.lr} | WD: {self.weight_decay}")
        if hasattr(self.model, 'kg_n_layers'):
            run_logger.info(f"  KG_LAYERS: {self.model.kg_n_layers}")

        run_logger.info(f"  WORKERS: {self.num_workers}")

        run_logger.info(f"  PARAMS  : {self.model.parameter_count():,}")
        run_logger.info("=" * 65)

        best_metric  = -float("inf")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None
        history:     List[Dict] = []
        running_loss = running_n = 0.0
        start_epoch  = 1
        epoch_times: List[float] = []

        ckpt_path = self._get_resume_path(seed)
        if os.path.exists(ckpt_path):
            try:
                resume_info  = self._load_checkpoint_for_resume(ckpt_path)
                start_epoch  = resume_info["epoch"] + 1
                best_metric  = resume_info.get("best_metric",  best_metric)
                best_epoch   = resume_info.get("best_epoch",   resume_info["epoch"])
                patience_ctr = resume_info.get("patience_ctr", 0)
                history      = resume_info.get("history",      [])
                run_logger.info(
                    f"  RESUMED từ epoch {resume_info['epoch']}, "
                    f"best={best_metric:.6f}")
            except Exception as e:
                run_logger.warning(f"  Không thể resume ({e}). Bắt đầu mới.")
                start_epoch = 1

        for epoch in range(start_epoch, self.epochs + 1):
            t0      = time.time()
            loss    = self._train_one_epoch()
            elapsed = time.time() - t0
            running_loss += loss
            running_n    += 1

            if epoch % self.log_interval == 0:
                epoch_times.append(elapsed)
                avg_t = sum(epoch_times[-10:]) / len(epoch_times[-10:])
                eta   = time.strftime(
                    "%H:%M:%S", time.gmtime(avg_t * (self.epochs - epoch)))
                run_logger.info(
                    f"  [Epoch {epoch}/{self.epochs}] loss={loss:.4f} | "
                    f"{elapsed:.1f}s | ETA={eta}")

            if epoch % self.eval_interval == 0:
                # [v10] Dùng split "valid" thay cho "val"
                val_metrics  = self.evaluator.evaluate(self.model, split="valid")
                monitor_val  = val_metrics.get(self.monitor_metric, 0.0)
                avg_loss     = running_loss / max(running_n, 1)
                running_loss = running_n = 0.0
                epoch_logger.log(epoch, avg_loss, val_metrics, time_s=elapsed)
                history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if monitor_val > best_metric:
                    best_metric  = monitor_val
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = deepcopy(self.model.state_dict())
                    self._save_best(
                        seed, epoch, val_metrics, best_metric,
                        best_epoch, patience_ctr, history)
                    run_logger.info(
                        f"  *** New best @ epoch {epoch}: "
                        f"{self.monitor_metric}={best_metric:.6f} ***")
                else:
                    patience_ctr += 1
                    self._save_resume(
                        seed, epoch, val_metrics, best_metric,
                        best_epoch, patience_ctr, history)
                    if patience_ctr >= self.patience:
                        run_logger.info(
                            f"Early stopping @ epoch {epoch}. "
                            f"Best: epoch={best_epoch}, "
                            f"{self.monitor_metric}={best_metric:.6f}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_metrics = self.evaluator.evaluate(self.model, split="test")
        total_time   = time.time() - t_start

        run_logger.info("-" * 65)
        run_logger.info(f"FINAL TEST (best_epoch={best_epoch})")
        for k, v in sorted(test_metrics.items()):
            run_logger.info(f"  {k:<20} = {v:.6f}")
        run_logger.info(
            f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        run_logger.info("=" * 65)

        summary_logger.save(
            best_epoch=best_epoch, val_metric=best_metric,
            test_metrics=test_metrics, total_time_s=total_time)
        epoch_logger.close()

        return {
            "seed":         seed,
            "best_epoch":   best_epoch,
            "val_metric":   best_metric,
            "test_metrics": test_metrics,
            "history":      history,
        }

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = n_batches = 0.0

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            if self._is_simgcl:
                output   = self.model(users, pos_items, neg_items)
                user_emb, pos_emb, neg_emb = output[0], output[1], output[2]
                view1, view2 = output[3], output[4]
                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                cl_total = infonce_loss(view1, view2, self.temperature)
                if len(output) == 7:
                    item_cl  = infonce_loss(output[5], output[6], self.temperature)
                    cl_total = cl_total + item_cl
                l2   = self.model.l2_loss(users, pos_items, neg_items)
                loss = rec_loss + self.lambda_cl * cl_total + self.manual_l2_reg * l2
            else:
                user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
                loss = (
                    bpr_loss(user_emb, pos_emb, neg_emb)
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

        return total_loss / max(n_batches, 1)

    def _ckpt_dir(self, seed):
        d = os.path.join(
            self.checkpoint_dir, self.dataset_name, self.model_name)
        os.makedirs(d, exist_ok=True)
        return d

    def _get_best_path(self, seed):
        return os.path.join(self._ckpt_dir(seed), f"seed{seed}_best.pt")

    def _get_resume_path(self, seed):
        return os.path.join(self._ckpt_dir(seed), f"seed{seed}_resume.pt")

    def _build_ckpt(self, seed, epoch, metrics, best_metric,
                    best_epoch, patience_ctr, history):
        return {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":              metrics,
            "best_metric":          best_metric,
            "best_epoch":           best_epoch,
            "patience_ctr":         patience_ctr,
            "history":              history or [],
            "seed":                 seed,
            "model_name":           self.model_name,
            "dataset_name":         self.dataset_name,
        }

    def _save_best(self, seed, epoch, metrics, best_metric,
                   best_epoch, patience_ctr, history):
        torch.save(
            self._build_ckpt(seed, epoch, metrics, best_metric,
                             best_epoch, patience_ctr, history),
            self._get_best_path(seed),
        )
        self._save_resume(seed, epoch, metrics, best_metric,
                          best_epoch, patience_ctr, history)

    def _save_resume(self, seed, epoch, metrics, best_metric,
                     best_epoch, patience_ctr, history):
        ckpt = self._build_ckpt(
            seed, epoch, metrics, best_metric, best_epoch, patience_ctr, history)
        best_path = self._get_best_path(seed)
        if os.path.exists(best_path):
            try:
                ckpt["best_model_state_dict"] = torch.load(
                    best_path, map_location=self.device, weights_only=False,
                ).get("model_state_dict")
            except Exception:
                pass
        torch.save(ckpt, self._get_resume_path(seed))

    def _load_checkpoint_for_resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if (ckpt.get("model_name") and ckpt["model_name"] != self.model_name):
            raise ValueError(
                f"model_name mismatch: {ckpt['model_name']} vs {self.model_name}")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                logger.warning("Không restore được optimizer state.")
        return ckpt


def run_multi_seed(
    model_factory,
    train_loader_factory,
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
        logger.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        set_seed(seed)
        model   = model_factory()
        loader  = train_loader_factory(seed)
        trainer = Trainer(
            model=model, train_loader=loader, evaluator=evaluator,
            cfg=cfg, device=device,
            checkpoint_dir=checkpoint_dir, log_dir=log_dir,
        )
        per_seed_results.append(trainer.train(seed=seed))
        del trainer, model, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics: Dict[str, List[float]] = {}
    for res in per_seed_results:
        for k, v in res["test_metrics"].items():
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
    logger.info("MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_m):
        logger.info(f"  {k}: {mean_m[k]:.6f} ± {std_m[k]:.6f}")
    logger.info("=" * 60)
    return {"per_seed": per_seed_results, "mean": mean_m, "std": std_m}


def _save_multiseed_summary(
    model_name, dataset_name, seeds, mean_m, std_m,
    per_seed_results, log_dir,
) -> None:
    d = os.path.join(log_dir, dataset_name, model_name)
    os.makedirs(d, exist_ok=True)
    data = {
        "model": model_name, "dataset": dataset_name, "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_m.items()},
        "std":  {k: round(v, 6) for k, v in std_m.items()},
        "mean_std_str": {k: f"{mean_m[k]:.4f}±{std_m[k]:.4f}" for k in mean_m},
        "per_seed": [
            {
                "seed":       r["seed"],
                "best_epoch": r["best_epoch"],
                "test_metrics": {k: round(v, 6)
                                 for k, v in r["test_metrics"].items()},
            }
            for r in per_seed_results
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(d, "multiseed_summary.json"), "w") as f:
        json.dump(data, f, indent=2)
