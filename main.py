"""
main.py — v10. Unified entry point cho KG-LightGCN ICTA2026.

THAY ĐỔI v10 so với v7:
  [v10-1] Nguồn dữ liệu: KGAT repo DUY NHẤT (không dùng LightGCN/SimGCL/KGCL)
  [v10-2] Paths: /data/phuongtran/project_v10/unified/
  [v10-3] split "valid" thay cho "val"
  [v10-4] KG file: kg_final.txt (không phải kg_full.txt)
  [v10-5] ColdEvaluator: test_cold.txt + cold_item_ids.txt
  [v10-6] BẤT BIẾN: item_id == entity_id (KGAT convention)
  [v10-7] eval_protocol: full (cấu hình rõ ràng)
  [v10-8] item2entity = identity mapping (không cần lookup)

Usage:
  python main.py --model lightgcn       --dataset amazon-book --seeds 42 0 1 2 3
  python main.py --model kg_lightgcn_cl --dataset amazon-book --seeds 42
  python main.py --model all            --dataset amazon-book --seeds 42 0 1 2 3
  python main.py --model kgcl           --dataset yelp2018    --seeds 42
  python main.py --model kg_lightgcn    --dataset amazon-book --kg_type category
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from datasets.cf_dataset import CFDataset
from datasets.kg_dataset import KGDataset
from datasets.dataloader import get_cf_dataloader, get_kg_dataloader
from evaluation.evaluator import Evaluator
from evaluation.cold_evaluator import ColdEvaluator
from models.lightgcn import LightGCN
from models.simgcl import SimGCL
from models.kgat import KGAT
from models.kgcl import KGCL
from models.kg_lightgcn import KGLightGCN, KGLightGCNCL
from trainers.trainer import Trainer, run_multi_seed
from trainers.kg_trainer import KGTrainer, run_kg_multi_seed
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed, get_seeds

logger = get_logger("main")

CF_MODELS  = {"lightgcn", "simgcl"}
KG_MODELS  = {"kgat", "kgcl", "kg_lightgcn", "kg_lightgcn_cl"}
ALL_MODELS = CF_MODELS | KG_MODELS


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    dev = cfg.get("train", {}).get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def get_data_dir(cfg: dict, cold_split: Optional[str] = None) -> str:
    """[v10] Paths: /data/phuongtran/project_v10/unified/{dataset}/"""
    data_dir     = cfg.get("dataset", {}).get(
        "data_dir", "/data/phuongtran/project_v10/unified")
    dataset_name = cfg.get("dataset", {}).get("name", "amazon-book")
    base = os.path.join(data_dir, dataset_name)
    return os.path.join(base, cold_split) if cold_split else base


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_cf_dataset(data_dir: str, cfg: dict) -> CFDataset:
    seed = cfg.get("train", {}).get("seed", 42)
    return CFDataset(data_dir=data_dir, split="train", seed=seed)


def build_kg_dataset(data_dir: str, cfg: dict) -> KGDataset:
    kg_type = cfg.get("dataset", {}).get("kg_type", "full")
    seed    = cfg.get("train", {}).get("seed", 42)
    return KGDataset(
        data_dir=data_dir, split="train",
        kg_type=kg_type, seed=seed)


def build_evaluator(
    dataset: CFDataset, cfg: dict, device: torch.device
) -> Evaluator:
    """[v10] Dùng valid.txt thay val.txt."""
    train_d = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "train.txt"))
    valid_d = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "valid.txt"))   # [v10]
    test_d  = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "test.txt"))
    eval_cfg = cfg.get("eval", {})
    return Evaluator(
        train_user2items = train_d,
        valid_user2items = valid_d,      # [v10]
        test_user2items  = test_d,
        n_items          = dataset.n_items,
        device           = device,
        batch_size       = eval_cfg.get("batch_size", 2048),
        top_k_list       = eval_cfg.get("top_k", [10, 20]),
    )


# ── Model builders ────────────────────────────────────────────────────────────

def build_lightgcn(dataset, cfg, device) -> LightGCN:
    mc = cfg.get("model", {})
    return LightGCN(
        n_users=dataset.n_users, n_items=dataset.n_items,
        embedding_dim=mc.get("embedding_dim", 64),
        n_layers=mc.get("n_layers", 3),
        norm_adj=dataset.norm_adj_mat, device=device,
    ).to(device)


def build_simgcl(dataset, cfg, device) -> SimGCL:
    mc  = cfg.get("model", {})
    clc = cfg.get("contrastive", {})
    return SimGCL(
        n_users=dataset.n_users, n_items=dataset.n_items,
        embedding_dim=mc.get("embedding_dim", 64),
        n_layers=mc.get("n_layers", 3),
        eps=mc.get("eps", 0.1),
        temperature=clc.get("temperature", 0.2),
        lambda_cl=clc.get("lambda_cl", 0.5),
        apply_item_cl=mc.get("apply_item_cl", True),
        norm_adj=dataset.norm_adj_mat, device=device,
    ).to(device)


def _attach_kg_to_model(model, dataset: KGDataset, device: torch.device) -> None:
    """
    [v10] Attach normalised KG adjacency.
    KGAT convention: item_id == entity_id → KHÔNG cần item2entity mapping.
    """
    kg_norm = dataset.build_kg_norm_adj().to(device)
    model.set_kg_norm_adj(kg_norm)
    # [v10] item_id == entity_id → set_item_entity_map là no-op
    model.set_item_entity_map(None)


def build_kg_lightgcn(dataset: KGDataset, cfg, device) -> KGLightGCN:
    mc    = cfg.get("model", {})
    model = KGLightGCN(
        n_users=dataset.n_users, n_items=dataset.n_items,
        n_entities=dataset.n_entities, n_relations=dataset.n_relations,
        embedding_dim=mc.get("embedding_dim", 64),
        n_layers=mc.get("n_layers", 3),
        kg_n_layers=mc.get("kg_n_layers", 2),
        kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
        entity_agg=mc.get("entity_agg", "mean"),
        kg_reg=mc.get("kg_reg", 1e-5),
        norm_adj=dataset.norm_adj_mat, device=device,
    ).to(device)
    _attach_kg_to_model(model, dataset, device)
    return model


def build_kg_lightgcn_cl(dataset: KGDataset, cfg, device) -> KGLightGCNCL:
    mc    = cfg.get("model", {})
    clc   = cfg.get("contrastive", {})
    model = KGLightGCNCL(
        n_users=dataset.n_users, n_items=dataset.n_items,
        n_entities=dataset.n_entities, n_relations=dataset.n_relations,
        embedding_dim=mc.get("embedding_dim", 64),
        n_layers=mc.get("n_layers", 3),
        kg_n_layers=mc.get("kg_n_layers", 2),
        kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
        entity_agg=mc.get("entity_agg", "mean"),
        kg_reg=mc.get("kg_reg", 1e-5),
        cl_temp=clc.get("temperature", 0.2),
        lambda_cl=clc.get("lambda_cl", 0.5),
        eps=mc.get("eps", 0.1),
        norm_adj=dataset.norm_adj_mat, device=device,
    ).to(device)
    _attach_kg_to_model(model, dataset, device)
    return model


def build_kgat(dataset: KGDataset, cfg, device) -> KGAT:
    mc    = cfg.get("model", {})
    model = KGAT(
        n_users=dataset.n_users, n_items=dataset.n_items,
        n_entities=dataset.n_entities, n_relations=dataset.n_relations,
        embedding_dim=mc.get("embedding_dim", 64),
        relation_dim=mc.get("relation_dim", 64),
        n_layers=mc.get("n_layers", 3),
        kg_n_layers=mc.get("kg_n_layers", 2),
        agg_type=mc.get("agg_type", "bi-interaction"),
        norm_adj=dataset.norm_adj_mat,
        node_dropout=mc.get("node_dropout", 0.0),
        mess_dropout=mc.get("mess_dropout", 0.0),
        device=device,
    ).to(device)
    model.set_kg_adj(dataset.build_kg_adj_list())
    return model


def build_kgcl(dataset: KGDataset, cfg, device) -> KGCL:
    mc    = cfg.get("model", {})
    clc   = cfg.get("contrastive", {})
    model = KGCL(
        n_users=dataset.n_users, n_items=dataset.n_items,
        n_entities=dataset.n_entities, n_relations=dataset.n_relations,
        embedding_dim=mc.get("embedding_dim", 64),
        n_layers=mc.get("n_layers", 3),
        kg_n_layers=mc.get("kg_n_layers", 2),
        temp=mc.get("temp", clc.get("temperature", 0.2)),
        lambda_kg=mc.get("lambda_kg", 0.1),
        kg_p_drop=mc.get("kg_p_drop", 0.5),
        ui_p_drop=mc.get("ui_p_drop", 0.05),
        norm_adj=dataset.norm_adj_mat,
        kg_triples=dataset.kg_triples,
        device=device,
    ).to(device)
    kg_norm = dataset.build_kg_norm_adj().to(device)
    model.set_kg_norm_adj(kg_norm)
    return model


MODEL_BUILDERS = {
    "lightgcn":       (build_lightgcn,      "cf"),
    "simgcl":         (build_simgcl,         "cf"),
    "kgat":           (build_kgat,           "kg"),
    "kgcl":           (build_kgcl,           "kg"),
    "kg_lightgcn":    (build_kg_lightgcn,    "kg"),
    "kg_lightgcn_cl": (build_kg_lightgcn_cl, "kg"),
}


# ── Training entry point ──────────────────────────────────────────────────────

def train_model(
    model_name:  str,
    cfg:         dict,
    seeds:       List[int],
    cold_split:  Optional[str] = None,
) -> Dict:
    device       = get_device(cfg)
    data_dir     = get_data_dir(cfg, cold_split)
    result_dir   = cfg.get("logging", {}).get("result_dir",     "results/tables")
    ckpt_dir     = cfg.get("logging", {}).get("checkpoint_dir", "results/checkpoints")
    log_dir      = cfg.get("logging", {}).get("log_dir",        "results/logs")
    os.makedirs(result_dir, exist_ok=True)

    logger.info(
        f"[v10] Model: {model_name} | Data: {data_dir} | Device: {device}")
    logger.info(
        f"  eval_protocol: {cfg.get('eval', {}).get('eval_protocol', 'full')}")

    builder_fn, model_type = MODEL_BUILDERS[model_name]
    train_cfg = cfg.get("train", {})

    if model_type == "cf":
        dataset   = build_cf_dataset(data_dir, cfg)
        evaluator = build_evaluator(dataset, cfg, device)

        def model_factory():
            return builder_fn(dataset, cfg, device)

        def loader_factory(seed):
            return get_cf_dataloader(
                data_dir=data_dir, split="train",
                batch_size=train_cfg.get("batch_size", 2048),
                neg_samples=train_cfg.get("neg_samples", 1),
                num_workers=train_cfg.get("num_workers", 0),
                seed=seed,
            )

        results = run_multi_seed(
            model_factory=model_factory,
            train_loader_factory=loader_factory,
            evaluator=evaluator, cfg=cfg, device=device,
            seeds=seeds, checkpoint_dir=ckpt_dir, log_dir=log_dir,
        )

    else:  # KG models
        dataset   = build_kg_dataset(data_dir, cfg)
        evaluator = build_evaluator(dataset, cfg, device)

        def model_factory():
            return builder_fn(dataset, cfg, device)

        def loader_factory(seed):
            return get_kg_dataloader(
                data_dir=data_dir, split="train",
                batch_size=train_cfg.get("batch_size", 2048),
                neg_samples=train_cfg.get("neg_samples", 1),
                kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
                num_workers=train_cfg.get("num_workers", 0),
                seed=seed,
            )

        def kg_ds_factory():
            return build_kg_dataset(data_dir, cfg)

        results = run_kg_multi_seed(
            model_factory=model_factory,
            train_loader_factory=loader_factory,
            kg_dataset_factory=kg_ds_factory,
            evaluator=evaluator, cfg=cfg, device=device,
            seeds=seeds, checkpoint_dir=ckpt_dir, log_dir=log_dir,
        )

    # Lưu kết quả với per_seed
    suffix   = f"_{cold_split}" if cold_split else ""
    out_path = os.path.join(result_dir, f"{model_name}{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":      model_name,
            "dataset":    cfg.get("dataset", {}).get("name"),
            "cold_split": cold_split,
            "seeds":      seeds,
            "mean":       results["mean"],
            "std":        results["std"],
            "per_seed": [
                {
                    "seed":         r["seed"],
                    "best_epoch":   r["best_epoch"],
                    "test_metrics": {k: round(v, 6)
                                     for k, v in r["test_metrics"].items()},
                }
                for r in results["per_seed"]
            ],
            "v10_notes": {
                "data_source":       "KGAT repo (xiangwang1223)",
                "kg_file":           "kg_final.txt",
                "split_files":       "train.txt, valid.txt, test.txt",
                "kgat_invariant":    "item_id == entity_id",
                "eval_protocol":     "full",
            },
        }, f, indent=2)
    logger.info(f"Kết quả đã lưu → {out_path}")

    # Cold-start evaluation
    if cold_split is None and model_name in ALL_MODELS:
        _run_cold_eval(
            model_name=model_name, cfg=cfg, results=results,
            device=device, data_dir=data_dir,
            result_dir=result_dir, ckpt_dir=ckpt_dir,
            model_type=model_type, builder_fn=builder_fn,
        )

    return results


def _run_cold_eval(
    model_name, cfg, results, device, data_dir,
    result_dir, ckpt_dir, model_type, builder_fn,
) -> None:
    """
    [v10] Cold-start evaluation.
    Đọc từ: data_dir/cold_20/
      - cold_item_ids.txt (không phải cold_items.txt)
      - test_cold.txt     (không phải test.txt)
    """
    cold_dir = os.path.join(data_dir, "cold_20")
    if not os.path.isdir(cold_dir):
        logger.warning(
            f"cold_20/ không tìm thấy tại {cold_dir}. Bỏ qua cold eval.\n"
            "Chạy: python scripts/build_cold_split.py trước."
        )
        return

    # [v10] Kiểm tra cold_item_ids.txt (không phải cold_items.txt)
    cold_ids_path = os.path.join(cold_dir, "cold_item_ids.txt")
    test_cold_path = os.path.join(cold_dir, "test_cold.txt")
    if not os.path.exists(cold_ids_path) or not os.path.exists(test_cold_path):
        logger.warning(
            f"Thiếu cold_item_ids.txt hoặc test_cold.txt trong {cold_dir}.\n"
            "Chạy: python scripts/build_cold_split.py trước (v10 format)."
        )
        return

    dataset_name = cfg.get("dataset", {}).get("name", "amazon-book")
    ckpt_path    = os.path.join(
        ckpt_dir, dataset_name, model_name.lower().replace("-", ""), "seed42_best.pt")

    # Fallback tìm checkpoint
    ckpt_alternatives = [
        os.path.join(ckpt_dir, dataset_name, model_name, "seed42_best.pt"),
        os.path.join(ckpt_dir, dataset_name,
                     model_name.replace("_", ""), "seed42_best.pt"),
    ]
    for alt in ckpt_alternatives:
        if os.path.exists(alt):
            ckpt_path = alt
            break

    if not os.path.exists(ckpt_path):
        logger.warning(
            f"Checkpoint không tìm thấy: {ckpt_path}. Bỏ qua cold eval.")
        return

    if model_type == "cf":
        dataset = build_cf_dataset(data_dir, cfg)
    else:
        dataset = build_kg_dataset(data_dir, cfg)

    model = builder_fn(dataset, cfg, device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # [v10] ColdEvaluator dùng cold_item_ids.txt và test_cold.txt
    cold_evaluator = ColdEvaluator(
        cold_dir       = cold_dir,
        train_data_dir = data_dir,
        n_items        = dataset.n_items,
        device         = device,
    )
    cold_metrics = cold_evaluator.evaluate(model)
    logger.info(f"Cold-20 metrics ({model_name}): {cold_metrics}")

    out_path = os.path.join(result_dir, f"{model_name}_cold20_metrics.json")
    with open(out_path, "w") as f:
        json.dump(cold_metrics, f, indent=2)
    logger.info(f"Cold metrics đã lưu → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KG-LightGCN v10 — Train & Evaluate\n"
                    "Data source: KGAT repo DUY NHẤT")
    p.add_argument(
        "--model",
        choices=sorted(ALL_MODELS) + ["all"],
        default="lightgcn",
    )
    p.add_argument(
        "--dataset",
        choices=["amazon-book", "yelp2018"],
        default="amazon-book",
    )
    p.add_argument(
        "--cold_split",
        choices=["cold_10", "cold_20", "cold_30"],
        default=None,
    )
    p.add_argument(
        "--kg_type",
        choices=["full", "category", "brand", "none"],
        default=None,
    )
    p.add_argument("--seeds",         nargs="+", type=int, default=None)
    p.add_argument("--n_layers",      type=int,   default=None)
    p.add_argument("--embedding_dim", type=int,   default=None)
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--n_workers",     type=int,   default=None)
    p.add_argument("--base_config",   default="configs/base.yaml")
    p.add_argument("--override",      nargs="*",  default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg_path = (
        f"configs/model/{args.model}.yaml"
        if args.model != "all"
        and os.path.exists(f"configs/model/{args.model}.yaml")
        else None
    )

    overrides: Dict = {}
    for item in (args.override or []):
        k, _, v = item.partition("=")
        try:    v = int(v)
        except ValueError:
            try:  v = float(v)
            except ValueError: pass
        overrides[k] = v

    overrides["dataset.name"] = args.dataset
    if args.kg_type:
        overrides["dataset.kg_type"]     = args.kg_type
    if args.n_layers:
        overrides["model.n_layers"]      = args.n_layers
    if args.embedding_dim:
        overrides["model.embedding_dim"] = args.embedding_dim
    if args.weight_decay:
        overrides["train.weight_decay"]  = args.weight_decay
    if args.n_workers is not None:
        overrides["train.num_workers"]   = args.n_workers

    cfg   = load_config(
        base_path=args.base_config,
        model_config_path=model_cfg_path,
        overrides=overrides,
    )
    seeds         = args.seeds if args.seeds else get_seeds()
    models_to_run = list(ALL_MODELS) if args.model == "all" else [args.model]

    all_results = {}
    for model_name in models_to_run:
        # Kiểm tra KG file cho Yelp2018
        if args.dataset == "yelp2018" and model_name in KG_MODELS:
            kg_dir = os.path.join(
                cfg.get("dataset", {}).get(
                    "data_dir", "/data/phuongtran/project_v10/unified"),
                "yelp2018",
            )
            has_kg = os.path.exists(os.path.join(kg_dir, "kg_final.txt"))
            if not has_kg:
                logger.warning(
                    f"Bỏ qua {model_name} trên Yelp2018: "
                    "kg_final.txt chưa có. "
                    "Download từ KGAT repo và chạy preprocess.py."
                )
                continue

        logger.info(f"\n{'='*70}\nRunning: {model_name}\n{'='*70}")
        all_results[model_name] = train_model(
            model_name=model_name, cfg=cfg,
            seeds=seeds, cold_split=args.cold_split,
        )

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY (mean ± std)")
    logger.info("=" * 70)
    for name, res in all_results.items():
        logger.info(f"\n{name}:")
        for k in sorted(res.get("mean", {}).keys()):
            logger.info(f"  {k}: {res['mean'][k]:.6f} ± {res['std'][k]:.6f}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
