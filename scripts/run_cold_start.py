"""scripts/run_cold_start.py — v10 [T3.1]. Cold-start evaluation."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger

logger = get_script_logger("run_cold_start")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="amazon-book")
    p.add_argument("--levels", nargs="+", default=["cold_20"],
                   help="cold_10, cold_20, cold_30")
    p.add_argument("--models", nargs="+",
                   default=["lightgcn", "kg_lightgcn_cl"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--base_config", default="configs/base.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info(f"=== Cold-Start Evaluation v10 ===")
    logger.info(f"Dataset: {args.dataset} | Levels: {args.levels}")
    logger.info("[v10] Files: test_cold.txt, cold_item_ids.txt, valid.txt")

    for cold_split in args.levels:
        for model in args.models:
            logger.info(f"\n--- {model} @ {cold_split} ---")
            model_cfg = f"configs/model/{model}.yaml"
            cfg = load_config(
                base_path=args.base_config,
                model_config_path=model_cfg if os.path.exists(model_cfg) else None,
                overrides={"dataset.name": args.dataset},
            )
            train_model(
                model_name=model, cfg=cfg, seeds=args.seeds,
                cold_split=cold_split)
    logger.info("\n✓ Cold-start evaluation hoàn tất.")

if __name__ == "__main__":
    main()
