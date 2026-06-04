"""scripts/run_baselines.py — v10. Chạy 4 baselines bắt buộc."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger

logger = get_script_logger("run_baselines")
BASELINES = ["lightgcn", "simgcl", "kgat", "kgcl"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="amazon-book")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    p.add_argument("--base_config", default="configs/base.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info(f"=== Running 4 Baselines on {args.dataset} ===")
    for model in BASELINES:
        logger.info(f"\n{'='*60}\n{model.upper()}\n{'='*60}")
        model_cfg = f"configs/model/{model}.yaml"
        cfg = load_config(
            base_path=args.base_config,
            model_config_path=model_cfg if os.path.exists(model_cfg) else None,
            overrides={"dataset.name": args.dataset},
        )
        train_model(model_name=model, cfg=cfg, seeds=args.seeds)
    logger.info("\n✓ All 4 baselines completed.")

if __name__ == "__main__":
    main()
