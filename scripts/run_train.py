"""scripts/run_train.py — v10. Wrapper để train single model."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("run_train")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--dataset",     default="amazon-book")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--cold_split",  default=None)
    p.add_argument("--kg_type",     default=None)
    p.add_argument("--base_config", default="configs/base.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    model_cfg = f"configs/model/{args.model}.yaml"
    overrides = {"dataset.name": args.dataset}
    if args.kg_type:
        overrides["dataset.kg_type"] = args.kg_type
    cfg = load_config(
        base_path=args.base_config,
        model_config_path=model_cfg if os.path.exists(model_cfg) else None,
        overrides=overrides,
    )
    logger.info(f"Training {args.model} on {args.dataset} | seed={args.seed}")
    result = train_model(
        model_name=args.model, cfg=cfg, seeds=[args.seed],
        cold_split=args.cold_split,
    )
    logger.info(f"Result: {result.get('mean')}")

if __name__ == "__main__":
    main()
