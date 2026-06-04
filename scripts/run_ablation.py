"""scripts/run_ablation.py — v10. Entity type ablation A1/A2/A3/A4."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger

logger = get_script_logger("run_ablation")

ABLATION = {
    "A1": "none",
    "A2": "category",
    "A3": "brand",
    "A4": "full",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="amazon-book")
    p.add_argument("--kg_types", default="none,category,brand,full")
    p.add_argument("--seeds", nargs="+", type=int, default=[42,0,1,2,3])
    p.add_argument("--base_config", default="configs/base.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    kg_types = [t.strip() for t in args.kg_types.split(",")]
    logger.info(f"=== Entity Ablation Study === Dataset: {args.dataset}")

    for model_variant in ["kg_lightgcn", "kg_lightgcn_cl"]:
        for kg_type in kg_types:
            label = next((k for k, v in ABLATION.items() if v == kg_type), kg_type)
            logger.info(f"\n--- {model_variant} | {label}: kg_type={kg_type} ---")
            model_cfg = f"configs/model/{model_variant}.yaml"
            cfg = load_config(
                base_path=args.base_config,
                model_config_path=model_cfg if os.path.exists(model_cfg) else None,
                overrides={"dataset.name": args.dataset, "dataset.kg_type": kg_type},
            )
            train_model(model_name=model_variant, cfg=cfg, seeds=args.seeds)

    logger.info("\n✓ Ablation study hoàn tất. Xem results/tables/ để tổng hợp.")

if __name__ == "__main__":
    main()
