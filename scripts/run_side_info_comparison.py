"""
scripts/run_side_info_comparison.py — v10 [T5.1/T5.2/T5.3]
So sánh 3 dạng structured side information (Settings A/B/C).
⚠ Phải có Fairness Sheet được duyệt trước khi chạy script này.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_model
from utils.config import load_config
from utils.logger import get_script_logger

logger = get_script_logger("run_side_info")

# Fairness Sheet requirements
FAIRNESS_SHEET = {
    "temperature": 0.2,    # τ = 0.2 cố định
    "embedding_dim": 64,   # HARD constraint
    "optimizer": "adam",
    "seeds": [42, 0, 1, 2, 3],
    "eval_protocol": "full",
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Settings A/B/C comparison [T5.1-T5.3]")
    p.add_argument("--settings", default="A,B,C")
    p.add_argument("--dataset",  default="amazon-book")
    p.add_argument("--seeds",    nargs="+", type=int, default=[42,0,1,2,3])
    p.add_argument("--base_config", default="configs/base.yaml")
    p.add_argument(
        "--fairness_approved",
        action="store_true",
        help="Xác nhận Fairness Sheet đã được giáo viên duyệt"
    )
    return p.parse_args()

def main():
    args = parse_args()

    if not args.fairness_approved:
        logger.error(
            "DỪNG LẠI! Fairness Sheet chưa được phê duyệt.\n"
            "Thêm --fairness_approved sau khi giáo viên đã duyệt Fairness Sheet.\n"
            "Fairness Sheet requirements:\n"
            f"  {FAIRNESS_SHEET}"
        )
        return

    settings = [s.strip() for s in args.settings.split(",")]
    logger.info(f"=== Settings Comparison: {settings} ===")
    logger.info(f"Fairness Sheet: τ={FAIRNESS_SHEET['temperature']}, d={FAIRNESS_SHEET['embedding_dim']}")

    # Setting A: Flat Category (kg_type=category)
    if "A" in settings:
        logger.info("\n--- Setting A: Flat Category CL ---")
        cfg = load_config(
            base_path="configs/base.yaml",
            model_config_path="configs/model/kg_lightgcn_cl.yaml",
            overrides={
                "dataset.name": args.dataset,
                "dataset.kg_type": "category",
            },
        )
        train_model(model_name="kg_lightgcn_cl", cfg=cfg, seeds=args.seeds)

    # Setting B: Brand (proxy cho taxonomy)
    if "B" in settings:
        logger.info("\n--- Setting B: Brand/Taxonomy CL ---")
        cfg = load_config(
            base_path="configs/base.yaml",
            model_config_path="configs/model/kg_lightgcn_cl.yaml",
            overrides={
                "dataset.name": args.dataset,
                "dataset.kg_type": "brand",
            },
        )
        train_model(model_name="kg_lightgcn_cl", cfg=cfg, seeds=args.seeds)

    # Setting C: Full KG
    if "C" in settings:
        logger.info("\n--- Setting C: Full KG CL ---")
        cfg = load_config(
            base_path="configs/base.yaml",
            model_config_path="configs/model/kg_lightgcn_cl.yaml",
            overrides={
                "dataset.name": args.dataset,
                "dataset.kg_type": "full",
            },
        )
        train_model(model_name="kg_lightgcn_cl", cfg=cfg, seeds=args.seeds)

    logger.info("\n✓ Settings comparison hoàn tất.")

if __name__ == "__main__":
    main()
