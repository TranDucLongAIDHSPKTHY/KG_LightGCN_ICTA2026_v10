"""scripts/run_all.py — v10. Chạy toàn bộ pipeline từ đầu."""
import argparse, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("run_all")

def run_step(cmd: list, step_name: str) -> int:
    logger.info(f"\n{'='*65}\n{step_name}\n{'='*65}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"✗ {step_name} THẤT BẠI (code={result.returncode})")
    else:
        logger.info(f"✓ {step_name} hoàn tất")
    return result.returncode

def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline v10")
    p.add_argument("--dataset", default="amazon-book")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info("=== KG-LightGCN v10 — FULL PIPELINE ===")
    logger.info(f"Dataset: {args.dataset}")

    if not args.skip_download:
        run_step([sys.executable, "scripts/download_data.py",
                  "--dataset", args.dataset],
                 "1. Download dữ liệu (KGAT repo)")

    if not args.skip_preprocess:
        run_step([sys.executable, "scripts/preprocess.py",
                  "--dataset", args.dataset],
                 "2. Preprocessing (80/10/10, KGAT format)")
        run_step([sys.executable, "scripts/build_cold_split.py",
                  "--dataset", args.dataset, "--ratio", "10", "20", "30"],
                 "3. Cold-start splits")

    run_step([sys.executable, "scripts/run_baselines.py",
              "--dataset", args.dataset, "--seeds", "42", "0", "1", "2", "3"],
             "4. Baselines (LightGCN, SimGCL, KGAT, KGCL)")

    run_step([sys.executable, "scripts/sensitivity_analysis.py",
              "--dataset", args.dataset, "--param", "n_layers"],
             "5. Sensitivity Analysis (K)")

    run_step([sys.executable, "main.py",
              "--model", "kg_lightgcn_cl",
              "--dataset", args.dataset,
              "--seeds", "42", "0", "1", "2", "3"],
             "6. KG-LightGCN-CL (Proposed)")

    run_step([sys.executable, "scripts/run_multiseed.py",
              "--compare", "kg_lightgcn_cl,lightgcn,kgcl",
              "--result_dir", "results/tables"],
             "7. Significance Tests")

    run_step([sys.executable, "scripts/diagnose_result_conflict.py",
              "--dataset", args.dataset],
             "8. Conflict Analysis [T1.4]")

    logger.info("\n✓ Full pipeline hoàn tất!")

if __name__ == "__main__":
    main()
