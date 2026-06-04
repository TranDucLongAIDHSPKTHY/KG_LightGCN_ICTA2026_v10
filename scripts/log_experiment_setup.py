"""
scripts/log_experiment_setup.py — v10 [T3.4]
Tự động sinh file results/experimental_setup.md từ config và môi trường.
"""
import os, sys, json, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("log_experiment_setup")

MODELS = ["lightgcn", "simgcl", "kgat", "kgcl", "kg_lightgcn", "kg_lightgcn_cl"]
LABELS = {
    "lightgcn":       "LightGCN",
    "simgcl":         "SimGCL",
    "kgat":           "KGAT",
    "kgcl":           "KGCL",
    "kg_lightgcn":    "KG-LightGCN",
    "kg_lightgcn_cl": "KG-LightGCN-CL",
}
HP = {
    "lightgcn":       {"lr": "0.001", "wd": "1e-4", "K": "3", "d": "64"},
    "simgcl":         {"lr": "0.001", "wd": "0",    "K": "3", "d": "64"},
    "kgat":           {"lr": "1e-4",  "wd": "1e-5", "K": "3", "d": "64"},
    "kgcl":           {"lr": "0.001", "wd": "1e-4", "K": "3", "d": "64"},
    "kg_lightgcn":    {"lr": "0.001", "wd": "1e-4", "K": "K*","d": "64"},
    "kg_lightgcn_cl": {"lr": "0.001", "wd": "1e-4", "K": "K*","d": "64"},
}

def get_hardware_info():
    info = {"OS": platform.system() + " " + platform.release(),
            "Python": platform.python_version()}
    try:
        import torch
        info["PyTorch"] = torch.__version__
        if torch.cuda.is_available():
            info["GPU"]  = torch.cuda.get_device_name(0)
            info["VRAM"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        else:
            info["GPU"] = "CPU only"
    except ImportError:
        pass
    try:
        import psutil
        info["RAM"] = f"{psutil.virtual_memory().total / 1e9:.1f} GB"
    except ImportError:
        info["RAM"] = "N/A (pip install psutil)"
    return info

def get_best_hparams(result_dir):
    """Load K* từ sensitivity results nếu có."""
    sens_path = os.path.join(result_dir, "sensitivity_results.md")
    if os.path.exists(sens_path):
        with open(sens_path) as f:
            content = f.read()
        # Tìm K*
        import re
        m = re.search(r"K tốt nhất\s*=\s*(\d+)", content)
        if m:
            return {"K_best": m.group(1)}
    return {"K_best": "Chưa có — chạy sensitivity_analysis.py trước"}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="results/tables")
    p.add_argument("--output", default="results/experimental_setup.md")
    args = p.parse_args()

    hw = get_hardware_info()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best = get_best_hparams(args.result_dir)

    lines = [
        "# Experimental Setup — KG-LightGCN v10\n",
        "## Hardware\n",
    ]
    for k, v in hw.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    lines.append("## Hyperparameters\n")
    lines.append("| Model | lr | weight_decay | batch_size | K | d | neg_ratio |")
    lines.append("|-------|-----|--------------|------------|---|---|-----------|")
    for model in MODELS:
        hp = HP[model]
        K  = f"{hp['K']}={best['K_best']}" if hp['K'] == 'K*' else hp['K']
        lines.append(
            f"| {LABELS[model]} | {hp['lr']} | {hp['wd']} | 2048 | {K} | 64 | 1 |")
    lines.append("")
    lines.append("## Protocol\n")
    lines.append("- **Evaluation:** Full-item ranking (eval_protocol: full)")
    lines.append("  - Candidate pool = tất cả items chưa tương tác trong training")
    lines.append("  - Sampled ranking chỉ dùng cho Appendix")
    lines.append("- **Data source:** KGAT repo (xiangwang1223)")
    lines.append("  - KHÔNG dùng LightGCN/SimGCL/KGCL repo")
    lines.append("- **Split:** Train 80% / Valid 10% / Test 10%")
    lines.append("  - Gộp KGAT train+test rồi chia lại, seed=42")
    lines.append("  - BẤT BIẾN: item_id == entity_id (KGAT convention)")
    lines.append("- **Seeds:** {42, 0, 1, 2, 3} → report mean ± std")
    lines.append("- **Negative sampling:** uniform random, 1 negative per positive")
    lines.append("- **Gradient clipping:** max_grad_norm=1.0 (tất cả models)")
    lines.append("")
    lines.append("## Note\n")
    lines.append(
        f"- K* = K tốt nhất từ sensitivity analysis: **{best['K_best']}**")
    lines.append(
        "- Tất cả KG models đọc kg_final.txt từ KGAT repo (KHÔNG build lại KG)")
    lines.append("- Cold-start: induced cold-start, seed=42, tương thích TaxPro-CL")

    content = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"✓ experimental_setup.md → {args.output}")
    print(content)

if __name__ == "__main__":
    main()
