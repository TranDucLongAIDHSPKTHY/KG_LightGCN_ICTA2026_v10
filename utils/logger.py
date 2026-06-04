"""utils/logger.py — Structured logging system (v10)."""
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

_CONSOLE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)
    return logger


def get_run_logger(model_name: str, dataset_name: str, seed: int,
                   base_log_dir: str = "results/logs",
                   level: int = logging.INFO) -> logging.Logger:
    logger_name = f"{model_name}_{dataset_name}_seed{seed}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)
    run_log_dir = os.path.join(base_log_dir, dataset_name, model_name, f"seed{seed}")
    os.makedirs(run_log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(run_log_dir, "train.log"), mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FILE_FMT)
    logger.addHandler(fh)
    return logger


def get_script_logger(script_name: str, base_log_dir: str = "results/logs",
                      level: int = logging.INFO) -> logging.Logger:
    logger_name = f"script_{script_name}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)
    script_log_dir = os.path.join(base_log_dir, script_name)
    os.makedirs(script_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(script_log_dir, f"run_{ts}.log"), mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FILE_FMT)
    logger.addHandler(fh)
    return logger


class EpochLogger:
    def __init__(self, run_logger, model_name, dataset_name, seed,
                 base_log_dir="results/logs"):
        self.logger = run_logger
        self._header_written = False
        run_log_dir = os.path.join(
            base_log_dir, dataset_name, model_name, f"seed{seed}")
        os.makedirs(run_log_dir, exist_ok=True)
        self._tsv_path = os.path.join(run_log_dir, "epoch_metrics.tsv")
        self._tsv_file = open(self._tsv_path, "w", newline="", encoding="utf-8")
        self._tsv_writer: Optional[csv.DictWriter] = None

    def log(self, epoch, loss, metrics, time_s=0.0):
        if not self._header_written:
            fieldnames = ["epoch", "loss"] + sorted(metrics.keys()) + ["time_s"]
            self._tsv_writer = csv.DictWriter(
                self._tsv_file, fieldnames=fieldnames, delimiter="\t")
            self._tsv_writer.writeheader()
            self._tsv_file.flush()
            self._header_written = True
        row = {"epoch": epoch, "loss": f"{loss:.6f}", "time_s": f"{time_s:.1f}"}
        row.update({k: f"{v:.6f}" for k, v in metrics.items()})
        self._tsv_writer.writerow(row)
        self._tsv_file.flush()
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        self.logger.info(
            f"[Epoch {epoch:>4}]  loss={loss:.4f}  {metrics_str}  ({time_s:.1f}s)")

    def close(self):
        if self._tsv_file and not self._tsv_file.closed:
            self._tsv_file.close()

    def __del__(self):
        self.close()

    @property
    def tsv_path(self):
        return self._tsv_path


class RunSummaryLogger:
    def __init__(self, model_name, dataset_name, seed,
                 base_log_dir="results/logs"):
        run_log_dir = os.path.join(
            base_log_dir, dataset_name, model_name, f"seed{seed}")
        os.makedirs(run_log_dir, exist_ok=True)
        self._path = os.path.join(run_log_dir, "summary.json")
        self.model_name   = model_name
        self.dataset_name = dataset_name
        self.seed         = seed

    def save(self, best_epoch, val_metric, test_metrics, total_time_s, extra=None):
        data = {
            "model":            self.model_name,
            "dataset":          self.dataset_name,
            "seed":             self.seed,
            "best_epoch":       best_epoch,
            "val_best_metric":  round(val_metric, 6),
            "test_metrics":     {k: round(v, 6) for k, v in test_metrics.items()},
            "total_time_s":     round(total_time_s, 1),
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            data.update(extra)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def path(self):
        return self._path
