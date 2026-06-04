"""utils/config.py — Config loader với .env + override support (v10)."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _resolve_paths(cfg: dict) -> dict:
    """[v10] Resolve paths từ environment variables."""
    raw_dir    = os.getenv("RAW_DIR",      "/data/phuongtran/project_v10/raw")
    data_dir   = os.getenv("DATA_DIR",     "/data/phuongtran/project_v10/unified")
    results_root = os.getenv("RESULTS_ROOT")
    num_workers  = os.getenv("NUM_WORKERS")
    device       = os.getenv("DEVICE")

    if raw_dir:
        cfg.setdefault("dataset", {})["raw_dir"]  = raw_dir
    if data_dir:
        cfg.setdefault("dataset", {})["data_dir"] = data_dir
    if results_root:
        cfg.setdefault("logging", {}).update({
            "log_dir":        os.path.join(results_root, "logs"),
            "checkpoint_dir": os.path.join(results_root, "checkpoints"),
            "result_dir":     os.path.join(results_root, "tables"),
        })
    if num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = int(num_workers)
    if device:
        cfg.setdefault("train", {})["device"] = device
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    base_path: str = "configs/base.yaml",
    model_config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dotenv_path: str = ".env",
) -> dict:
    _load_dotenv(dotenv_path)
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, model_cfg)
    cfg = _resolve_paths(cfg)
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    return cfg


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False,
                  sort_keys=True, allow_unicode=True)


class Config:
    def __init__(self, data: dict) -> None:
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v.to_dict() if isinstance(v, Config) else v
        return result
