"""
common.py — Pembantu pembaca config.yaml, path folder, dan logging untuk NusantaraBarcode.
"""

import logging
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> dict[str, Path]:
    paths = {k: PROJECT_DIR / v for k, v in cfg["paths"].items()}
    for k, p in paths.items():
        if k == "log_file":
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "results" / "trees").mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("NusantaraBarcode")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
