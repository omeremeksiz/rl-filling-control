# utils/logging_utils.py
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_legacy_training_logger(base_dir: str = "output") -> Tuple[logging.Logger, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    training_id = f"{timestamp}_{unique_id}"
    output_dir = os.path.join(base_dir, training_id)
    _ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "training_process.log")

    logger = logging.getLogger(f"training_{training_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False

    logger.info(f"Job ID: {training_id}")
    logger.info(f"All experiment outputs will be saved in: {output_dir}")
    logger.info("")
    logger.info(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info("")
    return logger, output_dir, log_path


def get_legacy_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        'qvalue_vs_state_path': os.path.join(output_dir, "qvalue_vs_state.png"),
        'switching_point_trajectory_path': os.path.join(output_dir, "switching_point_trajectory.png"),
        'log_file_path': os.path.join(output_dir, "training_process.log"),
    }



