"""
TITAN v2.0 — Centralized Logger
"""
import logging
import os
from core.config import LOG_LEVEL

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/titan.log"),
    ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
