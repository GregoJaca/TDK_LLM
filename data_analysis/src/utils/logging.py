# STATUS: DONE
import logging
from config import CONFIG

def get_logger(name=__name__):
    level = getattr(logging, CONFIG["logging"]["level"])
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger
