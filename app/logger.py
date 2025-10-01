import logging
import os
import time
from logging.handlers import RotatingFileHandler

def init_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "/data/recommenderr.log")

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"

    logging.Formatter.converter = time.gmtime
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialized level=%s file=%s", level_name, log_file)