import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def init_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "/data/recommenderr.log")

    class UtcFormatter(logging.Formatter):
        converter = staticmethod(lambda *args: datetime.utcnow().timetuple())

    fmt = "%Y-%m-%dT%H:%M:%SZ %(levelname)s [%(name)s] %(message)s"

    sh = logging.StreamHandler()
    sh.setFormatter(UtcFormatter(fmt))
    sh.setLevel(level)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(UtcFormatter(fmt))
    fh.setLevel(level)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialized level=%s file=%s", level_name, log_file)