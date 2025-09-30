# app/scheduler.py
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from .pipeline import ingest_once, rebuild_prefs_and_vectors, scan_library_all
from .config import DISABLE_AUTOSTART

log = logging.getLogger(__name__)

def start_scheduler(interval_min: int = 60):
    sched = BackgroundScheduler()

    def run_pipeline():
        try:
            scan_library_all()
            ingest_once()
            rebuild_prefs_and_vectors()
        except Exception as e:
            log.exception("Pipeline error: %s", e)

    if not DISABLE_AUTOSTART:
        run_pipeline()

    sched.add_job(run_pipeline, "interval", minutes=interval_min, id="pipeline")
    sched.start()
    return sched