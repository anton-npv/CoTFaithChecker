import logging
from pathlib import Path
from datetime import datetime

__all__ = ["init_logger"]

FMT = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"


def init_logger(log_dir: str, script_name: str) -> logging.Logger:
    """
    Create one logger per script.
    A file called {script_name}.log is written under LOG_DIR, and INFO+ goes to the console.
    No try/except used by design.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = Path(log_dir) / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handlers during interactive reloads
    if not logger.handlers:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FMT))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(name)s — %(levelname)s — %(message)s"))
        logger.addHandler(sh)

    logger.info(f"Logger initialised; log file = {logfile}")
    return logger
