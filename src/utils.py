"""
Helper functions
"""
import json
import logging
import os
import time
from datetime import datetime

from constants import *

log = logging.getLogger(__name__)


def timed(function):
    """
    Decorator, measures time of the given function while executing.
    :param function:
    :return:
    """

    def inner(*args, **kwargs):
        t1 = time.perf_counter()
        ret = function(*args, **kwargs)
        t2 = time.perf_counter()
        log.info(f"Took: {t2 - t1:.2f} s")
        return ret

    return inner


def disable_elsapy_logging():
    """
    Elsapy logs wherever it wants, and has no option to disable this. Running this before importing elsapy fixes this.
    """
    import elsapy.log_util
    elsapy.log_util.get_logger = lambda name: logging.getLogger(name)


def install_elsapy_workarounds() -> None:
    """
    Install workarounds for the eslapy package.
    :return:
    """
    disable_elsapy_logging()


def read_resource(path):
    lines = []
    with open(path, "r") as f:
        for line in f.readline():
            lines.append(line.strip())


def configure_logging(path=None):
    if path is None:
        path = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d-%H-%M-%S-%f')}.log")

    fmt = "[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
    formatter = logging.Formatter(fmt=fmt)
    logging.basicConfig(filename=path, level=logging.INFO, format=fmt)

    root = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    root.setLevel(logging.DEBUG)
    root.debug("Logging configured")


def get_elsa_config():
    """Load configuration for scopus scraping"""
    con_file = open(CONFIG_FILE)
    config = json.load(con_file)
    con_file.close()
    return config


def setup_nltk_cache():
    import nltk
    nltk.data.path.append(CACHE_DIR)
    nltk.download('wordnet', download_dir=CACHE_DIR)
    nltk.download('punkt', download_dir=CACHE_DIR)
    nltk.download('stopwords', download_dir=CACHE_DIR)
