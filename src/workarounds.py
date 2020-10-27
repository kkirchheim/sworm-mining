"""
Workarounds
"""
import logging


def disable_elsapy_logging():
    """
    Elsapy logs wherever it wants, and has no option to disable this. Running this before importing elsapy fixes this.
    """
    import elsapy.log_util
    elsapy.log_util.get_logger = lambda name: logging.getLogger(name)


def install() -> None:
    """
    Install workaround
    :return:
    """
    disable_elsapy_logging()

