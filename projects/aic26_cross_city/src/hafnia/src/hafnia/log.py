import logging
import os

from rich.logging import RichHandler

from hafnia import __package_name__

system_handler = RichHandler(rich_tracebacks=True, show_path=True, show_level=True)
user_handler = RichHandler(rich_tracebacks=False, show_path=False, show_level=False, log_time_format="[%X]")


def create_logger(handler: RichHandler, name: str, log_level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


sys_logger = create_logger(system_handler, f"{__package_name__}.system", os.getenv("HAFNIA_LOG", "INFO").upper())
user_logger = create_logger(user_handler, f"{__package_name__}.user", "DEBUG")
