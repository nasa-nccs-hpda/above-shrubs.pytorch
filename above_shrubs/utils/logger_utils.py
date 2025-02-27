import sys
import lightning as pl
from typing import List

__all__ = ["get_loggers"]


def get_loggers(loggers: List[str]) -> List:
    """
    Get logger functions from string evaluation.
    Args:
        loggers (List[str]): string with logger callable.
    Returns:
        List of Callables.
    """
    logger_functions = []
    for logger in loggers:
        try:
            logger_functions.append(eval(logger))
        except (NameError, AttributeError) as err:
            sys.exit(f'{err}. Accepted loggers from {pl}')
    return logger_functions
