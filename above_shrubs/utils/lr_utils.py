import sys
import torch
import lightning as pl

__all__ = ["get_lr_scheduler"]


def get_lr_scheduler(lr_scheduler: str) -> str:
    """
    Get callback functions from string evaluation.
    Args:
        callbacks (List[str]): string with optimizer callable.
    Returns:
        List of Callables.
    """
    try:
        lr_scheduler_function = eval(lr_scheduler)
    except (NameError, AttributeError) as err:
        sys.exit(f'{err}. Accepted lr_scheduler from {pl}')
    return lr_scheduler_function
