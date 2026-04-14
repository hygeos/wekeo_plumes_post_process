"""
Module to replaces hygeos dependencies with similar interfaces
"""


def _print_raise(*args, e=None, **kwargs):
    """
    Stub for hygeos's core.log.error that prints the message and raises the exception if provided.
    """
    print(*args, **kwargs)
    if e is not None:
        raise e

debug = print
info = print
warning = print
error = _print_raise
