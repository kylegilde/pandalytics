from functools import partial
from typing import Callable
import inspect


def safe_partial(func: Callable, *args, **kwargs):
    """
    Allows you to safely pass kwarg dictionaries to functions or methods that validate whether
    the dict keys are proper function parameters (i.e. don't have **kwargs)

    Parameters
    ----------
    func: a callable
    args: a list of values to pass by position
    kwargs: key-value pairs

    Returns
    -------
    the function output

    """
    func_parameters = inspect.signature(func).parameters

    if "kwargs" not in func_parameters:
        kwargs = {k: v for k, v in kwargs.items() if k in func_parameters}

    return partial(func, *args, **kwargs)


def replace_none(variable: object, replacement_value: object):
    """
    If a variable is a None value, replace it with something else
    Parameters
    ----------
    variable: an object
    replacement_value

    Returns
    -------
    replacement_value if variable is None, otherwise variable

    """
    return replacement_value if variable is None else variable