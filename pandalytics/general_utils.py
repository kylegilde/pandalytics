from functools import partial, wraps
from time import time
from typing import Callable
import inspect


def timing(f: Callable):
    """
    A decorator that measures & prints the duration of a function or method.
    It also prints the kwargs that are passed but not the positional args.
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        print(f"\n\nCalling {f.__name__} ---------->\n\n{kwargs = }")
        start = time()
        result = f(*args, **kwargs)
        seconds = time() - start

        duration = (
            f"{seconds:,.1f} seconds"
            if seconds < 60
            else f"{seconds / 60:,.1f} minutes"
        )
        print(f"\n{f.__name__} ----------> {duration}\n\n")

        return result

    return wrap


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
