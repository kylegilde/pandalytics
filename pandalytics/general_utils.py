import inspect
from collections.abc import Callable
from functools import partial, wraps
from time import time
import logging
from io import StringIO

import pandas as pd
from tqdm import tqdm


def log_data(
    *args,
    show_data: bool | None = False,
    show_info: bool | None = True,
    sort_dicts: bool | None = False,
    n_decimals: int | None = 4,
    **kwargs,
) -> None:
    if logging.INFO >= logging.root.level:
        assert len(args) == 0, "log_data doesn't accept any args"

        if len(kwargs) > 1:
            filler = "-" * 10
            msg = f"\n\n{filler}Logging some data{filler}"
            logging.info(msg)

        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                if show_info:
                    buf = StringIO()
                    v.info(memory_usage="deep", buf=buf)
                    msg = f"\n\n{k} info =\n{buf.getvalue()}\n"
                    logging.info(msg)

                if show_data:
                    msg = f"{k} = \n{v}"
                    logging.info(msg)

            else:
                if isinstance(v, pd.Series):
                    msg = f"{k} =\n{v}"
                elif isinstance(v, int):
                    msg = f"{k} = {v:,}"
                elif isinstance(v, float):
                    number_format = f",.{n_decimals}f"
                    msg = f"{k} = {v:{number_format}}"
                elif isinstance(v, dict):
                    dict_string = pformat(v, sort_dicts=sort_dicts)
                    msg = f"{k} =\n{dict_string}"
                else:
                    msg = f"{k} = {v}"
                logging.info(msg)

    return


def log_data_and_return(a_variable: object, *args, **kwargs):
    log_data(a_variable=a_variable, *args, **kwargs)
    return a_variable


def get_time_trials(
    arr: pd.Series,
    func: Callable,
    pass_as_kwargs: bool | None = False,
    s_name: str | None = "",
    weights: pd.Series | None = None,
    percentiles: list | tuple = (0.75, 0.95, 0.97, 0.99, 0.999),
    **kwargs,
):
    """
    Get the percentiles of your function's duration in milliseconds

    """
    arr = pd.Series(arr)

    times = []
    for x in tqdm(arr):
        if pass_as_kwargs:
            start = time()
            func(**x, **kwargs)
            stop = time()
        else:
            start = time()
            func(x, **kwargs)
            stop = time()

        times.append(stop - start)

    if isinstance(weights, pd.Series):
        assert len(weights) == len(arr), "These lengths are not matching"

        print("Using weights")
        times_weighted = []
        for t, w in zip(times, weights):
            times_weighted.extend([t] * w)

        print(f"{len(times_weighted)=:,}")

        times_weighted = pd.Series(times_weighted)

    else:
        times_weighted = pd.Series(times)

    times = pd.Series(times)
    longest_item_idx = times.idxmax()
    longest_item = arr.iloc[longest_item_idx]

    print(f"{longest_item=}")

    return (
        times_weighted
        .mul(1000)
        .describe(percentiles=percentiles)
        .rename(s_name)
        .to_frame()
    )


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

        duration = f"{seconds:,.1f} seconds" if seconds < 60 else f"{seconds / 60:,.1f} minutes"
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
