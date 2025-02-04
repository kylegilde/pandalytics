from collections.abc import Callable
import datetime as dt
from functools import wraps, partial
import hashlib
import inspect
from io import StringIO
import json
import logging
import os
from pprint import pformat
import time
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


def change_display(
    min_rows: int | None = 25,
    max_rows: int | None = 50,
    max_columns: int | None = 100,
    max_colwidth: int | None = 400,
    width: int | None = 1000,
    n_decimals: int | None = 4,
) -> pd.DataFrame:
    """
    Print more of your DataFrame by calling this function

    Parameters
    ----------
    min_rows: min number of rows
    max_rows: max number of rows
    max_columns: max number of columns
    max_colwidth: width of a column
    width: max width
    n_decimals: number of decimals to display in the float format

    Returns
    -------
    None
    """
    pd.set_option("display.min_rows", min_rows)
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("max_colwidth", max_colwidth)
    pd.set_option("display.width", width)

    format_string = "{:,.%df}" % n_decimals
    pd.set_option("display.float_format", format_string.format)

    return


def format_and_log(
    k: str, 
    v: object,
    show_data: bool | None = False,
    show_info: bool | None = True,
    sort_dicts: bool | None = False,
    n_decimals: int | None = 4,
) -> None:
    """
    Format and log keys & values in a pretty way
    """

    if isinstance(v, pd.DataFrame):
        if show_info:
            buf = StringIO()
            v.info(memory_usage="deep", buf=buf)
            msg = f"\n\n{k} info =\n{buf.getvalue()}\n"

        if show_data:
            msg = f"{k} = \n{v}"

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


def log_data(
    *args,
    title: str | None = "Logging data",
    show_data: bool | None = False,
    show_info: bool | None = True,
    sort_dicts: bool | None = False,
    n_decimals: int | None = 4,
    **kwargs,
) -> None:
    """Log variables"""
   
    if logging.INFO >= logging.root.level:
        
        if more_than_1 := (len(args) + len(kwargs) > 1):
            filler = "-" * 10
            msg = f"\n\n{filler}{title}{filler}"
            logging.info(msg)

        log_fn = partial(
            format_and_log,
            show_data=show_data,
            show_info=show_info,
            sort_dicts=sort_dicts,
            n_decimals=n_decimals,        
        )

        for arg in args:
            # if an arg is provided, use the value to get the name of the variable
            # search through the stacked scopes
            for frame_info in inspect.stack():
                for k, v in frame_info.frame.f_locals.items():
                    if  k != "arg" and id(v) == id(arg):
                        log_fn(k, v)
                        break

        for k, v in kwargs.items():
            log_fn(k, v)

        if more_than_1:
            msg = f"\n{filler}Logging done{filler}\n\n"
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
            start = time.time()
            func(**x, **kwargs)
            stop = time.time()
        else:
            start = time.time()
            func(x, **kwargs)
            stop = time.time()

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
        start = time.time()
        result = f(*args, **kwargs)
        seconds = time.time() - start

        duration = f"{seconds:,.1f} seconds" if seconds < 60 else f"{seconds / 60:,.1f} minutes"
        print(f"\n{f.__name__} ----------> {duration}\n\n")

        return result

    return wrap

def create_hashed_filename(
    fn: Callable, 
    is_method: bool | None = False,
    path: str | None = "data/", 
    filetype: Literal["parquet", "json", "npy"] | None = 'parquet',
    args: list | None = [], 
    kwargs: dict | None = {},
):
    """
    Create a hashed file path based upon the function name, date & function inputs  
    """
    today = str(dt.datetime.now().date())
    fn_name = fn.__name__
    
    if is_method:
        args = args[1:]

    string_inputs = str(args) + str(kwargs)
    
    m = hashlib.sha256()
    m.update(string_inputs.encode("utf-8"))
    hashed_string_inputs = m.hexdigest()

    return f"{path}{fn_name}_{today}_{hashed_string_inputs}.{filetype}"


def cache_to_disk_decorator(
    fn: Callable | None = None, 
    is_method: bool | None = False,
    filetype: Literal["parquet", "json", "npy"] | None = 'parquet',
    use_polars: bool | None = False,
    cache_to_disk_kwarg: str | None = "cache_to_disk",
    path: str | None = "data/", 
):
    """
    Decorator used to cache function results to disk
    The function inputs can be mutable
    
    Params
    ------
    fn: a callable
    filetype: type of file to write and/or read
    use_polars: should polars be used instead of pandas for parquets?
    cache_to_disk_kwarg: the functions kwarg that determines whether to cache to disk
    path: a folder to store the file
    """
    if fn is None:
        return partial(
            cache_to_disk_decorator, 
            is_method=is_method,
            filetype=filetype,
            use_polars=use_polars,
            cache_to_disk_kwarg=cache_to_disk_kwarg, 
            path=path,
        )

    if not os.path.exists(path):
        os.makedirs(path)

    @wraps(fn)
    def wrap(*args, **kwargs):

        if not kwargs.get(cache_to_disk_kwarg, False):
            logging.info(f"Skipping caching for {fn.__name__}  because '{cache_to_disk_kwarg}=True' was not in the kwargs")
            return fn(*args, **kwargs)

        if os.path.exists(filename := create_hashed_filename(fn, is_method, path, filetype, args, kwargs)):

            logging.info(f"Reading cached file: {filename}")
            if filetype == "parquet" and use_polars:
                return pl.read_parquet(filename)
            if filetype == "parquet":
                return pd.read_parquet(filename)
            if filetype == "json":
                with open(filename, "r") as fout:
                    return json.load(fout)
            if filetype == "npy":
                return np.load(filename)
            
        else: 
            logging.debug(f"{filename} does not exist")


        result = fn(*args, **kwargs)
        logging.info(f"Caching to disk: {filename}")
        if filetype == "parquet" and use_polars:
            result.write_parquet(filename)
        elif filetype == "parquet":
            result.to_parquet(filename)
        elif filetype == "json":
            with open(filename, "w") as fout:
                json.dump(result, fout)
        elif filetype == "npy":
            np.save(filename, result)

        return result

    return wrap
    
