from collections.abc import Callable
import datetime as dt
from functools import wraps, partial
import hashlib
import json
import logging
import os

import pandas as pd


def create_hashed_filename(
    fn: Callable, 
    path: str, 
    filetype: str,
    args: list | None = list(), 
    kwargs: dict | None = dict(),
):
    """
    Create a hashed file path based upon the function name, date & function inputs  
    """

    today = str(dt.datetime.now().date())
    fn_name = fn.__name__
    string_inputs = str(args) + str(kwargs)

    m = hashlib.sha256()
    m.update(string_inputs.encode("utf-8"))
    hashed_string_inputs = m.hexdigest()

    return f"{path}{fn_name}_{today}_{hashed_string_inputs}.{filetype}"


def cache_to_disk_decorator(
  fn: Callable | None = None, 
  filetype: str | None = 'parquet',
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
    cache_to_disk_kwarg: the functions kwarg that determines whether to cache to disk
    path: a folder to store the file
    """
    if fn is None:
        return partial(
          cache_to_disk_decorator, 
          filetype=filetype,
          cache_to_disk_kwarg=cache_to_disk_kwarg, 
          path=path,
        )

    if not os.path.exists(path):
        os.makedirs(path)

    @wraps(fn)
    def wrap(*args, **kwargs):

        if not kwargs.get(cache_to_disk_kwarg, False):
            logging.info(f"Skipping caching for {fn.__name__}")
            return fn(*args, **kwargs)

        if os.path.exists(filename := create_hashed_filename(fn, path, filetype, args, kwargs)):

            logging.info(f"Reading cached file: {filename}")
            if filetype == "parquet":
                return pd.read_parquet(filename)
            if filetype == "json":
                with open(filename, "r") as fout:
                    return json.load(fout)

        result = fn(*args, **kwargs)
        logging.info(f"Caching to disk: {filename}")
        if filetype == "parquet":
            result.to_parquet(filename)
        elif filetype == "json":
            with open(filename, "w") as fout:
                json.dump(result, fout)

        return result

    return wrap
