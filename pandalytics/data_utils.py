from collections.abc import Callable
import datetime as dt
from functools import wraps, partial
import hashlib
import json
import logging
import os

import pandas as pd

# set to "True" to use this decorator
os.environ["cache_to_disk"] = "False"

def cache_to_disk(
  fn: Callable | None = None, 
  filetype: str | None = 'parquet'
  path: str | None = "data/", 
):
    """
    Decorator used to cache function results to disk
    The function inputs can be mutable

    os.environ["cache_to_disk"] must be "True" to use this decorator
    
    Params
    ------
    fn: a callable
    filetype: type of file to write and/or read
    path: a folder to store the file
    """
    if fn is None:
        return partial(cache_to_disk, path=path, filetype=filetype)

    if not os.path.exists(path):
        os.makedirs(path)

    @wraps(fn)
    def wrap(*args, **kwargs):

        today = str(dt.datetime.now().date())
        fn_name = fn.__name__
        string_inputs = str(args) + str(kwargs)

        m = hashlib.sha256()
        m.update(string_inputs.encode("utf-8"))
        hashed_string_inputs = m.hexdigest()

        filename = f"{path}{fn_name}_{today}_{hashed_string_inputs}.{filetype}"

        if ast.literal_eval(os.environ.get("cache_to_disk", "False")) and os.path.exists(filename):
            logging.info(f"Reading cached file: {filename}")

            if filetype == "parquet":
                result = pd.read_parquet(filename)
            elif filetype == "json":
                with open(filename, "r") as fout:
                    result = json.load(fout)

        else:
            result = fn(*args, **kwargs)

            if ast.literal_eval(os.environ.get("cache_to_disk", "False")):
                logging.info(f"Caching to disk: {filename}")
                if filetype == "parquet":
                    result.to_parquet(filename)
                elif filetype == "json":
                    with open(filename, "w") as fout:
                        json.dump(result, fout)

        return result

    return wrap
