from functools import partial, wraps
from time import time
from typing import Callable, Union, List, Tuple, Optional
import inspect

from tqdm import tqdm
import pandas as pd


def get_time_trials(
    arr: pd.Series, 
    func: Callable, 
    s_name: Optional[str] = "", 
    weights: Optional[pd.Series] = None,
    percentiles: Union[List, Tuple] = (.75, .95, .97, .99, .999),
    **kwargs
):
    """
    Get the percentiles of your function's duration in milliseconds
    
    """
    arr = pd.Series(arr)
    
    times = []
    for x in tqdm(arr):
        if isinstance(x, dict):
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
    longest_item  = arr.iloc[longest_item_idx]
    
    print(f"{longest_item=}")
    
    return (
        times_weighted.mul(1000)
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
