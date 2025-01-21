import ast
from collections.abc import Callable
import inspect
import logging
import sys


def parse_command_line_arguments(separator: str | None = "=") -> dict:
    """
    Dynamically parse any command line arguments into a dictionary with the correct data types
    
    Parameters
    ----------
    separator: the string value separating the key and value
    
    Returns
    -------
    a dict containing the key-value pairs
    """
    args = sys.argv[1:]  # Exclude the script name
    arg_dict = {}

    if args:
        logging.info(f"{args = }")

        for arg in args:
            assert arg.startswith("--") and "=" in arg, f"{arg} must start with -- and contain an ="
            key, value = arg[2:].split(separator, 1)
    
            try:
                value = ast.literal_eval(value)
            except ValueError:
                pass
    
            arg_dict[key] = value
            logging.info(f"{key}: {value} ({type(value)})")
            # logging.info(f"{type(value) = }")
    
        logging.info(f"{arg_dict = }")
    else:
        logging.info("No command line arguments")
    
    return arg_dict

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
