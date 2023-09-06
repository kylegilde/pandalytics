from typing import Optional, List, Tuple, Union, Callable, Literal
from dataclasses import dataclass
from functools import partial
import humanize

import pandas as pd


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get the size of the DataFrame in human-readable format
    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    string
    """
    return humanize.naturalsize(df.memory_usage(deep=True).sum())


def _print_dtype_changes(df_new, old_dtypes, old_size: Optional[str] = None):
    """
    Prints a DataFrame of dtype changes

    Parameters
    ----------
    df_new: DataFrame with the new dtypes
    old_dtypes: the Series that is contained in the df.dtypes attribute.

    Returns
    -------
    None, It prints a DataFrame where the index is the column names.
    It has two columns: old_dtype and new_dtype

    """
    old_dtypes = old_dtypes.rename("old_dtype")
    new_dtypes = df_new.dtypes.rename("new_dtype")

    df_changes = (
        pd.concat([old_dtypes, new_dtypes], axis=1)
        .loc[lambda df: df.old_dtype.ne(df.new_dtype)]
        .rename_axis("column", axis=0)
        .reset_index()
    )

    n_changes = len(df_changes)

    print(f"{n_changes} of {len(old_dtypes)} dtypes were changed\n\n")

    if n_changes > 0:
        print(df_changes, "\n")
        if old_size:
            print(f"Resized from {old_size} to {get_memory_usage(df_new)}")


@dataclass
class DtypeCasting:
    """
    This class programmatically and dynamically updates the dtypes in a DataFrame

    Parameters
    ----------
    dtypes_to_check: the dtypes that should be checked.
    new_dtype: the intended dtype
    errors: How the errors should be handled
    downcast: The smallest numerical dtype to attempt when downcasting
    coerce_func: One of the Pandas to_{dtype} functions
    verbose: Should the dtype changes be printed?
    """

    dtypes_to_check: Union[List, Tuple, str, object]
    new_dtype: Union[str, object]
    coerce_func: Optional[Callable] = None
    errors: Optional[str] = "ignore"
    downcast: Optional[Literal["integer", "signed", "unsigned", "float"]] = ("integer",)
    verbose: Optional[bool] = True

    def cast(
        self,
        df: pd.DataFrame,
        cols_to_check: Optional[Union[List, Tuple, str, int, float]] = None,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        df: DataFrame
        cols_to_check: a subset of columns to check.

        Returns
        -------
        A DataFrame with new dtypes
        """
        old_dtypes = df.dtypes
        old_size = get_memory_usage(df)

        df_subset = df

        if cols_to_check is not None:
            df_subset = df_subset[cols_to_check]

        df_subset = df_subset.select_dtypes(self.dtypes_to_check)

        if self.coerce_func:
            if self.coerce_func is pd.to_numeric:
                final_func = partial(
                    self.coerce_func, errors=self.errors, downcast=self.downcast
                )
            elif self.coerce_func is pd.to_datetime:
                final_func = partial(self.coerce_func, errors=self.errors)
            else:
                final_func = self.coerce_func

            df_subset = df_subset.apply(final_func).select_dtypes(self.new_dtype)
        else:
            df_subset = df_subset.astype(self.new_dtype, errors=self.errors)

        casted_cols = df_subset.columns.tolist()

        if casted_cols:
            df[casted_cols] = df_subset
            if self.verbose:
                coersion_func = (
                    self.coerce_func.__name__
                    if self.coerce_func
                    else f"astype({self.new_dtype})"
                )
                print(f"Ran {coersion_func}")
                _print_dtype_changes(df, old_dtypes, old_size)

        return df


def cast_to_numeric(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple]] = None,
    errors: Optional[Literal["ignore", "raise", "coerce"]] = "ignore",
    downcast: Optional[Literal["integer", "signed", "unsigned", "float"]] = "integer",
    verbose: Optional[bool] = True,
):
    """
    Cast columns to numeric dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    downcast: The smallest numerical dtype to attempt when downcasting
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with numeric dtypes
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="number",
        coerce_func=pd.to_numeric,
        errors=errors,
        downcast=downcast,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def to_boolean(s: pd.Series) -> pd.Series:
    """
    Coerce a non-boolean Series to a boolean Series if possible

    Parameters
    ----------
    s: Pandas Series

    Returns
    -------
    a coerced Pandas bool Series if possible or the original Series
    """
    if not pd.api.types.is_bool_dtype(s):
        s_values = set(s)
        if len(s_values) == 2:
            if s_values == {True, False}:
                s = s.astype("boolean")
            elif s_values == {"True", "False"}:
                s = s.eq("True")
    return s


def cast_to_boolean(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple]] = None,
    verbose: Optional[bool] = True,
    **kwargs,
):
    """
    Cast columns to numeric dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    verbose: Should the dtype changes be printed?
    kwargs: Just here for compatibility. Not currently in use.

    Returns
    -------
    A DataFrame with boolean dtypes if possible
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="boolean",
        coerce_func=to_boolean,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_category(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple]] = None,
    errors: Optional[Literal["ignore", "raise", "coerce"]] = "ignore",
    verbose: Optional[bool] = True,
):
    """
    Cast columns to categorical dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with categorical dtypes
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="category",
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_datetime(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple]] = None,
    errors: Optional[Literal["ignore", "raise", "coerce"]] = "ignore",
    verbose: Optional[bool] = True,
):
    """
    Cast columns to datetime dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with datetime dtypes
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype=["datetime", "datetimetz"],
        coerce_func=pd.to_datetime,
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_category(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple]] = None,
    errors: Optional[Literal["ignore", "raise", "coerce"]] = "ignore",
    verbose: Optional[bool] = True,
):
    """
    Cast columns to categorical dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with categorical dtypes
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="category",
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def clean_dtypes(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    cols_to_check: Optional[Union[List, Tuple, pd.Series]] = None,
    errors: Optional[Literal["ignore", "raise", "coerce"]] = "ignore",
    downcast: Optional[Literal["integer", "signed", "unsigned", "float"]] = "integer",
    super_verbose: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Casts all the DataFrame dtypes to more optimal dtypes

    - The convert_dtypes method will convert numeric and datetime object columns to
        numeric and datetime dtypes, but not for numbers and datetimes that are being
        stored as strings
    - cast_to_numeric casts all numeric-like non-numeric columns to numeric.
        (The convert_dtypes method doesn't do this.)
    - cast_to_datetime: Because the convert_dtypes method fails to find datetime-like
        strings, cast_to_datetime does this.
    - cast_to_boolean casts bool object columns and "True"/"False" string values
        to the boolean dtype.
    - cast_to_category casts any remaining columns to categories

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    downcast: The smallest numerical dtype to attempt when downcasting
    super_verbose: Should the dtype changes be printed in all 3 cast functions?

    Returns
    -------
    A DataFrame with more appropriate dtypes
    """

    cast_func_args = dict(
        dtypes_to_check=dtypes_to_check,
        errors=errors,
        verbose=super_verbose,
        cols_to_check=cols_to_check,
    )

    old_dtypes = df.dtypes
    old_size = get_memory_usage(df)

    df = (
        # TODO: create convert_dtypes wrapper
        df.convert_dtypes()
        .pipe(cast_to_numeric, **cast_func_args, downcast=downcast)
        .pipe(cast_to_datetime, **cast_func_args)
        .pipe(cast_to_boolean, **cast_func_args)
        .pipe(cast_to_category, **cast_func_args)
    )
    print("Ran clean_dtypes")
    _print_dtype_changes(df, old_dtypes, old_size)

    return df
