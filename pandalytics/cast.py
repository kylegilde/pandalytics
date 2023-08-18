from typing import Optional, List, Tuple, Union, Callable, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _print_dtype_changes(df_new, old_dtypes):
    old_dtypes = old_dtypes.rename("old_dtype")
    new_dtypes = df_new.dtypes.rename("new_dtype")

    df_changes = (
        pd.concat([old_dtypes, new_dtypes], axis=1)
        .loc[lambda df: df.old_dtype.ne(df.new_dtype)]
        .rename_axis("column", axis=0)
        .reset_index()
    )

    n_changes = len(df_changes)

    print(
        f"\n{n_changes} of {len(old_dtypes)} dtypes were changed\n\n", df_changes, "\n"
    )


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

        df_subset = df

        if cols_to_check is not None:
            df_subset = df_subset[cols_to_check]

        df_subset = df_subset.select_dtypes(self.dtypes_to_check)

        if self.coerce_func:
            coerce_func_kwargs = dict(errors=self.errors)
            if self.coerce_func is pd.to_numeric:
                coerce_func_kwargs["downcast"] = self.downcast

            df_subset = df_subset.apply(
                self.coerce_func, **coerce_func_kwargs
            ).select_dtypes(self.new_dtype)
        else:
            df_subset = df_subset.astype(self.new_dtype, errors=self.errors)

        casted_cols = df_subset.columns.tolist()

        if casted_cols:
            df[casted_cols] = df_subset
            if self.verbose:
                _print_dtype_changes(df, old_dtypes)

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

    - Casts all numeric-like non-numeric columns to numeric. convert_dtypes doesn't do this
    - convert_dtypes method
        - casts all integer-like float columns to Int64 and string-like object columns to string.
        - if possible, it casts object columns to strings. It won't do this if there are Python objects in the column.
    - cast_to_datetime: Because the convert_dtypes method fails to find datetime-like string & object columns, & cast them to datetime, cast_to_datetime does this.
    - cast_to_category casts any remaining columns (which probably contain arrays or non-scalar values like lists or dicts) to strings

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

    df = (
        df.pipe(cast_to_numeric, **cast_func_args, downcast=downcast)
        .pipe(cast_to_datetime, **cast_func_args)
        .pipe(cast_to_category, **cast_func_args)
    )

    _print_dtype_changes(df, old_dtypes)

    return df


# def get_memory_usage(df):
#     # ! pip install humanize
#     # import humanize
#     # from numerize import numerize
#     return humanize.naturalsize(df.memory_usage(deep=True).sum())
