from typing import Optional, List, Tuple, Union, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DtypeCasting:
    """
    -> pd.DataFrame:
        Parameters
        ----------
        df: DataFrame
        dtypes_to_check: a string or array of strings
        new_dtype: a new dtype
        cols_to_check: Limit the casting to a specific subset of columns. An array of strings
        verbose: Should the printing happen?

        Returns
        -------
        A DataFrame with new column dtypes

    """

    dtypes_to_check: Union[List, Tuple, str, object]
    new_dtype: Union[str, object]
    verbose: Optional[bool] = True
    coerce_func: Optional[Callable] = None
    errors: Optional[str] = "ignore"

    def cast(
        self,
        df: pd.DataFrame,
        cols_to_check: Optional[Union[List, Tuple, str, int, float]] = None,
    ) -> pd.DataFrame:
        df_subset = df

        if cols_to_check is not None:
            df_subset = df_subset[cols_to_check]

        df_subset = df_subset.select_dtypes(self.dtypes_to_check)

        if self.coerce_func:
            df_subset = df_subset.apply(
                self.coerce_func, errors=self.errors
            ).select_dtypes(self.new_dtype)
        else:
            df_subset = df_subset.astype(self.new_dtype, errors=self.errors)

        casted_cols = df_subset.columns.tolist()

        if casted_cols:
            if self.verbose:
                dtypes_str = (
                    " & ".join(self.dtypes_to_check)
                    if isinstance(self.dtypes_to_check, (list, tuple))
                    else self.dtypes_to_check
                )

                cols_str = "\n".join(casted_cols)

                print(
                    f"Casting {len(casted_cols)} {dtypes_str} columns to "
                    f'{self.new_dtype}:\n\n{cols_str}'
                )

            df[casted_cols] = df_subset

        return df


def cast_to_numeric(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    verbose: Optional[bool] = True,
    errors: Optional[str] = "ignore",
    cols_to_check: Optional[Union[List, Tuple]] = None,
    **kwargs,
):
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="number",
        coerce_func=pd.to_numeric,
        errors=errors,
        verbose=verbose,
        **kwargs,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_category(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    verbose: Optional[bool] = True,
    errors: Optional[str] = "ignore",
    cols_to_check: Optional[Union[List, Tuple, pd.Series]] = None,
    **kwargs,
):
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="category",
        errors=errors,
        verbose=verbose,
        **kwargs,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_datetime(
    df: pd.DataFrame,
    dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    verbose: Optional[bool] = True,
    errors: Optional[str] = "ignore",
    cols_to_check: Optional[Union[List, Tuple, pd.Series]] = None,
    **kwargs,
):
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="number",
        coerce_func=pd.to_datetime,
        errors=errors,
        verbose=verbose,
        **kwargs,
    ).cast(df, cols_to_check=cols_to_check)


def clean_dtypes(
    df: pd.DataFrame,
    cols_to_check: Optional[Union[List, Tuple, pd.Series]] = None,
    numeric_dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    datetime_dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    category_dtypes_to_check: Optional[Union[List, Tuple, str, object]] = (
        "object",
        "string",
    ),
    verbose: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Cleans up all suboptimal Pandas dtypes by running 4 methods/functions

    - Casts all numeric-like non-numeric columns to numeric. convert_dtypes doesn't do this
    - convert_dtypes method
        - casts all integer-like float columns to Int64 and string-like object columns to string.
        - if possible, it casts object columns to strings. It won't do this if there are Python objects in the column.
    - cast_to_datetime: Because the convert_dtypes method fails to find datetime-like string & object columns, & cast them to datetime, cast_to_datetime does this.
    - cast_to_string casts any remaining object columns (which probably contain arrays or non-scalar values like lists or dicts) to strings

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    numeric_dtypes_to_check: the dtypes that should be checked.
    datetime_dtypes_to_check: the dtypes that should be checked.
    category_dtypes_to_check: the dtypes that should be checked.
    verbose: Should the printing happen?

    Returns
    -------
    A DataFrame with more appropriate dtypes
    """

    return (
        df.pipe(
            cast_to_numeric,
            dtypes_to_check=numeric_dtypes_to_check,
            verbose=verbose,
            cols_to_check=cols_to_check,
        )
        .convert_dtypes()
        .pipe(
            cast_to_datetime,
            dtypes_to_check=datetime_dtypes_to_check,
            verbose=verbose,
            cols_to_check=cols_to_check,
        )
        .pipe(
            cast_to_category,
            dtypes_to_check=category_dtypes_to_check,
            verbose=verbose,
            cols_to_check=cols_to_check,
        )
    )


# def get_memory_usage(df):
#     # ! pip install humanize
#     # import humanize
#     # from numerize import numerize
#     return humanize.naturalsize(df.memory_usage(deep=True).sum())