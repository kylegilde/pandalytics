from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import humanize
import numpy as np
import pandas as pd
from pandalytics.general_utils import safe_partial


@dataclass
class DtypeCasting:
    """
    This class programmatically and dynamically updates the dtypes in a DataFrame

    Parameters
    ----------
    dtypes_to_check: the dtypes that should be checked.
    new_dtype: the intended dtype
    errors: How the errors should be handled
    coerce_func: One of the Pandas to_{dtype} functions or a UDF
    coerce_func_kws: a dict of key-value pairs for coerce_func
    verbose: Should the dtype changes be printed?
    """

    dtypes_to_check: list | tuple | str | object
    new_dtype: str | object | None = None
    coerce_func: Callable | None = None
    coerce_func_kws: dict = field(default_factory=dict)
    errors: str | None = "ignore"
    verbose: bool | None = True

    def cast(
        self,
        df: pd.DataFrame,
        cols_to_check: list | tuple | str | int | float | None = None,
    ) -> pd.DataFrame:
        """
        Applies a casting / coersion to an entire DataFrame by column

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

        if self.verbose:
            casting_function = (
                self.coerce_func.__name__ if self.coerce_func else f"astype({self.new_dtype})"
            )
            print(f"Running {casting_function}")

        if self.coerce_func:
            final_func = safe_partial(
                self.coerce_func,
                errors=self.errors,
                **self.coerce_func_kws,
            )
            df_subset = df_subset.apply(final_func)
        else:
            df_subset = df_subset.astype(self.new_dtype, errors=self.errors)

        dtype_changes = get_dtype_changes(
            df_subset, old_dtypes, old_size=old_size, verbose=self.verbose
        )

        if dtype_changes:
            df[dtype_changes] = df_subset[dtype_changes]

        return df


@dataclass
class IntCasting:
    """
    Downcast Floats and Integers to the smallest int dtype possible based upon the
    min and max values

    Attributes
    ----------
    df_int_metadata: contains the dtype names and min/max values
    """

    df_int_metadata = pd.DataFrame(
        {
            "pandas_dtype": [
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
            ],
            "min_value": [0, 0, 0, 0, -128, -32768, -2147483648, -9223372036854775808],
            "max_value": [
                255,
                65535,
                4294967295,
                18446744073709551615,
                127,
                32767,
                2147483647,
                9223372036854775807,
            ],
        }
    )

    df_int_metadata["pyarrow_dtypes"] = df_int_metadata.pandas_dtype.str.lower() + "[pyarrow]"

    # df_int_metadata.melt(id_vars=["min_value", "max_value"], value_name="dtype_string",
    #                      var_name="backend").set_index("backend")

    def could_be_int(self, s: pd.Series) -> bool:
        """
        Is the number an int or float and has no decimal values?

        Parameters
        ----------
        s: Series

        Returns
        -------
        bool
        """
        self.int_eligible = pd.api.types.is_integer_dtype(s) or (
            pd.api.types.is_float_dtype(s) and s.dropna().mod(1).eq(0).all()
        )

        return self.int_eligible

    def downcast_integer(self, s: pd.Series) -> pd.Series:
        """
        Downcast a numeric Series to the smallest signed or unsigned integer dtype
        based upon the Series min and max.

        Parameters
        ----------
        s: Series

        Returns
        -------
        Series
        """
        if not hasattr(self, "int_eligible"):
            self.could_be_int(s)

        if not self.int_eligible:
            return s

        # It cannot get smaller than UInt8
        if (current_dtype := str(s.dtype)) == "UInt8":
            return s

        mx, mn = s.max(), s.min()
        smallest_dtype = self.df_int_metadata.loc[
            lambda df: df.min_value.le(mn) & df.max_value.ge(mx), "pandas_dtype"
        ].iloc[0]

        return s.astype(smallest_dtype) if current_dtype != smallest_dtype else s


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


def get_dtype_changes(
    df_new, old_dtypes, old_size: str | None = None, verbose: bool | None = True
):
    """
    Prints a DataFrame of dtype changes

    Parameters
    ----------
    df_new: DataFrame with the new dtypes
    old_dtypes: the Series that is contained in the df.dtypes attribute.
    old_size: The string output from get_memory_usage
    verbose: Should the dtype changes be printed?


    Returns
    -------
    None, It prints a DataFrame where the index is the column names.
    It has two columns: old_dtype and new_dtype

    """
    old_dtypes = old_dtypes.rename("old_dtype")
    new_dtypes = df_new.dtypes.rename("new_dtype")

    df_changes = (
        pd.concat([old_dtypes, new_dtypes], join="inner", axis=1)
        .loc[lambda df: df.old_dtype.ne(df.new_dtype)]
        .rename_axis("column", axis=0)
        .reset_index()
    )
    if verbose:
        n_changes = len(df_changes)

        print(f"{n_changes} of {len(old_dtypes)} dtypes were changed\n\n")

        if n_changes > 0:
            print(df_changes, "\n")
            if old_size:
                print(f"Resized from {old_size} to {get_memory_usage(df_new)}")

    return df_changes.column.to_list()


def cast_to_numeric(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = (
        "object",
        "string",
    ),
    cols_to_check: list | tuple | None = None,
    errors: Literal["ignore", "raise", "coerce"] | None = "ignore",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    verbose: bool | None = True,
):
    """
    Cast / Coerce columns to numeric dtypes

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
        coerce_func=pd.to_numeric,
        coerce_func_kws=dict(downcast=downcast),
        errors=errors,
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
                s = s.eq("True").astype("boolean")
    return s


def cast_to_boolean(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = (
        "object",
        "string",
    ),
    cols_to_check: list | tuple | None = None,
    verbose: bool | None = True,
):
    """
    Cast columns to numeric dtypes

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with boolean dtypes if possible
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        coerce_func=to_boolean,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_string(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = ("object",),
    cols_to_check: list | tuple | None = None,
    errors: Literal["ignore", "raise", "coerce"] | None = "ignore",
    verbose: bool | None = True,
):
    """
    Cast columns to the string dtype
    Use this first before trying to use cast_to_category on actually Python object columns

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    errors: How the errors should be handled
    verbose: Should the dtype changes be printed?

    Returns
    -------
    A DataFrame with string dtypes

    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="string",
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_category(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = (
        "object",
        "string",
    ),
    coerce_objects: bool | None = True,
    cols_to_check: list | tuple | None = None,
    errors: Literal["ignore", "raise", "coerce"] | None = "ignore",
    verbose: bool | None = True,
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

    # if (
    #     coerce_objects
    #     and "object" in dtypes_to_check
    #     and "object" in df.columns.dtypes.astype("string")
    # ):
    #     if verbose:
    #         print("Coercing objects")

    #     df = cast_to_string(df, cols_to_check=cols_to_check, verbose=verbose)

    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        new_dtype="category",
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def cast_to_datetime(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = (
        "object",
        "string",
    ),
    cols_to_check: list | tuple | None = None,
    errors: Literal["ignore", "raise", "coerce"] | None = "ignore",
    verbose: bool | None = True,
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
        coerce_func=pd.to_datetime,
        errors=errors,
        verbose=verbose,
    ).cast(df, cols_to_check=cols_to_check)


def downcast_to_float32_if_unique(s: pd.Series) -> pd.Series:
    """
    Downcast a Float64 Series to Float32 if the number of unique values can be
    preserved.

    Parameters
    ----------
    s: Series

    Returns
    -------
    a Float32 Series if possible
    """
    if (s_new := s.astype("Float32")).nunique() == s.nunique():
        return s_new
    return s


def is_object_or_string(s: pd.Series) -> bool:
    """
    Convenient wrapper function for testing if a Series is an object or string dtype

    Parameters
    ----------
    s: Series

    Returns
    -------
    bool

    """
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def cast_dtype(
    s: pd.Series, downcast: bool | None = True, use_categories: bool | None = True
) -> pd.Series:
    """
    Dynamically coerces a Series to the correct dtype & by default, downcasts numeric dtypes
    to the smallest dtype possible without reducing the number of unique values.

    Parameters
    ----------
    s: Series
    downcast: Should a numeric dtype be downcast to the smallest dtype possible without
        reducing the number of unique values
    use_categories: Should remaining object and string Series be cast to as
        memory-efficient categories?

    Returns
    -------
    a correct and more memory-efficient dtype if possible
    """
    # First, coerce to numeric if possible
    if is_object_or_string(s):
        s = s.pipe(pd.to_numeric, errors="ignore")

    # Then, convert_dtypes will converts datetime or boolean objects to datetime or boolean.
    # For consistency, it also converts any numeric numpy dtypes to pandas dtypes.
    s = s.convert_dtypes()

    if downcast:
        # If the Series could be an integer, downcast it to the smallest integer
        # dtype based upon the Series min and max.
        ic = IntCasting()
        if ic.could_be_int(s):
            return ic.downcast_integer(s)

        # Downcast a Float64 Series to Float32 if the number of unique values can be
        # preserved.
        if s.dtype in (pd.Float64Dtype(), np.float64):
            return downcast_to_float32_if_unique(s)

    # If it is still a string or object, then try 3 other dtypes. Otherwise, cast it to
    # a category
    if is_object_or_string(s):
        # If the Series cannot be coerced to a number, then try a datetime.
        if pd.api.types.is_datetime64_any_dtype(s := pd.to_datetime(s, errors="ignore")):
            return s
        # Then try boolean
        if pd.api.types.is_bool_dtype(s := to_boolean(s)):
            return s
        # Then try the timedelta dtype, which does not like pd.NA.
        if pd.api.types.is_timedelta64_dtype(
            s := pd.to_timedelta(s.fillna(np.nan), errors="ignore")
        ):
            return s
        # Cast any remaining strings or objects to a memory-efficient category.
        if use_categories:
            return s.astype("category")

    return s


def cast_dtypes(
    df: pd.DataFrame,
    dtypes_to_check: list | tuple | str | object | None = (
        "object",
        "string",
        "number",
    ),
    cols_to_check: list | tuple | pd.Series | None = None,
    downcast: bool | None = True,
    use_categories: bool | None = True,
) -> pd.DataFrame:
    """
    Dynamically coerces the columns in a DataFrame to the correct dtype.

    By default, it also downcasts numeric dtypes to the smallest dtype possible without
    reducing the number of unique values.

    Parameters
    ----------
    df: DataFrame
    cols_to_check: a subset of columns to check.
    dtypes_to_check: the dtypes that should be checked.
    downcast: Should a numeric dtype be downcast to the smallest dtype possible without
        reducing the number of unique values
    use_categories: Should remaining object and string Series be cast to as
        memory-efficient categories?

    Returns
    -------
    A DataFrame with more correct and more memory-efficient dtypes if possible
    """
    return DtypeCasting(
        dtypes_to_check=dtypes_to_check,
        coerce_func=cast_dtype,
        coerce_func_kws=dict(use_categories=use_categories, downcast=downcast),
        verbose=True,
    ).cast(df, cols_to_check=cols_to_check)
