from collections.abc import Callable

import pandas as pd


def groupby_apply(
    df: pd.DataFrame,
    groupby_cols: str | int | float | list | pd.Series,
    apply_func: Callable,
    observed: bool | None = True,
    dropna: bool | None = False,
    **kwargs,
) -> pd.DataFrame:
    """

    Apply a Function to Groups of Column Values.
    This is essentially doing many-nested For Loop in Pandas.
    Only use this function if you input and output are DataFrames.
    Your apply_func is likely a custom aggregation that can't be done in .agg()

    Parameters
    ----------
    df: DataFrame
    groupby_cols: a string or array of column names
    apply_func: a function that returns a DataFrame or Series.
        Your apply apply_func should return a Series if your apply_func is returning only 1 row of data.
        Create a dictionary containing the names of the columns you want as its keys and the values.
        Wrap your dictionary in a Series.
        The dict keys will become the Series index, which will become comes in your new DataFrame.
    observed: Should only observed value combinations be used if any groupby columns are categories?
        Changes the Pandas default.
    dropna: Should NaN group keys be removed? Changes the Pandas default.
    kwargs: parameters for apply_func

    Returns
    -------
    DataFrame containing the groupby_cols first, followed by any columns created by apply_func
    """
    # FYI, It's not good to use the as_index=False because weird things happen.
    return (
        df.groupby(groupby_cols, observed=observed, dropna=dropna)
        .apply(apply_func, **kwargs)
        .reset_index(groupby_cols)
    )


pd.DataFrame.groupby_apply = groupby_apply


def cast_dict_to_columns(
    a_dict: dict, key_col: str | None = "key", value_col: str | None = "value"
) -> pd.DataFrame:
    """

    Cast a Dictionary into a 2-Column DataFrame
    The dictionary values should be scalars

    Parameters
    ----------
    a_dict: dictionary containing values that are scalar
    key_col: name of the column that will contain the dictionary's keys
    value_col:  name of the column that will contain the dictionary's values

    Returns
    -------
    a 2-column DataFrame with a key_col and value_col

    """

    return pd.Series(a_dict, name=value_col).rename_axis(key_col).reset_index()


def flatten_column_names(df: pd.DataFrame, sep: str | None = "_"):
    """

    Concatenate MultiIndex Columns using the Separator String.
    This useful after a df.pivot() is performed.
    The smallest level is concatenated first.

    Parameters
    ----------
    df: DataFrame
    sep: a separator string

    Returns
    -------
    a DataFrame with 1 level of concatenated column names

    """

    df.columns = (
        df.columns.to_flat_index().to_series().apply(lambda t: str(t[1]) + sep + str(t[0])).values
    )

    return df


def sort_all_values(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """

    Sorts your Entire DataFrame by Each Column

    Parameters
    ----------
    df: DataFrame
    args: positional arguments passed to sort_values
    kwargs: key-value pairs passed to the sort_values method

    Returns
    -------
    a completely sorted DataFrame

    """

    return df.sort_values(df.columns.tolist(), *args, **kwargs)


def drop_single_value_cols(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    """
    Drop all the columns w/ only a single value

    :param df: DataFrame
    :param dropna: should NAs be dropped before counting?

    :return: DataFrame
    """
    single_value_cols = df.nunique(dropna=dropna).loc[lambda x: x == 1].index.to_series().tolist()

    if single_value_cols:
        print(f'\n\nDropping these single value columns:\n{",".join(single_value_cols)}\n\n')

    return df.drop(columns=single_value_cols)


def format_percentage(s: pd.Series, n_decimals: int | None = 2) -> pd.Series:
    """
    Formats a numeric Series as a percentage string

    Parameters
    ----------
    s: numeric Series
    n_decimals: the number of decimal places to retain in the percentage

    Returns
    -------
    a percentage

    """
    return s.apply(lambda x: f"{x:.{n_decimals}%}").astype("string")


def move_column(df: pd.DataFrame, col_name: str, new_position: int | None = 0) -> pd.DataFrame:
    """
    Move column to a certain position among the other columns
    """

    col_values = df.pop(col_name)
    df.insert(new_position, col_name, col_values)

    return df
    

# def apply_to_columns(df: pd.DataFrame, columns, fn, **kwargs) -> pd.DataFrame:
#     """
#     Apply a Function to a Subset of Columns
#     Useful when using .pipe()
#
#     :param df: DataFrame
#     :param columns: an iterable of column names
#     :param fn: a function to apply
#
#     :return: updated DataFrame
#
#     If Koalas/Pandas API on Spark, set_option("compute.ops_on_diff_frames", True) must be run
#     """
#
#     df[columns] = df[columns].apply(fn, **kwargs)
#
#     return df
