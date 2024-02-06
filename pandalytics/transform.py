from typing import Union, Optional, List, Dict, Callable
import pandas as pd


def groupby_apply(
    df: pd.DataFrame,
    groupby_cols: Union[str, int, float, List, pd.Series],
    apply_func: Callable,
    observed: Optional[bool] = True,
    dropna: Optional[bool] = False,
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
    a_dict: Dict, key_col: Optional[str] = "key", value_col: Optional[str] = "value"
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


def flatten_column_names(df: pd.DataFrame, sep: Optional[str] = "_"):
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
    single_value_cols = (
        df.nunique(dropna=dropna).loc[lambda x: x == 1].index.to_series().tolist()
    )

    if single_value_cols:
        print(
            f'\n\nDropping these single value columns:\n{",".join(single_value_cols)}\n\n'
        )

    return df.drop(columns=single_value_cols)


def format_percentage(s: pd.Series, n_decimals: Optional[int] = 2) -> pd.Series:
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


def change_display(
    min_rows: Optional[int] = 25,
    max_rows: Optional[int] = 50,
    max_columns: Optional[int] = 100,
    max_colwidth: Optional[int] = 400,
    width: Optional[int] = 1000,
    n_decimals: Optional[int] = 4,
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
