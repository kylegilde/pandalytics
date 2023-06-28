from typing import Union, Optional, List, Dict, Callable
import pandas as pd


def groupby_apply(
    df: pd.DataFrame,
    groupby_cols: Union[str, int, float, List, pd.Series],
    apply_func: Callable,
    observed: Optional[bool] = True,
    dropna: Optional[bool] = False,
    **kwargs
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
