
import pandas as pd
from pandalytics.transform import format_percentage


def value_counts_pct(
    s: pd.Series, n_decimals: int | None = 2, dropna: bool | None = False
) -> pd.Series:
    """
    Count Values by their Percentage

    Parameters
    ----------
    s: a Series
    n_decimals: the number of decimals to show in the percentages
    dropna: should NAs be dropped?

    Returns
    -------
    a Series
    """
    return (
        s.value_counts(dropna=dropna)
        .to_frame("n")
        .assign(pct=lambda df: format_percentage(df.n.div(df.n.sum()), n_decimals=n_decimals))
    )


def count_nas(
    df: pd.DataFrame,
    n_decimals: int | None = 2,
    use_inf_as_na: bool | None = True,
) -> pd.DataFrame:
    """
    Count the NAs in your DF including the Indices
    Koalas/Pandas API on Spark Compatible

    Parameters
    ----------
    df: DataFrame
    n_decimals: the number of decimals to show in the percentages
    use_inf_as_na: Should any np.inf's be counted as NAs?

    Returns
    -------
    a DataFrame containing n_NAs & pct_NAs for each column

    """

    with pd.option_context("mode.use_inf_as_na", use_inf_as_na):
        return (
            df.isna()
            .sum()
            .sort_values(ascending=False)
            .to_frame("n_NAs")
            .assign(
                pct_NAs=lambda summary_df: summary_df.n_NAs.div(df.shape[0]).pipe(
                    format_percentage, n_decimals=n_decimals
                )
            )
        )


def count_unique(
    df: pd.DataFrame, n_decimals: int | None = 2, dropna: bool | None = False
) -> pd.DataFrame:
    """
    Count the Unique Values in your DataFrame

    Parameters
    ----------
    df: DataFrame
    n_decimals: the number of decimals to show in the percentages
    dropna: should the NAs be dropped?

    Returns
    -------
    a DataFrame containing n_unique & pct_unique for each column
    """
    return (
        df.nunique(dropna=dropna)
        .sort_values(ascending=False)
        .to_frame(name="n_unique")
        .assign(
            pct_unique=lambda _df: _df.n_unique.div(df.shape[0]).pipe(
                format_percentage, n_decimals=n_decimals
            )
        )
    )


def count_duplicates(df_or_s: pd.DataFrame | pd.Series) -> int:
    """
    Count the instances of duplicated values

    Parameters
    ----------
    df_or_s: DataFrame or Series

    Returns
    -------
    an integer
    """
    return df_or_s.duplicated().sum()
