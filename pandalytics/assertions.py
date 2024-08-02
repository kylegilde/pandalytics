import pandas as pd
import logging

def assert_no_nas(df: pd.DataFrame) -> None:
    """Makes sure there are no NaNs in DataFrame"""
    assert (n_nas := df.isna().sum().sort_values(ascending=False)).eq(0).all(), f"{n_nas = }"
    logging.info("No NAs!")


def assert_no_duplicates(x: pd.DataFrame | pd.Series | pd.Index) -> None:
    """
    Makes sure there are no duplicated rows in DataFrame
    """
    duplicates = x.duplicated(keep=False)

    if (n_duplicated := duplicates.sum()) > 0:
        # subset to dupe data
        x_dupes = x[duplicates] if isinstance(x, pd.Index) else x.loc[duplicates]

        if isinstance(x_dupes, pd.DataFrame):
            x_dupes = x_dupes.sort_all_values()
        else:
            x_dupes = x_dupes.sort_values()

        msg = f"{n_duplicated = :,} out of {len(x):,}\n{x_dupes}"

        raise AssertionError(msg)

    logging.info("No dupes!")
