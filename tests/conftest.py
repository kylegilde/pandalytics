import numpy as np
import pandas as pd
import pytest

N_ROWS = 100
SEED = 0
NORMAL_LOC = 1_000
NORMAL_SCALE = 200


@pytest.fixture(scope="module")
def df_pytest():
    np.random.seed(SEED)
    df_pytest = pd.DataFrame(
        dict(
            cat_col=pd.Series(np.random.choice(list("AB"), N_ROWS), dtype="category"),
            string_col=pd.Series(np.random.choice(list("CD"), N_ROWS), dtype="string"),
            object_col=pd.Series(np.random.choice(list("EF"), N_ROWS), dtype="object"),
            date_col=pd.Series(
                np.random.choice(pd.date_range("2023-06-27", periods=5), N_ROWS)
            ),
            binary_col=pd.Series(np.random.choice([1, 0], N_ROWS), dtype="Int8"),
            int_col=pd.Series(np.random.randint(0, 1_000, N_ROWS), dtype="Int16"),
            int_col_2=pd.Series(
                np.random.randint(-1_000, 1_000, N_ROWS), dtype="Int16"
            ),
            int_col_3=pd.Series(
                np.random.randint(-1_000_000, 1_000_000, N_ROWS), dtype="Int32"
            ),
            float_col=pd.Series(np.random.uniform(0, 1, N_ROWS), dtype="Float64"),
            float_col_2=pd.Series(np.random.uniform(-1, 1, N_ROWS), dtype="Float64"),
            float_col_3=pd.Series(
                np.random.uniform(-1_000, 1_000, N_ROWS), dtype="Float64"
            ),
        )
    )

    # Throw in some NaNs
    mask = np.random.random(df_pytest.shape) < 0.1
    df_pytest.mask(mask, inplace=True)

    # Create some columns without NaNs
    df_pytest["bool_col"] = pd.Series(
        np.random.choice([True, False], N_ROWS), dtype="boolean"
    )
    df_pytest["normal_1"] = pd.Series(
        np.random.normal(loc=NORMAL_LOC, scale=NORMAL_SCALE, size=N_ROWS),
        dtype="Float64",
    )
    df_pytest["normal_2"] = pd.Series(
        np.random.normal(loc=NORMAL_LOC, scale=NORMAL_SCALE, size=N_ROWS),
        dtype="Float64",
    )

    df_pytest["date_col_2"] = pd.date_range("2023-06-27", periods=N_ROWS, tz="UTC")
    df_pytest["date_col_3"] = pd.date_range("2020-06-27", "2023-06-27", periods=N_ROWS)

    return df_pytest
