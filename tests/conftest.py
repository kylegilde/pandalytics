import numpy as np
import pandas as pd
import pytest

N_ROWS = 100


@pytest.fixture(scope="module")
def df_pytest():
    np.random.seed(0)
    df_pytest = pd.DataFrame(
        dict(
            cat_col=pd.Series(np.random.choice(list("AB"), N_ROWS), dtype="category"),
            string_col=pd.Series(np.random.choice(list("CD"), N_ROWS), dtype="string"),
            object_col=pd.Series(np.random.choice(list("EF"), N_ROWS), dtype="object"),
            date_col=pd.Series(
                np.random.choice(pd.date_range("2023-06-27", periods=5), N_ROWS)
            ),
            bool_col=pd.Series(np.random.choice([True, False], N_ROWS), dtype="object"),
            int_col=pd.Series(np.random.randint(0, 1_000, N_ROWS), dtype=int),
            int_col_2=pd.Series(np.random.randint(-1_000, 1_000, N_ROWS), dtype=int),
            int_col_3=pd.Series(
                np.random.randint(-1_000_000, 1_000_000, N_ROWS), dtype=int
            ),
            float_col=pd.Series(np.random.uniform(0, 1, N_ROWS), dtype=float),
            float_col_2=pd.Series(np.random.uniform(-1, 1, N_ROWS), dtype=float),
            float_col_3=pd.Series(
                np.random.uniform(-1_000, 1_000, N_ROWS), dtype=float
            ),
        )
    )

    df_pytest.mask(np.random.random(df_pytest.shape) < 0.1, inplace=True)

    return df_pytest
