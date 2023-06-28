import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def df_pytest():
    np.random.seed(0)
    df_pytest = pd.DataFrame(
        dict(
            cat_col=pd.Series(np.random.choice(list("AB"), 100), dtype="category"),
            string_col=pd.Series(np.random.choice(list("CD"), 100), dtype="string"),
            object_col=pd.Series(np.random.choice(list("EF"), 100), dtype="object"),
            date_col=pd.Series(
                np.random.choice(pd.date_range("2023-06-27", periods=5), 100)
            ),
            int_col=pd.Series(np.random.randint(0, 1_000, 100), dtype=int),
            float_col=pd.Series(np.random.uniform(0, 1, 100), dtype=float),
            float_col_2=pd.Series(np.random.uniform(-1, 1, 100), dtype=float),
            float_col_3=pd.Series(np.random.uniform(-1_000, 1_000, 100), dtype=float),
        )
    )

    df_pytest.mask(np.random.random(df_pytest.shape) < 0.1, inplace=True)

    return df_pytest
