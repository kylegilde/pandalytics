import numpy as np
import pandas as pd
import pytest


print(pd.Series(np.random.randint(0, 1_000, 100), dtype=int))


@pytest.fixture(scope="module")
def df_pytest():
    np.random.seed(0)

    df = pd.DataFrame(
        dict(
            cat_col=pd.Series(np.random.choice(list("ABC"), 100), dtype="category"),
            string_col=pd.Series(np.random.choice(list("DEF"), 100), dtype="string"),
            object_col=pd.Series(np.random.choice(list("GHI"), 100), dtype="object"),
            date_col=pd.Series(
                np.random.choice(pd.date_range("2023-06-27", periods=5), 100)
            ),
            int_col=pd.Series(np.random.randint(0, 1_000, 100), dtype=int),
            float_col=pd.Series(np.random.uniform(0, 1, 100), dtype=float),
        )
    )

    df.mask(np.random.random(df.shape) < 0.1, inplace=True)

    return df
