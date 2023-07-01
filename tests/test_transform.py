import numpy as np
import pandas as pd
from pandalytics.transform import groupby_apply

df_grouped_metrics_expected = pd.DataFrame(
    {
        "cat_col": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "string_col": [
            "C",
            "C",
            "C",
            "D",
            "D",
            None,
            "C",
            "C",
            "C",
            "D",
            "D",
            "D",
            None,
            None,
            None,
            "C",
            "C",
            "C",
            "D",
            "D",
        ],
        "object_col": [
            "E",
            "F",
            np.nan,
            "E",
            "F",
            "E",
            "E",
            "F",
            np.nan,
            "E",
            "F",
            np.nan,
            "E",
            "F",
            np.nan,
            "E",
            "F",
            np.nan,
            "E",
            "F",
        ],
        "a": [
            631.9,
            628.8333333333334,
            533.5,
            586.8333333333334,
            564.6363636363636,
            503.75,
            545.2222222222222,
            258.875,
            226.0,
            645.2727272727273,
            483.14285714285717,
            148.0,
            697.0,
            632.3333333333334,
            259.0,
            343.0,
            383.25,
            284.0,
            765.5,
            369.0,
        ],
        "b": [
            0.4038279286739412,
            0.32945009773618733,
            0.6995750602247514,
            0.28705151991962974,
            0.6223844285660048,
            0.41354673981340584,
            0.5436780538280952,
            0.6131744314303385,
            0.7400975256176825,
            0.41210986031588004,
            0.2562826979416177,
            0.3742961833209161,
            0.9249669119531921,
            0.46874793390963654,
            0.23780724253903884,
            0.4240889884358493,
            0.4375342636551285,
            0.3416981148647321,
            0.4141742723825833,
            0.5249704422542643,
        ],
    },
).astype(
    {
        "cat_col": pd.CategoricalDtype(categories=["A", "B"], ordered=False),
        "string_col": "string[python]",
        "object_col": np.dtype("O"),
        "a": np.dtype("float64"),
        "b": np.dtype("float64"),
    }
)


def test_groupby_apply_function(df_pytest):
    df_grouped_metrics = groupby_apply(
        df_pytest,
        ["cat_col", "string_col", "object_col"],
        lambda df: pd.Series(dict(a=df.int_col.mean(), b=df.float_col.median())),
    )

    pd.testing.assert_frame_equal(df_grouped_metrics_expected, df_grouped_metrics)


def test_groupby_apply_method(df_pytest):
    df_grouped_metrics = df_pytest.groupby_apply(
        ["cat_col", "string_col", "object_col"],
        lambda df: pd.Series(dict(a=df.int_col.mean(), b=df.float_col.median())),
    )

    pd.testing.assert_frame_equal(df_grouped_metrics_expected, df_grouped_metrics)
