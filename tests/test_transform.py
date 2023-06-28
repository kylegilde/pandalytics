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
            "A",
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
            None,
            "C",
            "C",
            "C",
            "D",
            "D",
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
            "F",
            "E",
            "F",
            np.nan,
            "E",
            "F",
            "E",
            "F",
            "E",
            "F",
            np.nan,
            "E",
            "F",
        ],
        "a": [
            635.7692307692307,
            534.5,
            449.0,
            514.3333333333334,
            533.7142857142857,
            663.0,
            282.0,
            522.2857142857143,
            543.1818181818181,
            800.0,
            561.0,
            657.4285714285714,
            521.6666666666666,
            512.3333333333334,
            290.0,
            273.0,
            316.5,
            np.nan,
            507.5,
        ],
        "b": [
            0.22431702897473926,
            0.5549096570776055,
            0.18115096173690304,
            0.5399917223903892,
            0.36171136200792897,
            0.7567786427368892,
            0.011427458625031028,
            0.6394725163987236,
            0.7705807485027762,
            0.6449622496357191,
            0.3799269559001205,
            0.3685846061296175,
            0.5573687913239169,
            0.6297127437750096,
            0.7270442627113283,
            0.3451864847154928,
            0.14814086094816503,
            0.01323685775889949,
            0.40724117141380733,
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