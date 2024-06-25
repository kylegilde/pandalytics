import numpy as np
import pandas as pd
from pandalytics.transform import (
    groupby_apply,
    cast_dict_to_columns,
    flatten_column_names,
    sort_all_values,
    drop_single_value_cols,
    change_display,
    format_percentage,
)


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


def test_cast_dict_to_columns():
    d = dict(a=1, b=2)
    df_expected = pd.DataFrame(dict(key=["a", "b"], value=[1, 2]))
    df_test = cast_dict_to_columns(d)
    pd.testing.assert_frame_equal(df_test, df_expected)


def test_flatten_column_names():
    df_input = pd.DataFrame(np.arange(16).reshape(4, 4), dtype="int32")
    df_input.columns = pd.MultiIndex.from_product([[1, 2], ["A", "B"]])

    df_expected = pd.DataFrame(
        {
            "A_1": [0, 4, 8, 12],
            "B_1": [1, 5, 9, 13],
            "A_2": [2, 6, 10, 14],
            "B_2": [3, 7, 11, 15],
        },
        dtype="int32",
    )
    df_test = flatten_column_names(df_input)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_sort_all_values():
    df_input = pd.DataFrame(dict(a=list("zyx"), b=range(2, -1, -1)))
    df_expected = pd.DataFrame(dict(a=list("xyz"), b=range(3)))
    df_test = sort_all_values(df_input, ignore_index=True)

    pd.testing.assert_frame_equal(df_test, df_expected), "sort_all_values did not work."

    # Revert df_test back to df_input
    df_revert = sort_all_values(df_test, ignore_index=True, ascending=False)
    (
        pd.testing.assert_frame_equal(df_revert, df_input),
        "Descending sort_all_values did not work.",
    )


def test_drop_single_value_cols(df_pytest):
    df_input = df_pytest.copy()
    df_input[["a", "b", "c", "d", "e"]] = (
        pd.NA,
        pd.Timestamp("2000-01-01"),
        1.0,
        "hello",
        "bye",
    )
    df_input["e"] = df_input["e"].astype("category")

    expected_columns = df_pytest.columns
    test_columns = drop_single_value_cols(df_input).columns

    pd.testing.assert_index_equal(test_columns, expected_columns)


def test_change_display():
    change_display()
    assert pd.get_option("display.min_rows") == 25
    assert pd.get_option("display.max_rows") == 50
    assert pd.get_option("display.max_columns") == 100
    assert pd.get_option("max_colwidth") == 400
    assert pd.get_option("display.width") == 1000

    format_string = "{:,.%df}" % 4
    assert pd.get_option("display.float_format")(
        0.0000000000001
    ) == format_string.format(0.0000000000001)


def test_format_percentage():
    s_input = pd.Series(
        [
            0.36111968691526775,
            -0.7542024178311975,
            -1.006893731921199,
            -0.8882585711641657,
            -0.7263882615576498,
            0.9157907845736917,
            0.27678388892377814,
            1.1561438902373338,
            1.3207863193402076,
            1.3717394372051284,
        ]
    )
    s_expected = pd.Series(
        [
            "36.11%",
            "-75.42%",
            "-100.69%",
            "-88.83%",
            "-72.64%",
            "91.58%",
            "27.68%",
            "115.61%",
            "132.08%",
            "137.17%",
        ],
        dtype="string",
    )
    s_test = format_percentage(s_input)
    pd.testing.assert_series_equal(s_test, s_expected)
