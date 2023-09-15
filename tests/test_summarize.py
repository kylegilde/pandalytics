import pandas as pd

from pandalytics.summarize import (
    value_counts_pct,
    count_nas,
    count_unique,
    count_duplicates,
)

N_TEST_ROWS = 93


def test_value_counts_pct(df_pytest):
    df_expected = pd.DataFrame(
        {
            "n": {
                ("D", "E", "B"): 11,
                ("C", "F", "B"): 10,
                ("D", "F", "A"): 10,
                ("C", "E", "A"): 9,
                ("D", "F", "B"): 9,
                ("C", "E", "B"): 8,
                ("D", "E", "A"): 6,
                ("C", "F", "A"): 5,
            },
            "pct": {
                ("D", "E", "B"): "16.18%",
                ("C", "F", "B"): "14.71%",
                ("D", "F", "A"): "14.71%",
                ("C", "E", "A"): "13.24%",
                ("D", "F", "B"): "13.24%",
                ("C", "E", "B"): "11.76%",
                ("D", "E", "A"): "8.82%",
                ("C", "F", "A"): "7.35%",
            },
        }
    )
    test_cols = ["string_col", "object_col", "cat_col"]
    df_expected.index.names = test_cols
    df_test = value_counts_pct(
        df_pytest[:N_TEST_ROWS][test_cols].dropna().astype("object")
    )
    pd.testing.assert_frame_equal(
        df_test, df_expected, check_index_type=False, check_dtype=False
    )


def test_count_nas(df_pytest):
    df_expected = pd.DataFrame(
        {
            "n_NAs": {
                "float_col_3": 13,
                "float_col_2": 13,
                "string_col": 10,
                "cat_col": 9,
                "int_col_2": 8,
                "time_delta_col": 8,
                "binary_col": 8,
                "date_col": 8,
                "object_col": 8,
                "int_col": 7,
                "float_col": 5,
                "int_col_3": 3,
                "bool_col": 0,
                "normal_1": 0,
                "normal_2": 0,
                "date_col_2": 0,
                "date_col_3": 0,
            },
            "pct_NAs": {
                "float_col_3": "13.98%",
                "float_col_2": "13.98%",
                "string_col": "10.75%",
                "cat_col": "9.68%",
                "int_col_2": "8.60%",
                "time_delta_col": "8.60%",
                "binary_col": "8.60%",
                "date_col": "8.60%",
                "object_col": "8.60%",
                "int_col": "7.53%",
                "float_col": "5.38%",
                "int_col_3": "3.23%",
                "bool_col": "0.00%",
                "normal_1": "0.00%",
                "normal_2": "0.00%",
                "date_col_2": "0.00%",
                "date_col_3": "0.00%",
            },
        }
    )
    df_test = count_nas(df_pytest[:N_TEST_ROWS])
    pd.testing.assert_frame_equal(df_test, df_expected, check_dtype=False)


def test_count_unique(df_pytest):
    df_expected = pd.DataFrame(
        {
            "n_unique": {
                "date_col_3": 93,
                "date_col_2": 93,
                "normal_2": 93,
                "normal_1": 93,
                "int_col_3": 91,
                "float_col": 89,
                "time_delta_col": 86,
                "int_col_2": 85,
                "int_col": 83,
                "float_col_2": 81,
                "float_col_3": 81,
                "date_col": 6,
                "string_col": 3,
                "binary_col": 3,
                "object_col": 3,
                "cat_col": 3,
                "bool_col": 2,
            },
            "pct_unique": {
                "date_col_3": "100.0000%",
                "date_col_2": "100.0000%",
                "normal_2": "100.0000%",
                "normal_1": "100.0000%",
                "int_col_3": "97.8495%",
                "float_col": "95.6989%",
                "time_delta_col": "92.4731%",
                "int_col_2": "91.3978%",
                "int_col": "89.2473%",
                "float_col_2": "87.0968%",
                "float_col_3": "87.0968%",
                "date_col": "6.4516%",
                "string_col": "3.2258%",
                "binary_col": "3.2258%",
                "object_col": "3.2258%",
                "cat_col": "3.2258%",
                "bool_col": "2.1505%",
            },
        }
    )
    df_test = count_unique(df_pytest[:N_TEST_ROWS], n_decimals=4)
    pd.testing.assert_frame_equal(df_test, df_expected, check_dtype=False)


def test_count_duplicates(df_pytest):
    assert count_duplicates(df_pytest[["string_col", "object_col", "cat_col"]]) == 80
