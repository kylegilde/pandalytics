import pandas as pd

from pandalytics.cast import (
    cast_to_datetime,
    cast_to_category,
    cast_to_numeric,
    clean_dtypes,
)


def test_cast_to_category(df_pytest):
    df_expected = df_pytest.copy()
    cat_cols = ["cat_col", "object_col", "string_col"]
    df_expected[cat_cols] = df_expected[cat_cols].astype("category")
    df_test = cast_to_category(df_pytest)
    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_datetime_from_object(df_pytest):
    date_cols = df_pytest.select_dtypes(["datetime", "datetimetz"]).columns
    df_expected = df_pytest.copy()

    df_test = df_pytest.copy()
    df_test[date_cols] = df_test[date_cols].astype(object)
    df_test = cast_to_datetime(df_pytest)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_datetime_from_string(df_pytest):
    date_cols = df_pytest.select_dtypes(["datetime", "datetimetz"]).columns
    df_expected = df_pytest

    df_test = df_pytest.copy()
    df_test[date_cols] = df_test[date_cols].astype("string")
    df_test = cast_to_datetime(df_pytest)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_datetime_from_category(df_pytest):
    date_cols = df_pytest.select_dtypes(["datetime", "datetimetz"]).columns
    df_expected = df_pytest

    df_test = df_pytest.copy()
    df_test[date_cols] = df_test[date_cols].astype("category")
    df_test = cast_to_datetime(df_pytest)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_numeric_from_string(df_pytest):
    df_expected = df_pytest.select_dtypes("number")
    df_test = df_expected.astype("string").pipe(cast_to_numeric)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_numeric_from_object(df_pytest):
    df_expected = df_pytest.select_dtypes("number")
    df_test = df_expected.astype("object").pipe(cast_to_numeric)
    # The numeric dtypes are not the same in df_test, but they are numeric.
    pd.testing.assert_frame_equal(df_test, df_expected, check_dtype=False)


# def test_clean_dtypes_from_string(df_pytest):
#     df_expected = df_pytest.select_dtypes(
#         ["number", "object", "string", "datetime", "datetimetz"]
#     )
#     # df_test.dtypes.astype("string").to_dict()
#     expected_dtypes = pd.Series(
#         {
#             "string_col": "category",
#             "object_col": "category",
#             "date_col": "datetime64[ns]",
#             "binary_col": "Int8",
#             "int_col": "Int16",
#             "int_col_2": "Int16",
#             "int_col_3": "Int32",
#             "float_col": "Float64",
#             "float_col_2": "Float64",
#             "float_col_3": "Float64",
#             "normal_1": "Float64",
#             "normal_2": "Float64",
#             "date_col_2": "datetime64[ns, UTC]",
#             "date_col_3": "datetime64[ns]",
#         },
#         dtype="string",
#     )
#     df_test = df_expected.astype("string").pipe(clean_dtypes)
#     pd.testing.assert_series_equal(expected_dtypes, df_test.dtypes.astype("string"))
