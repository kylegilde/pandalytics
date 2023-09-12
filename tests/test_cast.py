import pandas as pd

from pandalytics.cast import (
    cast_to_datetime,
    cast_to_category,
    cast_to_numeric,
    to_boolean,
    cast_to_boolean,
    cast_dtypes,
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


def test_to_boolean_from_string(df_pytest):
    s_input = df_pytest.bool_col.astype("string")
    s_expected = df_pytest.bool_col
    s_test = to_boolean(s_input)

    pd.testing.assert_series_equal(s_test, s_expected)


def test_to_boolean_from_object(df_pytest):
    s_input = df_pytest.bool_col.astype("object")
    s_expected = df_pytest.bool_col
    s_test = to_boolean(s_input)

    pd.testing.assert_series_equal(s_test, s_expected)


def test_to_boolean_from_object_na(df_pytest):
    # If the object Series contains an NA,
    # then the same object Series should be returned
    s_input = df_pytest.bool_col.copy().astype("object")
    s_input[0] = pd.NA
    s_expected = s_input
    s_test = to_boolean(s_input)

    pd.testing.assert_series_equal(s_test, s_expected)


def test_cast_to_boolean_from_string(df_pytest):
    bool_cols = df_pytest.select_dtypes(["boolean"]).columns
    df_expected = df_pytest

    df_test = df_pytest.copy()
    df_test[bool_cols] = df_test[bool_cols].astype("string")
    df_test = cast_to_boolean(df_pytest)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_to_boolean_from_object(df_pytest):
    bool_cols = df_pytest.select_dtypes(["boolean"]).columns
    df_expected = df_pytest

    df_test = df_pytest.copy()
    df_test[bool_cols] = df_test[bool_cols].astype("object")
    df_test = cast_to_boolean(df_pytest)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_cast_dtypes_from_string(df_pytest):
    df_expected = df_pytest
    expected_dtypes = pd.Series(
        {
            "cat_col": "category",
            "string_col": "category",
            "object_col": "category",
            "date_col": "datetime64[ns]",
            "binary_col": "UInt8",
            "int_col": "UInt16",
            "int_col_2": "Int16",
            "int_col_3": "Int32",
            "float_col": "Float32",
            "float_col_2": "Float32",
            "float_col_3": "Float32",
            "bool_col": "boolean",
            "normal_1": "Float32",
            "normal_2": "Float32",
            "date_col_2": "datetime64[ns, UTC]",
            "date_col_3": "datetime64[ns]",
        },
        dtype="string",
    )
    test_dtypes = df_expected.astype("string").pipe(cast_dtypes).dtypes.astype("string")
    # test_dtypes.to_dict()

    pd.testing.assert_series_equal(expected_dtypes, test_dtypes)


def test_cast_dtypes_from_object(df_pytest):
    df_expected = df_pytest
    expected_dtypes = pd.Series(
        {
            "cat_col": "category",
            "string_col": "category",
            "object_col": "category",
            "date_col": "datetime64[ns]",
            "binary_col": "UInt8",
            "int_col": "UInt16",
            "int_col_2": "Int16",
            "int_col_3": "Int32",
            "float_col": "Float32",
            "float_col_2": "Float32",
            "float_col_3": "Float32",
            "bool_col": "boolean",
            "normal_1": "Float32",
            "normal_2": "Float32",
            "date_col_2": "datetime64[ns, UTC]",
            "date_col_3": "datetime64[ns]",
        },
        dtype="string",
    )
    test_dtypes = df_expected.astype("object").pipe(cast_dtypes).dtypes.astype("string")
    # test_dtypes.to_dict()

    pd.testing.assert_series_equal(expected_dtypes, test_dtypes)
