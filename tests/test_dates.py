import pytest
import pandas as pd

from pandalytics.dates import (
    get_holiday_dates,
    create_bday_flag,
    get_datetime_attributes,
    filter_to_business_dates,
    count_fractional_business_days,
)


def test_get_holiday_dates():
    expected_values = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-01 00:00:00"),
            pd.Timestamp("2021-05-31 00:00:00"),
            pd.Timestamp("2021-07-05 00:00:00"),
            pd.Timestamp("2021-09-06 00:00:00"),
            pd.Timestamp("2021-11-25 00:00:00"),
            pd.Timestamp("2021-12-24 00:00:00"),
            pd.Timestamp("2021-12-31 00:00:00"),
            pd.Timestamp("2022-05-30 00:00:00"),
            pd.Timestamp("2022-07-04 00:00:00"),
            pd.Timestamp("2022-09-05 00:00:00"),
            pd.Timestamp("2022-11-24 00:00:00"),
            pd.Timestamp("2022-12-26 00:00:00"),
        ]
    )

    test_values = get_holiday_dates("2021-01-01", "2023-01-01")
    pd.testing.assert_index_equal(test_values, expected_values)


def test_create_bday_flag(df_pytest):
    expected_values = pd.Series(
        [
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
        ]
    )
    test_values = create_bday_flag(df_pytest.date_col_2)
    pd.testing.assert_series_equal(test_values, expected_values, check_names=False)


def test_get_datetime_attributes():
    s_input = (
        pd.date_range("2023-09-15", periods=10)
        .rename("created_at")
        .to_series()
        .reset_index(drop=True)
    )

    df_expected = pd.DataFrame(
        {
            "created_at_hour": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "created_at_day": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            "created_at_day_of_week": [4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            "created_at_week": [37, 37, 37, 38, 38, 38, 38, 38, 38, 38],
            "created_at_month": [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            "created_at_year": [
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
            ],
            "created_at_quarter": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "created_at_day_of_year": [
                258,
                259,
                260,
                261,
                262,
                263,
                264,
                265,
                266,
                267,
            ],
        }
    )
    df_test = get_datetime_attributes(s_input)

    pd.testing.assert_frame_equal(df_test, df_expected, check_dtype=False)


def test_count_fractional_business_days():
    s_expected = pd.Series(
        [
            508.0,
            481.125,
            453.2083333333333,
            424.0,
            397.125,
            368.0,
            341.0,
            309.6666666666667,
            284.0,
            256.0,
        ]
    )
    s_start = (
        pd.date_range("2020-06-27", "2023-06-27", periods=10)
        .to_series()
        .reset_index(drop=True)
    )
    s_end = (
        pd.date_range("2022-06-27", "2024-06-27", periods=10)
        .to_series()
        .reset_index(drop=True)
    )
    s_test = count_fractional_business_days(s_start, s_end)
    pd.testing.assert_series_equal(s_test, s_expected)
