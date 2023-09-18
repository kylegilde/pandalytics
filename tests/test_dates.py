import pytest
import pandas as pd

from pandalytics.dates import (
    get_holiday_dates,
    create_bday_flag,
    get_datetime_attributes,
    filter_to_business_dates,
    count_fractional_business_days,
)

EXPECTED_BUSINESS_DATES = pd.Series(
    {
        1: pd.Timestamp("2020-07-08 01:27:16.363636363"),
        3: pd.Timestamp("2020-07-30 04:21:49.090909091"),
        4: pd.Timestamp("2020-08-10 05:49:05.454545454"),
        5: pd.Timestamp("2020-08-21 07:16:21.818181818"),
        6: pd.Timestamp("2020-09-01 08:43:38.181818182"),
        8: pd.Timestamp("2020-09-23 11:38:10.909090909"),
        10: pd.Timestamp("2020-10-15 14:32:43.636363636"),
        11: pd.Timestamp("2020-10-26 16:00:00"),
        12: pd.Timestamp("2020-11-06 17:27:16.363636364"),
        13: pd.Timestamp("2020-11-17 18:54:32.727272728"),
        15: pd.Timestamp("2020-12-09 21:49:05.454545454"),
        17: pd.Timestamp("2021-01-01 00:43:38.181818182"),
        18: pd.Timestamp("2021-01-12 02:10:54.545454546"),
        20: pd.Timestamp("2021-02-03 05:05:27.272727272"),
        22: pd.Timestamp("2021-02-25 08:00:00"),
        23: pd.Timestamp("2021-03-08 09:27:16.363636364"),
        24: pd.Timestamp("2021-03-19 10:54:32.727272728"),
        25: pd.Timestamp("2021-03-30 12:21:49.090909092"),
        27: pd.Timestamp("2021-04-21 15:16:21.818181816"),
        29: pd.Timestamp("2021-05-13 18:10:54.545454544"),
        30: pd.Timestamp("2021-05-24 19:38:10.909090908"),
        31: pd.Timestamp("2021-06-04 21:05:27.272727272"),
        32: pd.Timestamp("2021-06-15 22:32:43.636363636"),
        34: pd.Timestamp("2021-07-08 01:27:16.363636364"),
        35: pd.Timestamp("2021-07-19 02:54:32.727272728"),
        36: pd.Timestamp("2021-07-30 04:21:49.090909092"),
        37: pd.Timestamp("2021-08-10 05:49:05.454545456"),
        39: pd.Timestamp("2021-09-01 08:43:38.181818184"),
        41: pd.Timestamp("2021-09-23 11:38:10.909090912"),
        42: pd.Timestamp("2021-10-04 13:05:27.272727272"),
        43: pd.Timestamp("2021-10-15 14:32:43.636363632"),
        44: pd.Timestamp("2021-10-26 16:00:00"),
        46: pd.Timestamp("2021-11-17 18:54:32.727272728"),
        48: pd.Timestamp("2021-12-09 21:49:05.454545456"),
        49: pd.Timestamp("2021-12-20 23:16:21.818181816"),
        51: pd.Timestamp("2022-01-12 02:10:54.545454544"),
        53: pd.Timestamp("2022-02-03 05:05:27.272727272"),
        54: pd.Timestamp("2022-02-14 06:32:43.636363632"),
        55: pd.Timestamp("2022-02-25 08:00:00"),
        56: pd.Timestamp("2022-03-08 09:27:16.363636360"),
        58: pd.Timestamp("2022-03-30 12:21:49.090909088"),
        60: pd.Timestamp("2022-04-21 15:16:21.818181816"),
        61: pd.Timestamp("2022-05-02 16:43:38.181818184"),
        62: pd.Timestamp("2022-05-13 18:10:54.545454544"),
        63: pd.Timestamp("2022-05-24 19:38:10.909090912"),
        65: pd.Timestamp("2022-06-15 22:32:43.636363632"),
        66: pd.Timestamp("2022-06-27 00:00:00"),
        67: pd.Timestamp("2022-07-08 01:27:16.363636360"),
        68: pd.Timestamp("2022-07-19 02:54:32.727272728"),
        70: pd.Timestamp("2022-08-10 05:49:05.454545456"),
        72: pd.Timestamp("2022-09-01 08:43:38.181818184"),
        73: pd.Timestamp("2022-09-12 10:10:54.545454544"),
        74: pd.Timestamp("2022-09-23 11:38:10.909090912"),
        75: pd.Timestamp("2022-10-04 13:05:27.272727272"),
        77: pd.Timestamp("2022-10-26 16:00:00"),
        79: pd.Timestamp("2022-11-17 18:54:32.727272720"),
        80: pd.Timestamp("2022-11-28 20:21:49.090909088"),
        81: pd.Timestamp("2022-12-09 21:49:05.454545456"),
        82: pd.Timestamp("2022-12-20 23:16:21.818181824"),
        84: pd.Timestamp("2023-01-12 02:10:54.545454544"),
        85: pd.Timestamp("2023-01-23 03:38:10.909090912"),
        86: pd.Timestamp("2023-02-03 05:05:27.272727264"),
        87: pd.Timestamp("2023-02-14 06:32:43.636363632"),
        89: pd.Timestamp("2023-03-08 09:27:16.363636368"),
        91: pd.Timestamp("2023-03-30 12:21:49.090909088"),
        92: pd.Timestamp("2023-04-10 13:49:05.454545456"),
        93: pd.Timestamp("2023-04-21 15:16:21.818181824"),
        94: pd.Timestamp("2023-05-02 16:43:38.181818176"),
        96: pd.Timestamp("2023-05-24 19:38:10.909090912"),
        98: pd.Timestamp("2023-06-15 22:32:43.636363632"),
        99: pd.Timestamp("2023-06-27 00:00:00"),
    },
    name="date_col_3",
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


def test_filter_to_business_dates_w_series(df_pytest):
    s_test = filter_to_business_dates(df_pytest["date_col_3"])
    pd.testing.assert_series_equal(s_test, EXPECTED_BUSINESS_DATES)


def test_filter_to_business_dates_w_df(df_pytest):
    df_expected = EXPECTED_BUSINESS_DATES.to_frame()
    df_test = filter_to_business_dates(df_pytest[["date_col_3"]], "date_col_3")
    pd.testing.assert_frame_equal(df_test, df_expected)
