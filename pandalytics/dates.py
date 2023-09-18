"""
Contains date-related functions & classes
"""
from typing import Union, Optional, List, Tuple
from functools import partial
import datetime as dt

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


class USMajorHolidayCalendar(USFederalHolidayCalendar):
    """
     This class modifies the holidays used in the parent class.

    Usage:

         holiday_dates = USMajorHolidayCalendar().holidays(
             start=pd.Timestamp("2020-01-01"),
             end=pd.Timestamp('today')
             )



     Parent:
         USFederalHolidayCalendar: _description_
    """

    rules = [
        holiday
        for holiday in USFederalHolidayCalendar.rules
        if holiday.name
        in [
            "New Year's Day",
            "Memorial Day",
            "Independence Day",
            "Labor Day",
            "Thanksgiving Day",
            "Christmas Day",
        ]
    ]


def get_holiday_dates(
    start_date: Union[dt.date, dt.datetime, pd.Timestamp, str],
    end_date: Union[dt.date, dt.datetime, pd.Timestamp, str],
    only_major_holidays: Optional[bool] = True,
) -> pd.DatetimeIndex:
    """
    A function wrapper for the holiday calendar class
    that is more intuitive & convenient to use than the class.

    Notes: Sometimes these dates will vary when they fall on a weekend!

    You can pass the output to the `holidays` parameter in `pd.bdate_range` to remove them

    Parameters
    ----------

    start_date: datetime or date object or a string that pd.Timestamp can coerce. "today" also works
    end_date: datetime or date object or a string that pd.Timestamp can coerce. "today" also works
    only_major_holidays: do you want to use only the 6 Major holidays or all federal holidays?

    Returns
    -------
    DatetimeIndex array

    """

    holiday_class = (
        USMajorHolidayCalendar() if only_major_holidays else USFederalHolidayCalendar()
    )

    return holiday_class.holidays(pd.Timestamp(start_date), pd.Timestamp(end_date))


def create_bday_flag(
    dt_series: pd.Series,
    drop_holidays: Optional[bool] = True,
    only_major_holidays: Optional[bool] = True,
) -> pd.Series:
    """

    Parameters
    ----------
    dt_series: DateTime Series
    drop_holidays: Should holidays be removed?
    only_major_holidays: do you want to use only the 6 Major holidays or all federal
        holidays?

    Returns
    -------
    boolean Series

    """
    boolean_mask = dt_series.dt.day_of_week.lt(5)

    if drop_holidays:
        holidays = get_holiday_dates(
            dt_series.min(), dt_series.max(), only_major_holidays=only_major_holidays
        )
        boolean_mask &= ~dt_series.isin(holidays)

    return boolean_mask


def filter_to_business_dates(
    df_or_s: Union[pd.DataFrame, pd.Series],
    date_col: Optional[Union[str, int]] = None,
    drop_holidays: Optional[bool] = True,
    only_major_holidays: Optional[bool] = True,
) -> Union[pd.DataFrame, pd.Series]:
    """

    Parameters
    ----------
    df_or_s: a DataFrame or datetime Series
    date_col: The name of the datetime column if using a DataFrame
    drop_holidays: Should holidays be removed?
    only_major_holidays: do you want to use only the 6 Major holidays or all federal
        holidays?

    Returns
    -------
    a DataFrame or datetime Series subset to business days
    """
    dt_series = df_or_s[date_col] if isinstance(df_or_s, pd.DataFrame) else df_or_s

    boolean_mask = create_bday_flag(
        dt_series,
        drop_holidays=drop_holidays,
        only_major_holidays=only_major_holidays,
    )

    return df_or_s.loc[boolean_mask]


def get_datetime_attribute(s: pd.Series, attribute: str) -> pd.Series:
    """
    Get DateTime attribute from a DateTime Series.


    Parameters
    ----------
    s: Series
    attribute: name of the attribute

    Returns
    -------
    the attribute Series
    """
    if isinstance(s, pd.DatetimeIndex):
        raise ValueError(
            "get_datetime_attribute won't work with a DatetimeIndex. "
            "Use .to_series() to convert it to a Series"
        )

    return s.dt.isocalendar().week if attribute == "week" else getattr(s.dt, attribute)


def get_datetime_attributes(
    s: pd.Series,
    attributes_to_include: Optional[Union[List, pd.Series, Tuple]] = None,
    prefix_separator: Optional[str] = "_",
) -> pd.DataFrame:
    """
    Returns the Numeric and Boolean DateTime Attribute Values as a DataFrame
    
    Parameters
    ----------
    s: a datetime Series
    attributes_to_include: a list or set of attributes to include.
        If None, use all except the specified exclusions
    prefix_separator: the string to use as a separator between the Series name and attribute name

    Returns
    -------
    a DataFrame

    FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
    """
    if attributes_to_include is None:
        attributes_to_include = (
            "hour",
            "day",
            "day_of_week",
            "week",
            "month",
            "year",
            "quarter",
            "day_of_year",
        )

    prefix = s.name + prefix_separator if s.name else ""

    return pd.concat(
        [
            get_datetime_attribute(s, a).rename(prefix + a)
            for a in attributes_to_include
        ],
        axis=1,
    )


def count_fractional_business_days(
        start_dt_series: pd.Series,
        end_dt_series: pd.Series,
        drop_holidays: Optional[bool] = True,
        only_major_holidays: Optional[bool] = True,
        no_negative_values: Optional[bool] = True,
        business_hour_start: Optional[Union[int, float]] = 9,
        business_hour_end: Optional[Union[int, float]] = 17,
) -> pd.Series:
    """
    Calculate the amount of fractional business days between to datetimes

    Parameters
    ----------
    start_dt_series: datetime Series
    end_dt_series: datetime Series
    drop_holidays: Should holidays be removed?
    only_major_holidays: do you want to use only the 6 Major holidays or all federal
        holidays?
    no_negative_values: If an end date is comes before a start date, should this be set
        to zero?
    business_hour_start: The hour when the business day starts
    business_hour_end: The hour when the business day ends

    Returns
    -------
    Series
    start_dt_series, end_dt_series = df_pytest.date_col_3, df_pytest.date_col_2
    """

    # start the intermediate calculations
    business_hours_per_day = business_hour_end - business_hour_start

    all_dates = pd.concat([start_dt_series, end_dt_series])
    min_date, max_date = all_dates.min(), all_dates.max()

    holidays = get_holiday_dates(
        min_date, max_date, only_major_holidays=only_major_holidays
    )

    # get whole business days. The end date is excluded.
    whole_bdays = np.busday_count(
        start_dt_series.values.astype("datetime64[D]"),
        end_dt_series.values.astype("datetime64[D]"),
        holidays=holidays.values.astype("datetime64[D]"),
    )

    # Find out if the start and end dates are business days
    bday_fn = partial(
        create_bday_flag,
        drop_holidays=drop_holidays,
        only_major_holidays=only_major_holidays,
    )
    start_dt_loss = bday_fn(start_dt_series).astype("float")
    end_dt_gain = bday_fn(end_dt_series).astype("float")

    # If the start date is a business day, calculate partially lost business days
    # that preceded the start time. Otherwise, it will be zero.
    start_dt_loss.mask(
        lambda s: s == 1,
        (
                business_hour_end
                - start_dt_series.dt.hour
                - start_dt_series.dt.minute.div(60)
        )
        .div(business_hours_per_day)
        .clip(0, 1)  # Handles where the current hour < business start hour
        .sub(1),
        inplace=True,
    )

    # this adds the partial bday that occurs on the end date
    end_dt_gain.mask(
        lambda s: s == 1,
        (
            end_dt_series.dt.hour.add(end_dt_series.dt.minute.div(60))
            .sub(business_hour_start)
            .div(business_hours_per_day)
            .clip(0, 1)  # Handles where the current hour < business start hour
        ),
        inplace=True,
    )

    # add up the intermediate calculations
    partial_bdays = whole_bdays + start_dt_loss + end_dt_gain

    if no_negative_values:
        partial_bdays.clip(lower=0, inplace=True)

    return partial_bdays
