"""
Contains date-related functions & classes
"""
from typing import Union, Optional, List, Tuple
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


def get_business_dates(
        start_date: Union[dt.date, dt.datetime, pd.Timestamp, str],
        end_date: Union[dt.date, dt.datetime, pd.Timestamp, str],
        drop_holidays: Optional[bool] = True,
        only_major_holidays: Optional[bool] = True,
):
    """

    Parameters
    ----------
    start_date
    end_date
    drop_holidays
    only_major_holidays

    Returns
    -------

    """

    if drop_holidays:
        holidays = get_holiday_dates(
            start_date, end_date, only_major_holidays=only_major_holidays
        ).to_list()
    else:
        holidays = None

    return pd.bdate_range(start_date, end_date, holidays=holidays)


def filter_to_business_dates(
        df_or_s: Union[pd.DataFrame, pd.Series],
        date_col: Optional[Union[str, int]] = None,
        drop_holidays: Optional[bool] = True,
        only_major_holidays: Optional[bool] = True,
) -> Union[pd.DataFrame, pd.Series]:
    """

    Parameters
    ----------
    df_or_s
    date_col
    drop_holidays
    only_major_holidays

    Returns
    -------

    """

    if isinstance(df_or_s, pd.DataFrame):
        dt_series = df_or_s[date_col]
    else:
        dt_series = pd.Series(df_or_s)

    boolean_mask = dt_series.dt.day_of_week.lt(5)

    if drop_holidays:
        holidays = get_holiday_dates(
            dt_series.min(), dt_series.max(), only_major_holidays=only_major_holidays
        )
        boolean_mask &= ~dt_series.isin(holidays)

    return df_or_s.loc[boolean_mask]


def get_datetime_attribute(s, attribute):
    """

    Parameters
    ----------
    s
    attribute

    Returns
    -------

    """

    if isinstance(s, pd.DatetimeIndex):
        s, index = s.to_series(), s
    else:
        index = s.index

    return (
        s.dt.isocalendar().week if attribute == "week" else getattr(s.dt, attribute)
    ).reindex(index)


def get_datetime_attributes(
        s: pd.Series,
        attributes_to_include: Optional[Union[List, pd.Series, Tuple]] = None,
        prefix_separator="_",
) -> pd.DataFrame:
    """
    Returns the Numeric and Boolean DateTime Attribute Values as a DataFrame

    :param s: a datetime Series
    :param attributes_to_include: a list or set of attributes to include.
        If None, use all except the specified exclusions
    :param prefix_separator: the string to use as a separator between the Series name and attribute name
    :return: a DataFrame

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
        start_dt_series,
        end_dt_series,
        only_major_holidays: Optional[bool] = True,
        no_negative_values=True,
        business_hour_start=9,
        business_hour_end=17,
):
    """

    Parameters
    ----------
    start_dt_series
    end_dt_series
    no_negative_values
    business_hour_start
    business_hour_end

    Returns
    -------

    """
    start_dt_series, end_dt_series = pd.to_datetime(start_dt_series), pd.to_datetime(
        end_dt_series
    )

    # start the intermediate calculations
    business_hours_per_day = business_hour_end - business_hour_start

    min_date, max_date = start_dt_series.min(), end_dt_series.max()

    holidays = get_holiday_dates(
        min_date, max_date, only_major_holidays=only_major_holidays
    )

    # get whole business days. The end date is excluded.
    whole_bdays = np.busday_count(
        start_dt_series.values.astype("datetime64[D]"),
        end_dt_series.values.astype("datetime64[D]"),
        holidays=holidays.values,
    )

    start_dt_is_business_day = start_dt_series.dt.weekday.lt(5) & ~start_dt_series.isin(
        holidays
    )
    end_dt_is_business_day = end_dt_series.dt.weekday.lt(5) & ~end_dt_series.isin(
        holidays
    )

    # If the start date is a business day, calculate partially lost business days
    # that preceded the start time. Otherwise it will be zero.
    start_dt_loss = np.where(
        start_dt_is_business_day,
        (
                business_hour_end
                - start_dt_series.dt.hour
                - start_dt_series.dt.minute.div(60)
        )
        .div(business_hours_per_day)
        .clip(0, 1)
        .sub(1),
        0,  # zero if not a business day
    )

    # this adds the partial bday that occurs on the end date
    end_dt_gain = np.where(
        end_dt_is_business_day,
        end_dt_series.dt.hour.add(end_dt_series.dt.minute.div(60))
        .sub(business_hour_start)
        .div(business_hours_per_day)
        .clip(0, 1),
        0,
    )

    # add up the intermediate calculations
    partial_bdays = whole_bdays + start_dt_loss + end_dt_gain

    if no_negative_values:
        partial_bdays.clip(lower=0, inplace=True)

    return partial_bdays
