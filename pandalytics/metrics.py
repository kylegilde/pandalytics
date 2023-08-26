"""
Contains all the non-reliability, non-quantile metric functions
These functions are generally creating metric aggregations to be used in plotly express functions.
"""
from typing import Union, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def use_metric_abbreviations(df, metric_col: Optional[str] = "metric") -> None:
    """

    Parameters
    ----------
    df
    metric_col

    Returns
    -------

    """
    df[metric_col] = (
        df[metric_col]
        .str.extract("((?<=\().+(?=\)))", expand=False)
        .fillna(df[metric_col])
    )

    return None


def get_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    metric_col: Optional[str] = "metric",
    value_col: Optional[str] = "value",
    use_abbreviations: bool = False,
    use_dollar_sign: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Returns a 2-column DataFrame of standard regression metrics

    Parameters
    ----------
    y_true: array of actual values
    y_pred: array of predictions
    metric_col: name of the metric colum
    value_col: name of the value columns
    use_dollar_sign: Should the dollar sign be included in the metric names?
    use_abbreviations: Should only the abbreviations be used for the metric names?

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error
    MAPE & sMAPE output is non-negative floating point.
    The best value is 0.0. But note the fact that bad predictions can lead to arbitrarily large MAPE values,
    especially if some y_true values are very close to zero.
    Note that we return a large value instead of inf when y_true is zero.

    Returns
    -------
    A 2-column DataFrame containing the metric_col and value_col

    """

    dollar_sign = " $" if use_dollar_sign else ""

    errors = y_pred - y_true

    mean_error = np.mean(errors)

    metrics_dict = {
        "Volume": len(y_true),
        f"Mean Prediction {dollar_sign}": y_pred.mean(),
        f"Median Prediction {dollar_sign}": np.median(y_pred),
        f"Mean Actual {dollar_sign}": y_true.mean(),
        f"Median Actual {dollar_sign}": np.median(y_true),
        f"Mean Error (ME){dollar_sign}": mean_error,
        f"Absolute Mean Error (AME){dollar_sign}": abs(mean_error),
        f"Mean Percentage Error (MPE){dollar_sign}": np.mean(errors / y_true),
        f"Median Error{dollar_sign}": np.median(errors),
        f"Mean Absolute Error (MAE){dollar_sign}": mean_absolute_error(y_true, y_pred),
        "R-Squared": r2_score(y_true, y_pred),
        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Mean Absolute Percentage Error (MAPE) %": mean_absolute_percentage_error(
            y_true, y_pred
        ),
        "Smoothed Mean Absolute Percentage Error (sMAPE) %": (
            np.abs(errors) * 2 / (y_pred + y_true)
        ).mean(),
    }

    df_metrics = (
        pd.Series(metrics_dict, name=value_col).rename_axis(metric_col).reset_index()
    )

    if use_abbreviations:
        use_metric_abbreviations(df_metrics, metric_col=metric_col)

    return df_metrics


def create_regression_metrics(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, **kwargs
):
    """
    Convenient wrapper for get_regression_metrics that uses a DataFrame input

    Parameters
    ----------
    df: a DataFrame containing the y_true_col & y_pred_col
    y_true_col: name of the column containing  the dependent variable
    y_pred_col: name of the column containing predictions
    kwargs: key-value pairs passed to get_regression_metrics

    Returns
    -------
    the same output as get_regression_metrics
    A 2-column DataFrame containing the metric_col and value_col
    """

    return get_regression_metrics(df[y_true_col], df[y_pred_col], **kwargs)
