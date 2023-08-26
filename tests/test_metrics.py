import pytest
import numpy as np
import pandas as pd

from pandalytics.metrics import (
    get_regression_metrics,
    create_regression_metrics,
    use_metric_abbreviations,
)

df_expected = pd.DataFrame(
    {
        "metric": [
            "Volume",
            "Mean Prediction ",
            "Median Prediction ",
            "Mean Actual ",
            "Median Actual ",
            "Mean Error (ME)",
            "Absolute Mean Error (AME)",
            "Mean Percentage Error (MPE)",
            "Median Error",
            "Mean Absolute Error (MAE)",
            "R-Squared",
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Percentage Error (MAPE) %",
            "Smoothed Mean Absolute Percentage Error (sMAPE) %",
        ],
        "value": [
            100.0,
            1008.6489868164062,
            995.4647216796875,
            1005.0119018554688,
            1011.7374267578125,
            3.637106418609619,
            3.637106418609619,
            0.038056813180446625,
            -4.4300537109375,
            194.61508178710938,
            -0.6964595993912561,
            244.42440795898438,
            0.2064204216003418,
            0.19815784692764282,
        ],
    }
)

df_expected_2 = df_expected.assign(
    metric=[
        "Volume",
        "Mean Prediction ",
        "Median Prediction ",
        "Mean Actual ",
        "Median Actual ",
        "ME",
        "AME",
        "MPE",
        "Median Error",
        "MAE",
        "R-Squared",
        "RMSE",
        "MAPE",
        "sMAPE",
    ]
)


def test_get_regression_metrics(df_pytest):
    df_test = get_regression_metrics(df_pytest.normal_1, df_pytest.normal_2)

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_create_regression_metrics(df_pytest):
    df_test = create_regression_metrics(
        df_pytest, y_true_col="normal_1", y_pred_col="normal_2"
    )

    pd.testing.assert_frame_equal(df_test, df_expected)


def test_use_metric_abbreviations(df_pytest):
    df_test = create_regression_metrics(
        df_pytest, y_true_col="normal_1", y_pred_col="normal_2", use_abbreviations=True
    )

    pd.testing.assert_frame_equal(df_test, df_expected_2)
