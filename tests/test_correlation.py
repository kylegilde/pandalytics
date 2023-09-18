import pytest
import plotly.graph_objects as go
from pandalytics.correlation import PairwiseCorrelations


def test_transform(df_pytest):
    pass


@pytest.mark.parametrize(
    "y_col,method,dtypes,sep",
    [
        (None, "pearson", ["datetime"], " and "),
        ("bool_col", "kendall", ["number", "bool", "datetime", "datetimetz"], None),
        ("date_col", "spearman", ["bool"], None),
        ("float_col", "kendall", ["number", "bool", "datetime", "datetimetz"], " - "),
        (None, "kendall", ["number", "bool", "datetime", "datetimetz"], " - "),
    ],
)
def test_plot(df_pytest, y_col, method, dtypes, sep):
    pc = PairwiseCorrelations(y_col, method, dtypes, sep)
    fig = pc.plot(df_pytest)
    assert isinstance(fig, go.Figure), "The plot method did NOT return a go.Figure."
