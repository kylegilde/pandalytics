import pandas as pd
import plotly.graph_objects as go
from pandalytics.plot import plot_line_plot, plot_bar_plot


def test_add_new_keys():
    pass


def test_infer_yaxes_tickformat():
    pass


def test_plot_line_plot(df_pytest):
    fig = plot_line_plot(data_frame=df_pytest, x="date_col", y="normal_1")

    assert isinstance(fig, go.Figure), "plot_line_plot did NOT return a go.Figure."


def test_plot_bar_plot(df_pytest):
    df = pd.DataFrame(dict(cat=list("abc"), value=range(3)))
    fig = plot_bar_plot(data_frame=df, x="cat", y="value")

    assert isinstance(fig, go.Figure), "plot_bar_plot did NOT return a go.Figure."
