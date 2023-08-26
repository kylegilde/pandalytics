import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandalytics.correlation import PairwiseCorrelations


def test_transform(df_pytest):
    pass


def test_plot(df_pytest):
    pc = PairwiseCorrelations()

    fig = pc.plot(df_pytest)

    assert isinstance(fig, go.Figure), "The plot method did NOT return a go.Figure."
