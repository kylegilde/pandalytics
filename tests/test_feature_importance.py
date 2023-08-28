import plotly.graph_objects as go
from pandalytics.feature_importance import FeatureImportancePlot


def test_featureimportanceplot():
    fip = FeatureImportancePlot(list("abcd"), range(4))
    fig = fip.plot()

    assert isinstance(
        fig, go.Figure
    ), "The FeatureImportancePlot did NOT return a go.Figure."
