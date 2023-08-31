from typing import Optional, List
from dataclasses import dataclass

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


@dataclass
class FeatureImportancePlot:
    """

    Parameters
    ----------
    features: an array of feature names
    importance_values: an array of importance values
    max_scale: Should the importance values be scaled by the maximum value
        & mulitplied by 100? The scaled importance values can be talked about relative
        to the most important feature.
    n_decimals_displayed: How many decimals should be displayed?
    str_pad_width: the number of spaces to pad between the rank integer and feature name

    Attributes
    ----------
    df_importance: Contains columns feature, value, ranking, ranked_feature & text
    """

    features: List
    importance_values: List
    max_scale: Optional[bool] = True
    n_decimals_displayed: Optional[int] = 1
    str_pad_width: Optional[int] = 15

    def transform(self):
        self.df_importance: pd.DataFrame = (
            pd.Series(self.importance_values, index=self.features, name="value")
            .rename_axis("feature")
            .reset_index()
            .sort_values(["value", "feature"], ignore_index=True)
            .assign(
                ranking=lambda _df: _df["value"]
                .rank(method="dense", ascending=False)
                .astype(int)
            )
            # add the rank to the feature name
            .assign(
                ranked_feature=lambda _df: _df.ranking.astype("string")
                + ".  "
                + _df.feature.astype("string").str.pad(width=self.str_pad_width)
            )
        )

        if self.max_scale:
            max_value = self.df_importance["value"].max()
            self.df_importance["value"] = (
                self.df_importance["value"].div(max_value).mul(100)
            )

        self.df_importance["text"] = self.df_importance["value"].round(
            self.n_decimals_displayed
        )

        self.n_features = len(self.df_importance)

        return self.df_importance

    def plot(
        self,
        top_n_features: Optional[int] = None,
        height_per_feature: Optional[int] = 25,
        width: Optional[int] = 750,
        yaxes_tickfont_family: Optional[str] = "Courier New, monospace",
        yaxes_tickfont_size: Optional[int] = 15,
        **kwargs,
    ) -> go.Figure:
        """
        Plot the Feature Names & Importances

        Parameters
        ----------
        top_n_features: the number of features to plot, default is 100
        height_per_feature: if height is not specified,
            the plot height is calculated by top_n_features * height_per_feature.
            This allows all the features enough space to be displayed.
        width:  the width of the plot
        yaxes_tickfont_family: the font for the feature names. Default is Courier New.
        yaxes_tickfont_size: the font size for the feature names. Default is 15.

        Returns
        -------
        plot

        """

        if not hasattr(self, "df_importance"):
            self.transform()

        df_plot = self.df_importance.copy()

        # Create a title if one was not provided
        create_title = "title" not in kwargs
        if create_title:
            kwargs["title"] = "All Feature Importances"

        if "height" not in kwargs:
            kwargs["height"] = max(self.n_features * height_per_feature, 400)

        # If you have a lot of features, you may want to show only the top ones.
        if isinstance(top_n_features, int) and self.n_features > top_n_features:
            df_plot = df_plot[lambda df: df.ranking.le(top_n_features)]
            if create_title:
                kwargs[
                    "title"
                ] = f"Top {top_n_features} (of {self.n_features}) Feature Importances"

        # create the plot
        return (
            px.bar(
                df_plot,
                x="value",
                y="ranked_feature",
                text="text",
                width=width,
                **kwargs,
            )
            # center the title and do not show the legend
            .update_layout(title_x=0.5, showlegend=False).update_yaxes(
                tickfont=dict(family=yaxes_tickfont_family, size=yaxes_tickfont_size),
                title="",
            )
        )
