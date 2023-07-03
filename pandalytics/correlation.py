from typing import Union, Optional, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.express as px


@dataclass
class PairwiseCorrelations:
    """ """

    method: Literal["pearson", "kendall", "spearman"] = "spearman"
    sep: str = " & "

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_corr_matrix = df.corr(self.method, numeric_only=True)
        nan_mask = np.triu(np.ones(df_corr_matrix.shape)).astype(bool)

        self.df_pairwise_corr = (
            df_corr_matrix.mask(nan_mask)  # sets the upper triangle to nans
            .rename_axis("variable_1")
            .reset_index()
            .melt("variable_1", var_name="variable_2")
            .dropna()
            .assign(
                abs_value=lambda df: df.value.abs(),
                variable_pair=lambda df: df.variable_1.astype(str).str.cat(
                    df.variable_2.astype(str), sep=self.sep
                ),
                is_positive=lambda df: df.value.ge(0),
                text=lambda df: df.value.round(2),
            )
            .sort_values("abs_value", ignore_index=True)
        )

        return self.df_pairwise_corr

    def plot(
        self,
        df: pd.DataFrame,
        min_abs_correlation: Optional[float] = None,
        **kwargs,
    ):
        if not hasattr(self, "df_pairwise_corr"):
            self.transform(df)

        n_pairs = len(self.df_pairwise_corr)
        title = f"{n_pairs:,} Pairwise Correlations"

        if min_abs_correlation:
            if min_abs_correlation > self.df_pairwise_corr.abs_value.min():
                raise ValueError(
                    f"{min_abs_correlation = } is greater than "
                    "all of the absolute correlation values."
                )

            df_plot = self.df_pairwise_corr.loc[
                lambda df: df.abs_value.qe(min_abs_correlation)
            ]
        else:
            df_plot = self.df_pairwise_corr

        return (
            px.bar(
                df_plot,
                x="abs_value",
                y="variable_pair",
                color="value",
                text="text",
                title=title,
                # Create a standardize, static color scale,
                # where a correlation of 1 is solid green,
                # a correlation of -1 is solid red and a correlation of 0 is white.
                range_x=[0, 1],
                color_continuous_scale=["red", "white", "green"],
                color_continuous_midpoint=0,
                range_color=[-1, 1],
                **kwargs,
            )
            .update_layout(
                title_x=0.5,
                font=dict(color="black"),
                plot_bgcolor="lightgray",
                hovermode=False,  # turn off the hover tooltip data
            )
            .update_yaxes(
                title="",
            )
            .update_xaxes(
                title="",
            )
            .update_traces(
                textfont=dict(color="black"),
            )
        )
