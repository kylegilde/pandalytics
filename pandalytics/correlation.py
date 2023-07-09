from typing import Optional, Literal
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import plotly.express as px


@dataclass
class PairwiseCorrelations:
    """ """

    method: Optional[Literal["pearson", "kendall", "spearman"]] = "spearman"
    sep: Optional[str] = " & "
    dtypes: Optional[ArrayLike] = ("number", bool, "datetime64")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # df_corr_matrix = df.corr(self.method, numeric_only=True)
        df_corr_matrix = df.select_dtypes(self.dtypes).corr(self.method)
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
                method=self.method,
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

        if min_abs_correlation:
            max_abs_value: float = self.df_pairwise_corr.abs_value.max()
            if min_abs_correlation > max_abs_value:
                raise ValueError(
                    f"{min_abs_correlation = } is greater than "
                    f"maximum absolute value of {max_abs_value}."
                )

            df_plot = self.df_pairwise_corr.loc[
                lambda df: df.abs_value.ge(min_abs_correlation)
            ]

            title = f"Showing {df_plot.shape[0]} of {self.df_pairwise_corr.shape[0]:,} Pairwise Correlations"
        else:
            df_plot = self.df_pairwise_corr
            title = f"{df_plot.shape[0]} Pairwise Correlations"

        return (
            px.bar(
                df_plot,
                x="abs_value",
                y="variable_pair",
                color="value",
                # pattern_shape="method",
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
