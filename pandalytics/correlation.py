from typing import Optional, Literal, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


@dataclass
class PairwiseCorrelations:
    """

    Parameters
    ----------
    method:
    sep:
    dtypes:

    Returns
    -------

    """

    y_col: Optional[Union[int, float, str]] = None
    method: Optional[Literal["pearson", "kendall", "spearman"]] = "spearman"
    dtypes: Optional[ArrayLike] = ("number", "bool", "datetime", "datetimetz")
    sep: Optional[str] = " & "

    def _coersive_corr(self, s1, s2):
        return (
            pd.to_numeric(s1, errors="ignore")
            .astype(float)
            .corr(pd.to_numeric(s2, errors="ignore").astype(float), method=self.method)
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        df

        Returns
        -------

        """
        if self.y_col:
            # Calculate the correlation between 1 column and the rest
            self.df_corr = (
                df.drop(columns=self.y_col)
                .select_dtypes(self.dtypes)
                .apply(self._coersive_corr, s2=df[self.y_col])
                .rename("value")
                .rename_axis("y_label")
                .reset_index()
            )
        else:
            # Create the correlation matrix using the specified dtypes
            df_corr_matrix = df.select_dtypes(self.dtypes).corr(self.method)
            # Create a boolean mask for the upper triangle, including the diagonal
            nan_mask = np.triu(np.ones(df_corr_matrix.shape, dtype=bool))

            self.df_corr = (
                df_corr_matrix.mask(nan_mask)  # sets the upper triangle to nans
                .rename_axis("variable_1")
                .reset_index()
                .melt("variable_1", var_name="variable_2")
                .dropna()  # drops the upper triangle of nans
                .assign(
                    y_label=lambda _df: _df.variable_1.astype(str).str.cat(
                        _df.variable_2.astype(str), sep=self.sep
                    )
                )
            )

        # Create some metadata for the plot
        self.df_corr = self.df_corr.assign(
            abs_value=lambda _df: _df.value.abs(),
            is_positive=lambda _df: _df.value.ge(0),
            text=lambda _df: _df.value.round(2),
            method=self.method,
        ).sort_values("abs_value", ignore_index=True)

        return self.df_corr

    def plot(
            self,
            df: pd.DataFrame,
            min_abs_correlation: Optional[float] = None,
            **kwargs,
    ) -> go.Figure:
        """

        Parameters
        ----------
        df
        min_abs_correlation
        kwargs

        Returns
        -------

        """
        if not hasattr(self, "df_corr"):
            self.transform(df)

        corr_type = (
            f"{self.y_col} Correlations" if self.y_col else "Pairwise Correlations"
        )

        if min_abs_correlation:
            max_abs_value: float = self.df_corr.abs_value.max()
            if min_abs_correlation > max_abs_value:
                raise ValueError(
                    f"{min_abs_correlation = } is greater than "
                    f"maximum absolute value of {max_abs_value}."
                )

            df_plot = self.df_corr.loc[
                lambda _df: _df.abs_value.ge(min_abs_correlation)
            ]

            title = (
                f"Showing {df_plot.shape[0]} of {self.df_corr.shape[0]:,}"
                f" {corr_type}"
            )
        else:
            df_plot = self.df_corr
            title = f"{df_plot.shape[0]} {corr_type}"

        return (
            px.bar(
                df_plot,
                x="abs_value",
                y="y_label",
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
