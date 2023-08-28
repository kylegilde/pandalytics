"""
Contains custom plotting functions
"""
from functools import partial
from typing import Optional, Union, Dict, Callable, List
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


LINE_PLOT_AWESOME_DEFAULTS = dict(
    update_layout_dict=dict(
        hoverlabel=dict(font_size=20),
        hovermode="x",  # simplifies the data shown in the hover tooltip
        title_x=0.5,  # centers the plot title
        legend=dict(
            orientation="h", title_text="", y=-0.15
        ),  # puts the legend along the bottom of the plot
        font_size=20,
    ),
    update_yaxes_dict=dict(
        title=None,  # if you are using a good title or facet_rows, this the y-axis title is often redundant
        matches=None,  # in faceted plots, y-axis value scales will be independent of each other
    ),
    kwargs=dict(height=600, width=1400),
)
# Adds the 3-letter day name to the hover data
# define custom hovertemplate, plotlys formatting is dumb:/
# https://community.plotly.com/t/plotly-hovertemplate-date-time-format/39944/6
# DEFAULT_UPDATE_TRACES_DICT = dict(
#     mode="markers+lines", hovertemplate="<br>%{y} - %{x|%a}"
# )


BAR_PLOT_AWESOME_DEFAULTS = dict(
    update_layout_dict=dict(hoverlabel=dict(font_size=16), hovermode="x", title_x=0.5),
    update_yaxes_dict=dict(
        # if you are using a good title or facet_rows,
        # this the y-axis title is often redundant
        title=None,
        # in faceted plots, y-axis value scales will be independent of each other
        matches=None,
    ),
    update_traces_dict=dict(hovertemplate="%{y}"),
)


def _get_y_values(kwarg_dict: Dict) -> pd.Series:
    """
    Get the y values from the DataFrame or the y argument
    Parameters
    ----------
    kwarg_dict

    Returns
    -------

    """
    if "data_frame" not in kwarg_dict or "y" not in kwarg_dict or "x" not in kwarg_dict:
        warnings.warn(
            "\n\nYou most likely want to be using plotly express parameters like "
            "data_frame, y & x!!!\n"
        )

    y = kwarg_dict["y"]

    if "data_frame" in kwarg_dict:
        y = kwarg_dict["data_frame"][y]

    return y


def _infer_yaxes_tickformat(title, y_values, small_std_threshold) -> str:
    """

    Parameters
    ----------
    title
    y_values
    small_std_threshold

    Returns
    -------

    """
    title_lower = title.lower()

    # Determine if the metric is supposed to be dollars
    is_dollars = bool(re.search(r"(\$|dollar)", title_lower))
    # Determine if the metric is supposed to be percent-formatted
    is_percentage = bool(re.search("(%|percent|pct|mape)", title_lower))
    # Is there a small enough range that requires decimals in the numeric formatting?
    is_small_std = np.std(y_values) < small_std_threshold

    # Set the appropriate format: dollars, percentages or fractional formats
    if is_dollars:
        tickformat = "$,.2f" if is_small_std else "$,.0f"
    elif is_percentage:
        tickformat = "0.1%" if is_small_std else "0%"
    else:
        tickformat = ",.4f" if is_small_std else ",.0f"

    return tickformat


def _create_update_dicts(
    update_layout_dict: Dict,
    update_xaxes_dict: Dict,
    update_yaxes_dict: Dict,
    update_traces_dict: Dict,
) -> Dict:
    """
    Store the "update dict" parameters in a dict and replace None with an empty dict

    Parameters
    ----------
    update_layout_dict
    update_xaxes_dict
    update_yaxes_dict
    update_traces_dict

    Returns
    -------
    a dict containing dicts

    """
    update_dicts = dict(
        update_layout_dict=update_layout_dict,
        update_xaxes_dict=update_xaxes_dict,
        update_yaxes_dict=update_yaxes_dict,
        update_traces_dict=update_traces_dict,
    )

    for k, v in update_dicts.items():
        if v is None:
            update_dicts[k] = dict()

    return update_dicts


def _add_new_keys(existing_dict: Dict, new_dict: Dict) -> None:
    """

    Parameters
    ----------
    existing_dict
    new_dict

    Returns
    -------

    """
    for k, v in new_dict.items():
        if k not in existing_dict:
            existing_dict.update({k: v})

    return None


def _update_all_dicts(
    awesome_defaults: Dict, update_dicts: Dict, kwarg_dict: Dict
) -> None:
    """

    Parameters
    ----------
    awesome_defaults
    update_dicts
    kwarg_dict

    Returns
    -------

    """
    for k, v in awesome_defaults.items():
        if k == "kwargs":
            _add_new_keys(kwarg_dict, v)
        elif k in update_dicts:
            _add_new_keys(update_dicts[k], v)

    return None


def _create_figure(
    px_fig_func: Callable,
    update_layout_dict: Dict,
    update_xaxes_dict: Dict,
    update_yaxes_dict: Dict,
    update_traces_dict: Dict,
    **kwargs,
) -> go.Figure:
    """

    Parameters
    ----------
    px_fig_func: either px.bar or px.line
    update_layout_dict, update_xaxes_dict, update_yaxes_dict & update_traces_dict:
    key-value pairs passed to the corresponding method
    kwargs: key-value pairs to pass to px_fig_func

    Returns
    -------

    """
    return (
        px_fig_func(**kwargs)
        .update_layout(**update_layout_dict)
        .update_yaxes(**update_xaxes_dict)
        .update_xaxes(**update_yaxes_dict)
        .update_traces(**update_traces_dict)
        .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    )


def plot_line_plot(
    use_awesome_defaults: bool = True,
    infer_yaxes_tickformat: bool = True,
    small_std_threshold: Optional[Union[int, float]] = 1,
    zero_line_threshold: Optional[Union[int, float]] = 0.05,
    update_layout_dict: Optional[Dict] = None,
    update_xaxes_dict: Optional[Dict] = None,
    update_yaxes_dict: Optional[Dict] = None,
    update_traces_dict: Optional[Dict] = None,
    **kwargs,
) -> go.Figure:
    """

    Create a plotly line plot

    Parameters
    ----------
    small_std_threshold
    use_awesome_defaults: These are superior defaults. You can choose to specify some of them and not others
    infer_yaxes_tickformat: should the y-axis tickformat be determined via logic?
    zero_line_threshold: a dotted line will be drawn if any value in y is below this threshold.
        Set to None to turn it off.

    update_layout_dict, update_xaxes_dict, update_yaxes_dict & update_traces_dict:
    key-value pairs passed to the corresponding method

    kwargs: key-value pairs to pass to px.line

    Returns
    -------
    px.line plot
    """
    # Store the "update dict" parameters in a dict and replace None with an empty dict
    update_dicts: Dict = _create_update_dicts(
        update_layout_dict, update_xaxes_dict, update_yaxes_dict, update_traces_dict
    )

    y_values = _get_y_values(kwargs)

    if "tickformat" not in update_dicts["update_yaxes_dict"] and (
        infer_yaxes_tickformat or use_awesome_defaults
    ):
        title: str = kwargs.get("title", "")
        update_dicts["update_yaxes_dict"]["tickformat"]: str = _infer_yaxes_tickformat(
            title, y_values, small_std_threshold
        )

    # Use some or all of the awesome defaults
    if use_awesome_defaults:
        # update the values if the key does not exist
        _update_all_dicts(LINE_PLOT_AWESOME_DEFAULTS, update_dicts, kwargs)
        # _add_new_keys(update_traces_dict, LINE_PLOT_AWESOME_DEFAULTS)

    fig = _create_figure(px.line, **update_dicts, **kwargs)

    # If zero_line_threshold is a number & if any value is less than zero,
    # then add a thick black line at zero.
    if (
        isinstance(zero_line_threshold, (int, float))
        and (y_values < zero_line_threshold).any()
    ):
        fig.add_hline(y=0, line_width=3, line_dash="dash")

    return fig


def plot_bar_plot(
    use_awesome_defaults: bool = True,
    update_layout_dict: Optional[Dict] = None,
    update_xaxes_dict: Optional[Dict] = None,
    update_yaxes_dict: Optional[Dict] = None,
    update_traces_dict: Optional[Dict] = None,
    n_text_decimals: Optional[Union[int, float]] = 3,
    **kwargs,
) -> go.Figure:
    """

    Create a plotly bar plot

    Parameters
    ----------
    n_text_decimals
    use_awesome_defaults: these are some reasonable plotting parameters.
        You can choose to specify some of them and not others

    update_layout_dict, update_xaxes_dict, update_yaxes_dict & update_traces_dict:
    key-value pairs passed to the corresponding method

    kwargs: key-value pairs to pass to px.bar

    Returns
    -------
    px.bar plot
    """

    # Store the "update dict" parameters in a dict and replace None with an empty dict
    update_dicts: Dict = _create_update_dicts(
        update_layout_dict, update_xaxes_dict, update_yaxes_dict, update_traces_dict
    )

    y_values = _get_y_values(kwargs)

    # If `text` is not provided and `n_text_decimals` is greater than 0,
    # try to add some plot text on the bars
    if (
        "text" not in kwargs
        and n_text_decimals >= 0
        and pd.api.types.is_numeric_dtype(y_values)
    ):
        kwargs["text"] = y_values.round(n_text_decimals)

    if use_awesome_defaults:
        _update_all_dicts(BAR_PLOT_AWESOME_DEFAULTS, update_dicts, kwargs)

    return _create_figure(px.bar, **update_dicts, **kwargs)


def plot_many_plots(
    df: pd.DataFrame,
    plot_func: Callable,
    iterate_col: Optional[str] = "metric",
    fig_func: Callable = None,
    **plot_func_kwargs,
) -> None:
    """
    Iterate through a DataFrame and Display each DataFrame Subset as a Plot.

    This is useful when you are using facet_row for an attribute
    and you have many metrics using different scales that you want to plot.

    Parameters
    ----------
    df: DataFrame
    plot_func: a function that returns a figure, like plot_line_plot
    iterate_col: name of the column containing the values to iterate through
    fig_func: apply a function to the figure if necessary.
        examples: lambda fig: fig.show() or st.plotly_chart
    plot_func_kwargs: key-value pairs for plot_func

    Returns
    -------
    None
    """

    if "data_frame" in plot_func_kwargs:
        warnings.warn(
            "\nDid you mean to you the df parameter? "
            "Using the data_frame parameter will not work."
        )

    for v in df[iterate_col].drop_duplicates().sort_values():
        data_frame = df.loc[lambda df: df[iterate_col].eq(v)]

        fig = plot_func(data_frame=data_frame, title=v, **plot_func_kwargs)

        fig_func(fig) if fig_func else fig

    return None


# specific instance of plot_many_plots for line plots
plot_many_line_plots = partial(plot_many_plots, plot_func=plot_line_plot)

# specific instance of plot_many_plots for line plots
plot_many_bar_plots = partial(plot_many_plots, plot_func=plot_bar_plot)


@dataclass
class FeatureImportancePlot:
    """

    Parameters
    ----------
    features
    importance_values
    max_scale
    n_decimals_displayed
    str_pad_width

    Returns
    -------

    """

    features: List
    importance_values: List
    max_scale: bool = True
    n_decimals_displayed: int = 1
    str_pad_width: int = 15

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
        top_n_features=None,
        height_per_feature=25,
        width: int = 750,
        yaxes_tickfont_family="Courier New, monospace",
        yaxes_tickfont_size=15,
        **kwargs,
    ):
        """

        Plot the Feature Names & Importances

        Parameters
        ----------
        top_n_features : the number of features to plot, default is 100
        max_scale : Should the importance values be scaled by the maximum value & mulitplied by 100?  Default is True.
        n_decimals_displayed : How many decimal places should be displayed. Default is 1.
        height_per_feature : if height is None, the plot height is calculated by top_n_features * height_per_feature.
        This allows all the features enough space to be displayed
        width :  the width of the plot
        str_pad_width : When rank_features=True, this number of spaces to add between the rank integer and feature name.
            This will enable the rank integers to line up with each other for easier reading.
            Default is 15. If you have long feature names, you can increase this number to make the integers line up more.
            It can also be set to 0.
        yaxes_tickfont_family : the font for the feature names. Default is Courier New.
        yaxes_tickfont_size : the font size for the feature names. Default is 15.

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
            .update_layout(title_x=0.5, showlegend=False)
            .update_yaxes(
                tickfont=dict(family=yaxes_tickfont_family, size=yaxes_tickfont_size),
                title="",
            )
        )
