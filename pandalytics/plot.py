"""
Contains custom plotting functions
"""
from functools import partial
from typing import Optional, Union, Dict, Callable
from numpy.typing import ArrayLike
import re
import warnings

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


def _get_y_values(kwarg_dict: Dict) -> ArrayLike:
    """
    Get the y values from the DataFrame or the y argument
    Parameters
    ----------
    kwarg_dict: the plot's kwargs dict

    Returns
    -------
    an array of y values

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


def _infer_yaxes_tickformat(
    y_values: ArrayLike,
    small_std_threshold: Union[float, int],
    title: Optional[str] = "",
) -> str:
    """

    Infer if the tickformat should be expressed as dollars or percentages and how many
    decimal it should use.

    Parameters
    ----------
    title: The plot's title is used to see if the values should be formatted
        as percentages or dollars.
    y_values: the plot's y values
    small_std_threshold: if the y_values standard deviation is smaller than this,
        use fewer decimal places.

    Returns
    -------
    the tickformat string

    """
    title_lower = title.lower()

    # Determine if the metric is supposed to be dollars
    is_dollars = bool(re.search(r"(\$|dollar)", title_lower))
    # Determine if the metric is supposed to be percent-formatted
    is_percentage = bool(re.search("(%|percent|pct|mape)", title_lower))
    # Is there a small enough range that requires decimals in the numeric formatting?
    is_small_std = np.std(y_values) <= small_std_threshold

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
    update_layout_dict, update_xaxes_dict, update_yaxes_dict & update_traces_dict:
        key-value pairs passed to the corresponding method

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


def _add_new_keys(source_dict: Dict, target_dict: Dict) -> None:
    """
    If the key does not exist in the target dict, add the key-value from the source dict.

    Parameters
    ----------
    source_dict: the source of the new key-values
    target_dict: the dict to update

    Returns
    -------
    None

    """
    for k, v in source_dict.items():
        if k not in target_dict:
            target_dict.update({k: v})

    return None


def _update_all_dicts(
    awesome_defaults: Dict, update_dicts: Dict, kwarg_dict: Dict
) -> None:
    """
    Update the update_dicts and kwarg_dict using awesome_defaults dict

    Parameters
    ----------
    awesome_defaults: source of the default values
    update_dicts: target to update
    kwarg_dict:target to update

    Returns
    -------
    None
    """
    for k, v in awesome_defaults.items():
        if k == "kwargs":
            _add_new_keys(v, kwarg_dict)
        elif k in update_dicts:
            _add_new_keys(v, update_dicts[k])

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
    a plot

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
            y_values, small_std_threshold, title
        )

    # Use some or all of the awesome defaults
    if use_awesome_defaults:
        # update the values if the key does not exist
        _update_all_dicts(LINE_PLOT_AWESOME_DEFAULTS, update_dicts, kwargs)
        # _add_new_keys(LINE_PLOT_AWESOME_DEFAULTS, update_traces_dict)

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

# specific instance of plot_many_plots for bar plots
plot_many_bar_plots = partial(plot_many_plots, plot_func=plot_bar_plot)
