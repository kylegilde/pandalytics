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

    expected_fig = (
        "Figure({\n    'data': [{'alignmentgroup': 'True',\n              'hovertemplate': ('abs_value=%{x}"
        "<br>y_label=%{y}' ... '%{marker.color}<extra></extra>'),\n              'legendgroup': '',\n      "
        "        'marker': {'color': array([ 0.01087986, -0.02314951,  0.03048972,  0.03109561, -0.03204229,\n   "
        "                                       0.03342788, -0.03895749,  0.04447014, -0.04771265, "
        "-0.05404756,\n                                         -0.05690695, -0.09537289, -0.09896881, "
        "-0.12407396, -0.14729187,\n                                          0.1515505 , -0.17368409, "
        "-0.19568868, -0.20135828,  0.23431206,\n                                          0.28416893]),\n "
        "                        'coloraxis': 'coloraxis',\n                         'pattern': {'shape': ''}"
        "},\n              'name': '',\n              'offsetgroup': '',\n              'orientation': "
        "'h',\n              'showlegend': False,\n              'text': array([ 0.01, -0.02,  0.03,  0.03, "
        "-0.03,  0.03, -0.04,  0.04, -0.05, -0.05,\n                             -0.06, -0.1 , -0.1 , -0.12,"
        " -0.15,  0.15, -0.17, -0.2 , -0.2 ,  0.23,\n                              0.28]),\n            "
        "  'textfont': {'color': 'black'},\n              'textposition': 'auto',\n              "
        "'type': 'bar',\n              'x': array([0.01087986, 0.02314951, 0.03048972, 0.03109561, "
        "0.03204229, 0.03342788,\n                          0.03895749, 0.04447014, 0.04771265, 0.05404756, "
        "0.05690695, 0.09537289,\n                          0.09896881, 0.12407396, 0.14729187, 0.1515505 ,"
        " 0.17368409, 0.19568868,\n                          0.20135828, 0.23431206, 0.28416893]),\n     "
        "         'xaxis': 'x',\n              'y': array(['int_col_3 & date_col', 'float_col_2 & "
        "float_col',\n                          'float_col & date_col', 'float_col_3 & int_col',\n     "
        "                     'float_col & int_col_3', 'float_col_2 & int_col',\n                   "
        "       'int_col_3 & int_col_2', 'int_col_2 & date_col',\n                          "
        "'float_col_2 & int_col_2', 'float_col_3 & int_col_2',\n                          "
        "'int_col_2 & int_col', 'float_col_3 & date_col',\n                          "
        "'float_col_2 & date_col', 'int_col & date_col', 'float_col & int_col_2',\n                         "
        " 'float_col_3 & float_col_2', 'int_col_3 & int_col',\n                          "
        "'float_col_2 & int_col_3', 'float_col_3 & int_col_3',\n                        "
        "  'float_col & int_col', 'float_col_3 & float_col'], dtype=object),\n              "
        "'yaxis': 'y'}],\n    'layout': {'barmode': 'relative',\n               "
        "'coloraxis': {'cmax': 1,\n                             'cmid': 0,\n                             "
        "'cmin': -1,\n                             'colorbar': {'title': {'text': 'value'}},\n           "
        "                  'colorscale': [[0.0, 'red'], [0.5, 'white'], [1.0,\n                          "
        "                  'green']]},\n               'font': {'color': 'black'},\n              "
        " 'hovermode': False,\n               'legend': {'tracegroupgap': 0},\n              "
        " 'plot_bgcolor': 'lightgray',\n               'template': '...',\n               'title': "
        "{'text': '21 Pairwise Correlations', 'x': 0.5},\n               'xaxis': {'anchor': 'y', 'domain': "
        "[0.0, 1.0], 'range': [0, 1], 'title': {'text': ''}},\n               'yaxis': {'anchor':"
        " 'x', 'domain': [0.0, 1.0], 'title': {'text': ''}}}\n})"
    )

    assert (
        str(fig) == expected_fig
    ), "The PairwiseCorrelations plot did NOT render as expected."
