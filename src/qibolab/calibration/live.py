# -*- coding: utf-8 -*-
import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from qibolab.paths import qibolab_folder

# data = np.load("./buffer.npy")
# X = data[0]
# Y = data[1]
# print(X)
# print(Y)

app = dash.Dash(__name__)


def live_plotting(path):
    # path = qibolab_folder / 'calibration' / 'data' / 'buffer.npy'
    print(path)
    app.layout = html.Div(
        [
            dcc.Graph(id="live-graph", animate=False),
            dcc.Interval(id="graph-update", interval=1000, n_intervals=0),
        ]
    )

    @app.callback(Output("live-graph", "figure"), [Input("graph-update", "n_intervals")])
    def update_graph_scatter(n):
        df = np.load(path)
        X = df[0]
        Y = df[1]

        if len(df) == 2:
            data = go.Scatter(x=X, y=Y, name="Scatter", mode="lines+markers")

        elif len(df) == 3:
            Z = df[2]
            data = go.Heatmap(x=X, y=Y, z=Z, name="Scatter")

        else:
            raise ValueError("Not plottable")

        return {
            "data": [data],
            "layout": go.Layout(
                xaxis=dict(range=[min(X), max(X)]),
                yaxis=dict(range=[min(Y), max(Y)]),
            ),
        }


def start_server(path):
    live_plotting(path)
    app.run_server()
