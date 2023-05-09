import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.tools import mpl_to_plotly
import math
import plotly.graph_objs as go
from statsmodels.tsa.stattools import acf, pacf, adfuller


import toolkit

np.random.seed(6313)




# load data
url = 'https://raw.githubusercontent.com/tanmayk26/Air-Pollution-Forecasting/main/LSTM-Multivariate_pollution.csv'
df = pd.read_csv(url, index_col='date')
date = pd.date_range(start='1/2/2010',
                     periods=len(df),
                     freq='H')
df.index = date
df['Date'] = date

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Air Pollution Timeseries Forecasting"

# create the app
#app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# define the layout for each tab
tab1_layout = html.Div([
    html.H1("Data Distribution"),
    html.Div([
        dcc.Dropdown(
            id='target-selector',
            options=[
                {'label': 'Pollution', 'value': 'pollution'},
                {'label': 'Seasonal Difference (s=24)', 'value': 's_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'ns_diff'}
            ],
            value='pollution'
        ),

        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),

    html.Div([
        html.Label("Date range:"),
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            start_date=df.index.min().date(),
            end_date=df.index.max().date(),
        ),
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div(
        dcc.Graph(id="pollution-chart", style={'height': '500px'}, config={"displayModeBar": False})
    ),
])

tab2_layout = html.Div([
    html.H1("Tab2"),

    html.Div([
        html.Label("lags:"),
        dcc.Slider(
            id="lag-slider",
            min=10,
            max=100,
            step=10,
            value=20,
        ),
    ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'middle'}),


    dcc.Graph(id="acf-pacf-graph", style={'height': '500px'}),
])

# define the callbacks for each tab
@app.callback(
    Output("pollution-chart", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input('submit-button', 'submit'),
    State('target-selector', 'value')
)
def update_charts(submit, start_date, end_date, variable):
    filtered_data = df.query(
        "Date >= @start_date and Date <= @end_date"
    )
    if variable == 's_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(filtered_data['pollution'], seasons=s)
    elif variable == 'ns_diff':
        filtered_data = toolkit.differencing(filtered_data['seasonal_d_o_1'], 24)
    else:
        filtered_data = filtered_data['pollution']
    print(filtered_data)
    pollution_time_chart_figure = {
        "data": [
            {
                "x": filtered_data.index,
                "y": filtered_data["pollution"],
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Air Pollution over time",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    return pollution_time_chart_figure


@app.callback(
    Output("acf-pacf-graph", "figure"),
    Input("lag-slider", "value")
)
def update_tab2(lags):
    mat_fig = toolkit.ACF_PACF_Plot(df['pollution'], lags=lags)
    fig = mpl_to_plotly(mat_fig)
    return fig

# create the app layout and add the tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Data Distribution', value='tab1', children=[tab1_layout]),
        dcc.Tab(label='Tab-2', value='tab2', children=[tab2_layout]),
        # dcc.Tab(label='Calculator', value='tab3', children=[tab3_layout]),
        # dcc.Tab(label='Polynomial Function', value='tab4', children=[tab4_layout]),
        # dcc.Tab(label='Sinusoidal Function', value='tab5', children=[tab5_layout]),
        # dcc.Tab(label='Neural Network', value='tab6', children=[tab6_layout]),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=False)