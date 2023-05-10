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
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
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
s=24


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
                {'label': 'Seasonal Difference (s=24)', 'value': 'seasonal_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'nonseasonal_diff'}
            ],
            value='pollution'
        ),

        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    html.Div(
        dcc.Graph(id="pollution-chart",
                  style={'height': '500px'},
                  config={"displayModeBar": False})
    ),
])

tab2_layout = html.Div([
    html.H1("ACF/PACF Plot"),
    html.Div([
        dcc.Dropdown(
            id='target-selector2',
            options=[
                {'label': 'Pollution', 'value': 'pollution'},
                {'label': 'Seasonal Difference (s=24)', 'value': 'seasonal_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'nonseasonal_diff'}
            ],
            value='pollution'
        ),

        html.Button('Submit', id='submit-button2', n_clicks=0)
    ]),
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

tab3_layout = html.Div([
    html.H1("Stationarity - Rolling Mean/Variance"),
    html.Div([
        dcc.Dropdown(
            id='target-selector3',
            options=[
                {'label': 'Pollution', 'value': 'pollution'},
                {'label': 'Seasonal Difference (s=24)', 'value': 'seasonal_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'nonseasonal_diff'}
            ],
            value='pollution'
        ),

        html.Button('Submit', id='submit-button3', n_clicks=0)
    ]),

    dcc.Graph(id="rolling-mean-var-graph", style={'height': '500px'}),

])

tab4_layout = html.Div([
    html.H1("Stationarity - ADF/KPSS"),
    html.Div([
        dcc.Dropdown(
            id='target-selector4',
            options=[
                {'label': 'Pollution', 'value': 'pollution'},
                {'label': 'Seasonal Difference (s=24)', 'value': 'seasonal_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'nonseasonal_diff'}
            ],
            value='pollution'
        ),

        html.Button('Submit', id='submit-button4', n_clicks=0)
    ]),

    html.Div(id="adf-test-output"),
    html.Div(id="kpss-test-result"),

])



# define the callbacks for each tab
@app.callback(
    Output("pollution-chart", "figure"),
    Input('submit-button', 'n_clicks'),
    State('target-selector', 'value')
)
def update_charts(n_clicks, target):
    if target == 'seasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
    elif target == 'nonseasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = toolkit.differencing(filtered_data, 24)
    else:
        filtered_data = df['pollution']
    figure = {
        "data": [
            {
                "x": df['Date'],
                "y": filtered_data,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": target
            },
        ],
        "layout": {
            "title": {
                "text": "Data - " + target,
                "x": 0.05,
                "xanchor": "left",
            },
        },
    }

    return figure


@app.callback(
    Output("acf-pacf-graph", "figure"),
    Input('submit-button2', 'n_clicks'),
    Input("lag-slider", "value"),
    State('target-selector2', 'value')
)
def update_tab2(n_clicks, lags, target):
    if target == 'seasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = filtered_data[s:]
    elif target == 'nonseasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = toolkit.differencing(filtered_data, 24)
        filtered_data = filtered_data[s+1:]
    else:
        filtered_data = df['pollution']
    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Scatter(x=list(range(int(lags)+1)), y=acf(filtered_data, nlags=int(lags)), name='ACF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(int(lags)+1)), y=pacf(filtered_data, nlags=int(lags)), name='PACF'), row=2, col=1)
    fig.update_layout(height=500, showlegend=False, title_text="ACF/PACF of the raw data")
    return fig



@app.callback(
    Output("rolling-mean-var-graph", "figure"),
    Input('submit-button3', 'n_clicks'),
    State('target-selector3', 'value')
)
def update_tab3(n_clicks, target):
    if target == 'seasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = pd.Series(filtered_data[s:])
    elif target == 'nonseasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = toolkit.differencing(filtered_data, 24)
        filtered_data = pd.Series(filtered_data[s + 1:])
    else:
        filtered_data = pd.Series(df['pollution'])

    rolling_mean = filtered_data.rolling(window=7).mean()
    rolling_var = filtered_data.rolling(window=7).var()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name='Rolling Mean'), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_var.index, y=rolling_var, name='Rolling Var'), row=2, col=1)
    fig.update_layout(title=f'Rolling Mean and Rolling Var of {target}', height=600)
    return fig



@app.callback(
    [Output("adf-test-output", "children"),
     Output("kpss-test-result", "children")],
    Input('submit-button4', 'n_clicks'),
    State('target-selector4', 'value')
)
def update_tab3(n_clicks, target):
    if target == 'seasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = pd.Series(filtered_data[s:])
    elif target == 'nonseasonal_diff':
        s = 24
        filtered_data = toolkit.seasonal_differencing(df['pollution'], seasons=s)
        filtered_data = toolkit.differencing(filtered_data, 24)
        filtered_data = pd.Series(filtered_data[s + 1:])
    else:
        filtered_data = pd.Series(df['pollution'])
    adf_result = adfuller(filtered_data)
    adf_output = f"ADF Test Results:\nADF Statistic: {adf_result[0]:.4f}\np-value: {adf_result[1]:.4f}\nCritical Values: {adf_result[4]}"
    kpss_result = kpss(filtered_data)
    kpss_output = f"KPSS test statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}"

    return adf_output, kpss_output


# create the app layout and add the tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Data Distribution', value='tab1', children=[tab1_layout]),
        dcc.Tab(label='ACF/PACF', value='tab2', children=[tab2_layout]),
        dcc.Tab(label='Stationarity', value='tab3', children=[tab3_layout]),
        dcc.Tab(label='Stationarity-ADF/KPSS', value='tab4', children=[tab4_layout]),
        # dcc.Tab(label='Sinusoidal Function', value='tab5', children=[tab5_layout]),
        # dcc.Tab(label='Neural Network', value='tab6', children=[tab6_layout]),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=False)