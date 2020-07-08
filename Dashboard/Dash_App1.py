from dash import Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from datetime import datetime
# from .Dash_fun import apply_layout_with_auth,load_object, save_object
import pandas as pd

import array as arr

url_base = '/dash/app1/'
languages = ["German", "English", "Portuguese", "Slovene", "Spanish"]
layout = html.Div([

    html.Div([
        html.Label('Domains'),
        dcc.Dropdown(
            id='Domains',
            options=[
                {'label': 'Global Warming', 'value': 'GW'},
                {'label': 'FIFA World Cup', 'value': 'FF'},
                {'label': 'Earthquake', 'value': 'EQ'}
            ],
            value='GW',
        )],
        style={'margin-left': '25%', 'height': '30px', 'width': '15%', 'display': 'inline-block',
               'margin-bottom': '20px'}),

    html.Div([
        html.Label('Languages'),
        dcc.Dropdown(
            id='language',
            options=[{'label': i, 'value': i} for i in languages],
            value='',
            placeholder='Select a language',
            multi=False
        )
    ],
        style={'margin-left': '20px', 'width': '15%', 'display': 'inline-block', 'margin-bottom': '20px'}),

    html.Div([
        html.Label('Similarity Score'),
        dcc.Slider(
            id='score-slider',
            min=0,
            max=1.1,
            value=0.7,
            step=None,
            marks={'0.1': '>0.1', '0.3': '>0.3', '0.5': '>0.5', '0.7': '>0.7', '0.8': '>0.8', '0.9': '>0.9'}
        ),
    ],
        style={'width': '20%', 'display': 'inline-block', 'margin-bottom': '20px', 'margin-left': '20px'}),

    html.Div([
        html.Label('Years'),
        dcc.Slider(
            id='year-slider',
            min=2015,
            max=2020,
            value=2015,
            step=None,
            marks={'2015': '2015', '2016': '2016', '2017': '2017', '2018': '2018', '2019': '2019', '2020': '2020'}
        ),
    ],
        style={'width': '20%', 'display': 'inline-block', 'margin-bottom': '20px', 'margin-left': '20px'}),

    html.Div([
        dcc.Graph(id='life-exp-vs-gdp'),
    ],
        style={'margin-left': '25%', 'margin-right': '10px', 'width': '73%'}),
])

def add_dash(server):

    app = Dash(server=server, url_base_pathname=url_base)
    app.layout = layout
    app.css.config.serve_locally = False
    app.scripts.config.serve_locally = False
    # app.config['suppress_callback_exceptions'] = True
    # apply_layout_with_auth(app, layout)

    app.css.append_css({
        "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
    })

    @app.callback(
        dash.dependencies.Output('life-exp-vs-gdp', 'figure'),
        [dash.dependencies.Input('score-slider', 'value'),
         dash.dependencies.Input('language', 'value'),
         dash.dependencies.Input('year-slider', 'value'),
         dash.dependencies.Input('Domains', 'value')
         ])

    def update_graph(score, language, year, dataSet):
        df = pd.read_csv('data/CCEdgesAllTemporal.csv')
        if dataSet == "GW":
            df = pd.read_csv('data/CCEdgesAllTemporal.csv')
        elif dataSet == "FF":
            df = pd.read_csv('data/SPEdgesAllTemporal.csv')
        elif dataSet == "EQ":
            df = pd.read_csv('data/NDEdgesAllTemporal.csv')

        filtered_df = df.loc[df["weight"] > score]
        filtered_df['source-time'] = pd.to_datetime(filtered_df['source-time'], format="%Y-%m-%dT%H:%M:%S")
        filtered_df = filtered_df[filtered_df['source-time'].dt.year == year]
        print(language);

        if language != '' and language is not None:
            print(language);
            if language == 'German':
                filtered_df = filtered_df[filtered_df['source'].str.contains('German')]
                print(language);
            elif language == 'English':
                filtered_df = filtered_df[filtered_df['source'].str.contains('English')]

            elif language == 'Portuguese':
                filtered_df = filtered_df[filtered_df['source'].str.contains('Por')]

            elif language == 'Slovene':
                filtered_df = filtered_df[filtered_df['source'].str.contains('Slovene')]

            elif language == 'Spanish':
                filtered_df = filtered_df[filtered_df['source'].str.contains('Spanish')]

        traces = []
        co = ["salmon", "turquoise", "purple", "lightskyblue", "slategray"]
        count = 0
        seColor = 0
        for i in filtered_df['source']:
            df_by_continent = filtered_df[filtered_df['source'] == i]
            if 'German' in i:
                seColor = 0
            elif 'English' in i:
                seColor = 1
            elif 'Por' in i:
                seColor = 2
            elif 'Slovene' in i:
                seColor = 3
            elif 'Spanish' in i:
                seColor = 4

            traces.append(go.Scatter(
                x=pd.to_datetime(df_by_continent['source-time'], format="%Y-%m-%dT%H:%M:%S"),
                y=df_by_continent['weight'],
                text=df_by_continent['source'],
                mode='markers',
                opacity=1.0,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'},
                    'color': co[seColor]
                },
                name=i
            ))

        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Years', 'titlefont': dict(size=18, color='darkgrey'),
                       'range': ['2015-01-01T1-1-0Z', '2020-01-01T1-1-0Z'], 'zeroline': False, 'ticks': 'outside'},
                yaxis={'title': 'Similarity', 'titlefont': dict(size=18, color='darkgrey'), 'range': [0, 1.1],
                       'ticks': 'outside'},
                margin={'l': 60, 'b': 60, 't': 30, 'r': 20},
                legend={'x': 1, 'y': 1},
                hovermode='closest'
            )
        }

    return app.server
