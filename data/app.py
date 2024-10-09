import os
import pandas as pd
from flask import Flask, send_from_directory
import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from ticker_utils import remove_microcaps, remove_low_vol
# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Load data
file_path = 'nasdaq_screener_all.csv'
df = pd.read_csv(file_path)

# Convert 'Market Cap' to numeric, errors='coerce' will turn non-numeric values into NaN
df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

# Drop rows with NaN values in 'Market Cap'
df = df.dropna(subset=['Market Cap'])

#Drop IPO year and country
df = df.drop(columns=['IPO Year', 'Country'])

#Further data processing
df = remove_microcaps(df) # Comment this in order to uncomment Micro,Nano Caps
df = remove_low_vol(df)


# Define market cap categories
categories = {
    'Large Cap': (10**10, float('inf')),
    'Mid Cap': (2*10**9, 10**10),
    'Small Cap': (300*10**6, 2*10**9),
#    'Micro Cap': (50*10**6, 300*10**6),
#    'Nano Cap': (0, 50*10**6)
}

# Create a list of unique sectors
sectors = df['Sector'].unique()


# Layout of the Dash app
app.layout = html.Div([
    html.H1("Stock Screener Dashboard"),
    html.Div([
        html.Label("Select Sector:"),
        dcc.Dropdown(
            id='sector-dropdown',
            options=[{'label': sector, 'value': sector} for sector in sectors],
            value=sectors[0]
        )
    ]),
    html.Div([
        html.Label("Select Market Cap Category:"),
        dcc.Dropdown(
            id='marketcap-dropdown',
            options=[{'label': category, 'value': category} for category in categories.keys()],
            value='Large Cap'
        )
    ]),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        filter_action='native',
        sort_action='native',
        sort_mode='multi',
        column_selectable='single',
        row_selectable='multi',
        selected_columns=[],
        selected_rows=[],
        page_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'height': 'auto',
            'minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal'
        }
    ),
])

# Callback to update the table based on selected sector and market cap category
@app.callback(
    Output('table', 'data'),
    [Input('sector-dropdown', 'value'),
     Input('marketcap-dropdown', 'value')]
)
def update_table(selected_sector, selected_marketcap):
    min_cap, max_cap = categories[selected_marketcap]
    filtered_df = df[(df['Sector'] == selected_sector) & (df['Market Cap'] >= min_cap) & (df['Market Cap'] < max_cap)]
    sorted_df = filtered_df.sort_values(by='Market Cap', ascending=False)
    return sorted_df.to_dict('records')

# Serve the Dash app
@server.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

if __name__ == '__main__':
    server.run(debug=True)