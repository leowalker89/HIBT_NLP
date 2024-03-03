import dash
from dash import dcc, html, Input, Output
import dash_table
import plotly.express as px
import pandas as pd

# Create a dataframe from our csv file
df = pd.read_csv("../answers/company_answers.csv")

# Initialize Dash App
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='luck-histogram', figure=px.histogram(df, x='luck', title="Luck Distribution")),
        dcc.Graph(id='work-histogram', figure=px.histogram(df, x='work', title="Work Distribution"))
    ], style={'display': 'flex', 'flexDirection': 'row'}),  # This places the graphs side by side
    
    dash_table.DataTable(
        id='company-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        filter_action="native",  # Enables filtering
        sort_action="native",  # Enables sorting
        style_table={'overflowX': 'auto'},  # Allows horizontal scrolling for the table itself
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',  # Adjusts row height based on content
        },
        style_cell={'textAlign': 'left', 'padding': '5px'},  # Styles for text alignment and padding
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
    )
])



# Example callback (more complex interactions would require more detailed callbacks)
@app.callback(
    Output('company-table', 'data'),
    [Input('luck-histogram', 'clickData'), Input('work-histogram', 'clickData')])
def filter_table(luck_click_data, work_click_data):
    # Your filtering logic here
    # For now, just return the unfiltered data
    return df.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)