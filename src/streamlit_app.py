import streamlit as st
import pandas as pd
import plotly.express as px

# Set page to wide mode
st.set_page_config(layout="wide")

df = pd.read_csv("answers/company_answers_cleaned.csv")

df = df.drop(columns=['summary'])

import plotly.io as pio
pio.json.config.default_engine = 'orjson'

# Function to create horizontal histograms
def create_histogram(column_name):
    fig = px.histogram(df, y=column_name, nbins=11, orientation='h', 
                       title=f"Distribution of {column_name.capitalize()}")
    fig.update_layout(bargap=0.1)
    return fig


# Sidebar for filters and sorting options
st.sidebar.title("Filters and Sorting")

# Filters in the sidebar - with a default empty option for 'no filter'
# Sort the unique values alphabetically before adding them to the dropdown
company_filter = st.sidebar.selectbox(
    'Filter by company',
    [''] + sorted(list(df['company'].unique())),
    index=0,
    key='company_filter'
)

guest_filter = st.sidebar.selectbox(
    'Filter by guest',
    [''] + sorted(list(df['guest'].unique())),
    index=0,
    key='guest_filter'
)

# Sorting options in the sidebar
sort_by = st.sidebar.selectbox('Sort by', ['luck', 'work', 'people'])
sort_order = st.sidebar.selectbox('Sort order', ['Ascending', 'Descending'], index=1)

# Apply filters if a specific company or guest is selected
filtered_df = df
if company_filter != '':
    filtered_df = filtered_df[filtered_df['company'].str.contains(company_filter, case=False)]
if guest_filter != '':
    filtered_df = filtered_df[filtered_df['guest'].str.contains(guest_filter, case=False)]

ascending = sort_order == 'Ascending'
filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

# Main page layout with 2 columns, adjusted for wide mode
col1, col2 = st.columns([3, 1])  # You may not need to change the ratio for wide mode

# Display the quotes in the left column (col1)
with col1:
    st.title('Entrepreneurs Answers')
    for index, row in filtered_df.iterrows():
        st.markdown(f'{row["guest"]}, {row["company"]}\n\nLuck: {row["luck"]}, Work: {row["work"]}, People: {row["people"]},\n\n "{row["answer"]}"\n\n')
        st.divider()

# Display histograms in the right column (col2)
with col2:
    st.header("Distributions")
    st.plotly_chart(create_histogram('luck'), use_container_width=True)
    st.plotly_chart(create_histogram('work'), use_container_width=True)
