import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("../answers/company_answers.csv")

# Function to create and return the histogram for luck or work
def create_histogram(column_name):
    fig = px.histogram(df, x=column_name, nbins=10, title=f"Distribution of {column_name.capitalize()}")
    return fig

# Display histograms
st.header("Distribution of Scores")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_histogram('luck'), use_container_width=True)
with col2:
    st.plotly_chart(create_histogram('work'), use_container_width=True)

# Interactive table
st.header("Company Scores")
st.dataframe(df)  # This displays the table which by default is interactive in Streamlit

# To make the table update based on histogram selection, additional callback functions are needed.
# This feature requires more advanced handling of interactions between Plotly and Streamlit.
# Streamlit's `session_state` can be used to store filter conditions and update the displayed DataFrame.
# However, directly linking clicks from a Plotly chart to filter the Streamlit table involves more complex JavaScript callbacks, which are not natively supported by Streamlit.

# The example above provides a basic setup. For more dynamic interactions, consider exploring custom components or alternative approaches to achieve direct linkage between chart selections and the DataFrame view.
