import streamlit as st
import pandas as pd

from utils.data_cleaning import convert_to_float


def main():
    st.title("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        if 'New #' in df.columns:
            df = df.drop(columns=['New #'])
        df = convert_to_float(df)

        st.session_state['data'] = df  # Store the cleaned data in session state

        st.subheader("Cleaned Data Preview")
        st.dataframe(df)

        st.subheader("Data Summary")
        st.write(df.describe())

        st.subheader("Basic Data Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Distribution of 'Country':")
            plotting_data = {
                'data': {
                    "x": df['Country'].value_counts().index,
                    "y": df['Country'].value_counts().values,
                    'type': 'bar',
                    'name': 'Country Distribution',
                    'marker': {'color': 'green'},
                    'text': df['Country'].value_counts().values
                },
                'layout': {
                    'title': 'Country Distribution',
                    'xaxis': {'title': 'Country'},
                    'yaxis': {'title': 'Count'},
                    'template': 'plotly_white'
                }
            }
            st.plotly_chart(plotting_data)
        with c2:
            st.write("Distribution of 'Flavor':")
            plotting_data = {
                'data': {
                    "x": df['Flavor'].value_counts().index,
                    "y": df['Flavor'].value_counts().values,
                    'type': 'bar',
                    'name': 'Flavor Distribution',
                    'marker': {'color': 'green'},
                    'text': df['Flavor'].value_counts().values
                },
                'layout': {
                    'title': 'Flavor Distribution',
                    'xaxis': {'title': 'Flavor'},
                    'yaxis': {'title': 'Count'},
                    'template': 'plotly_white'
                }
            }
            st.plotly_chart(plotting_data)
        c1, c2 = st.columns(2)
        with c1:
            # Country + Total Flavor Spend Bar chart
            st.write("Total Flavor Spend by Country:")
            total_spend_by_country = df.groupby('Country')['Flavor Spend'].sum().reset_index()
            plotting_data = {
                'data': {
                    "x": total_spend_by_country['Country'],
                    "y": total_spend_by_country['Flavor Spend'],
                    'type': 'bar',
                    'name': 'Total Flavor Spend by Country',
                    'marker': {'color': 'green'},
                    'text': total_spend_by_country['Flavor Spend']
                },
                'layout': {
                    'title': 'Total Flavor Spend by Country',
                    'xaxis': {'title': 'Country'},
                    'yaxis': {'title': 'Total Flavor Spend'},
                    'template': 'plotly_white'
                }
            }
            st.plotly_chart(plotting_data)
        with c2:
            # Flavor + Total Flavor Spend Bar chart
            st.write("Total Flavor Spend by Flavor:")
            total_spend_by_flavor = df.groupby('Flavor')['Flavor Spend'].sum().reset_index()
            plotting_data = {
                'data': {
                    "x": total_spend_by_flavor['Flavor'],
                    "y": total_spend_by_flavor['Flavor Spend'],
                    'type': 'bar',
                    'name': 'Total Flavor Spend by Flavor',
                    'marker': {'color': 'green'},
                    'text': total_spend_by_flavor['Flavor Spend']
                },
                'layout': {
                    'title': 'Total Flavor Spend by Flavor',
                    'xaxis': {'title': 'Flavor'},
                    'yaxis': {'title': 'Total Flavor Spend'},
                    'template': 'plotly_white'
                }
            }
            st.plotly_chart(plotting_data)

    else:
        st.warning("Please upload a CSV file.")
