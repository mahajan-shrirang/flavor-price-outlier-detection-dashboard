import numpy as np
import streamlit as st
import pandas as pd


def main():
    st.title("Spend Impact Analysis")
    st.write("This page is under construction. Please check back later for updates.")

    df = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing Spend Impact.")
        return

    options_columns = ['Product', 'Flavor', 'Supplier', 'Brand', 'Country', 'Format', 'Application', 'Segment', 'RM code', 'Region', 'Process']

    c1, c2 = st.columns(2)
    with c1:
        x_axis_col = st.selectbox(
            "Select X-axis Column",
            options=options_columns,
            index=0,
            key='x_axis_col'
        )
    with c2:
        y_axis_col = st.selectbox(
            "Select Y-axis Column",
            options=['Flavor Spend', 'CIU', 'Total CIU'],
            index=0,
            key='y_axis_col'
        )
        y_axis_options = {
            'Flavor Spend': 'Flavor Spend',
            'CIU': 'CIU curr / vol',
            'Total CIU': 'Total CIU curr / vol'
        }
        y_axis_col = y_axis_options[y_axis_col]
    
    st.write(f"Filtering data")

    filter_columns = np.array(options_columns)
    filter_columns = filter_columns[filter_columns != x_axis_col]

    selected_filters = st.multiselect(
        "Select Filters",
        options=filter_columns.tolist(),
        default=[],
        key='spend_impact_filters'
    )

    if len(selected_filters) > 0:
        filter_conditions = []
        cols = st.columns(len(selected_filters))
        for i, col in enumerate(selected_filters):
            with cols[i]:
                unique_values = df[col].unique()
                selected_values = st.selectbox(
                    f"Select value for {col}",
                options=unique_values,
                index=0,
                key=f'spend_impact_filter_{col}'
            )
            if selected_values:
                filter_conditions.append(df[col] == selected_values)

        if filter_conditions:
            df = df[np.logical_and.reduce(filter_conditions)]
            st.write(f"Filtered data based on selected filters: {', '.join(selected_filters)}")
        else:
            st.warning("No filters selected, displaying all data.")

    st.write(f"Total records after filtering: {len(df)}")
    value = (len(df) // 2) + 1
    if len(df) < 10:
        value = len(df)
    else:
        value = 10
    count_of_records = st.number_input("Number of Records to Display", min_value=1, max_value=len(df), value=value, step=1, key='count_of_records')

    if y_axis_col not in ['CIU curr / vol', 'Total CIU curr / vol']:
        top_products = df.groupby(x_axis_col)[y_axis_col].sum().nlargest(count_of_records).reset_index()
    else:
        top_products = df.groupby(x_axis_col)[y_axis_col].mean().nlargest(count_of_records).reset_index()
    top_products[y_axis_col] = top_products[y_axis_col].round(2)
    top_products = top_products.sort_values(by=y_axis_col, ascending=False)

    plotting_data = {
        'data': {
            "x": top_products[x_axis_col],
            "y": top_products[y_axis_col],
            'type': 'bar',
            'name': f'Top 10 {x_axis_col} by {y_axis_col}',
            'marker': {'color': 'lightgreen'},
            'text':  top_products[y_axis_col].apply(lambda x: f"${x:,.2f}"),
            'textposition': 'auto'
        },
        'layout': {
            'title': f'Top 10 {x_axis_col} by {y_axis_col}',
            'xaxis': {'title': x_axis_col},
            'yaxis': {'title': f"{y_axis_col} [$]"},
            'template': 'plotly_white'
        }
    }
    st.plotly_chart(plotting_data)

    st.divider()

    # Pareto chart: Product-wise cumulative % of total spend
    total_spend = df['Flavor Spend'].sum()
    product_spend = df.groupby('Product')['Flavor Spend'].sum().reset_index()
    product_spend = product_spend.sort_values(by='Flavor Spend', ascending=False)
    product_spend['Cumulative Spend'] = product_spend['Flavor Spend'].cumsum()
    product_spend['Cumulative %'] = (product_spend['Cumulative Spend'] / total_spend) * 100
    product_spend['Flavor Spend'] = product_spend['Flavor Spend'].round(2)

    plotting_data = {
        'data': {
            "x": product_spend['Product'],
            "y": product_spend['Cumulative %'],
            'type': 'bar',
            'name': 'Cumulative % of Total Spend',
            'marker': {'color': 'lightgreen'},
            'text': product_spend['Cumulative %'].apply(lambda x: f"{x:.2f}%"),
            'textposition': 'auto'
        },
        'layout': {
            'title': 'Pareto Chart: Product-wise Cumulative % of Total Spend',
            'xaxis': {'title': 'Product'},
            'yaxis': {'title': 'Cumulative %'},
            'template': 'plotly_white'
        }
    }
    st.plotly_chart(plotting_data)

    # Breakdown of spend by: Country, Supplier, Product, RM code
    st.subheader("Spend Breakdown by Country, Supplier, Product, and RM code")
    breakdown_columns = ['Country', 'Supplier', 'Product', 'RM code']
    breakdown_data = df.groupby(breakdown_columns)['Flavor Spend'].sum().reset_index()
    breakdown_data['Flavor Spend'] = breakdown_data['Flavor Spend'].round(2)

    st.write(breakdown_data)
