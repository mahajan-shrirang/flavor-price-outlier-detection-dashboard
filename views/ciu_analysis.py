import streamlit as st
import pandas as pd

def get_outliers(df):
    lower_quantile = df["Total CIU curr / vol"].quantile(0.25)
    upper_quantile = df["Total CIU curr / vol"].quantile(0.75)

    iqr = upper_quantile - lower_quantile

    lower_bound = lower_quantile - 1.5 * iqr
    upper_bound = upper_quantile + 1.5 * iqr

    df["Outlier"] = df["Total CIU curr / vol"].apply(
        lambda x: 1 if (x < lower_bound or x > upper_bound) else 0
    )
    return df, lower_bound, upper_bound

def main():
    st.title("📊 CIU Analysis")
    
    # Instructions for users
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown("""
        **What it does:** Finds unusual pricing (outliers) in your data. Helps identify where you might be paying too much or too little compared to normal prices.
        
        **How to use it:**
        1. **Select grouping options** - Choose how to group your data (e.g., by Flavor and Region)
        2. **Pick a specific group** from the table by clicking on a row
        3. **Review the results**:
           - See how many outliers were found
           - Look at the chart showing price distribution
           - Check the outlier details table
        
        **What you'll learn:**
        - Which products have unusual pricing
        - How many items are priced outside normal ranges
        - Specific records that need investigation
        - Price ranges for different product groups
        
        **When to use this:**
        - Monthly price reviews
        - When you suspect pricing errors
        - To validate supplier quotes
        - Before contract negotiations
        """)
    
    df = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing CIU.")
        return

    st.subheader("Select Group By")
    group_by = st.multiselect(
        "Group By",
        options=['Segment', 'Brand', 'Application', 'Product', 'Flavor', 'Supplier', 'Region', 'Process', 'Format', 'Country', 'RM code'],
        default=['Flavor', 'Region']
    )

    if len(group_by) != 0:
        grouped_df = df.groupby(group_by)

        grouped_list = []
        for name, group in grouped_df:
            temp_df, lower_bound, upper_bound = get_outliers(group)
            grouped_list.append(temp_df)

        grouped_df = pd.concat(grouped_list).groupby(group_by).agg({'Total CIU curr / vol': ['count', 'min', 'mean', 'median', 'max', 'std'], 'Flavor Spend': 'sum', 'Outlier': 'sum'}).reset_index()

        st.header("Aggregated CIU Stats")
        st.write(f"Aggregated CIU stats by {group_by}:")

        display_df = grouped_df.copy()
        display_df = display_df.round(2)
        # Convert to Single Index for better readability
        display_df_columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in display_df.columns.values]

        display_df.columns = display_df_columns
        display_df = display_df.rename(columns={
            'Total CIU curr / vol count': 'Record Count', 
            'Total CIU curr / vol min': 'Min CIU [$]',
            'Total CIU curr / vol max': 'Max CIU [$]',
            'Total CIU curr / vol mean': 'Mean CIU [$]',
            'Total CIU curr / vol median': 'Median CIU [$]',
            'Total CIU curr / vol std': 'CIU Std Dev [$]',
            'Flavor Spend sum': 'Flavor Spend Sum [$]',
            'Outlier sum': 'Outliers Count'
        })

        selection = st.dataframe(display_df, on_select="rerun", selection_mode="single-row", use_container_width=True)
        if len(selection['selection']['rows']) != 0:
            selected_row = grouped_df.iloc[selection['selection']['rows'][0]]
            selected_group = {col: selected_row[col][0] for col in group_by}
            st.subheader("Selected Group:")
            for col in group_by:
                st.write(f"{col}: {selected_group[col]}")
            filtered_df = df.copy()
            for col in group_by:
                filtered_df = filtered_df[filtered_df[col] == selected_group[col]]

        # filtered_df = df.copy()
        # columns = st.columns(len(group_by))
        # for i, col in enumerate(group_by):
        #     with columns[i]:
        #         selected_group = st.selectbox(
        #             "Select Group",
        #             options=filtered_df[col].unique().tolist(),
        #             index=0,
        #             key=f'outlier_group_select_{col}_1'
        #         )

        if len(selection['selection']['rows']) == 0:
            st.warning("Please select a group to analyze CIU outliers.")
            return
    else:
        st.write(df)
        filtered_df = df.copy()
        st.warning("No group selected, showing overall CIU distribution.")

    st.header("CIU Outlier Detection and Distribution")
    st.subheader("CIU Distribution")
    plotting_data = {
        'data': {
            "x": filtered_df['Total CIU curr / vol'],
            'type': 'histogram',
            'name': 'Total CIU curr / vol Distribution',
            'marker': {'color': 'lightgreen', 'opacity': 0.7},
        },
        'layout': {
            'title': 'Total CIU curr / vol Distribution',
            'xaxis': {'title': 'Total CIU curr / vol [$]'},
            'yaxis': {'title': 'Count'},
            'template': 'plotly_white'
        }
    }
    st.plotly_chart(plotting_data)

    lower_quantile = filtered_df["Total CIU curr / vol"].quantile(0.25)
    upper_quantile = filtered_df["Total CIU curr / vol"].quantile(0.75)

    iqr = upper_quantile - lower_quantile

    lower_bound = lower_quantile - 1.5 * iqr
    upper_bound = upper_quantile + 1.5 * iqr

    if lower_bound < 0:
        lower_bound = 0

    filtered_df["Outlier"] = filtered_df["Total CIU curr / vol"].apply(
        lambda x: 1 if (x < lower_bound or x > upper_bound) else 0
    )

    ciu_outliers = filtered_df[filtered_df["Outlier"] == 1]

    ciu_non_outliers = filtered_df[~filtered_df.index.isin(ciu_outliers.index)]

    st.subheader("Detected Outliers")
    if ciu_outliers.empty:
        st.write("No outliers detected in the selected group.")
    else:
        st.write(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        st.write(f"Number of outliers detected: {ciu_outliers.shape[0]}")

        st.subheader("Outliers:")
        st.write(ciu_outliers)

        st.subheader("Complete Data:")
        st.write(filtered_df)
