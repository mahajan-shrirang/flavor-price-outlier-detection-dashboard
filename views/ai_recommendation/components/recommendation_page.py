import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    similarity_df_records = st.session_state.get('similarity_df_records', None)
    df = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing AI Recommendations.")
        return
    if similarity_df_records is None:
        st.warning("Please train the AI model on the Preference Selection page to start analyzing Recommendations.")
        return

    st.subheader("Top Opportunities for CIU Improvement")

    col1, col2 = st.columns(2)
    with col1:
        similarity = st.number_input(
            "Minimum Similarity Threshold",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="Set the minimum similarity threshold to filter the records."
        )
    with col2:
        ciu_delta = st.number_input(
            "Minimum CIU Delta Threshold (%)",
            min_value=0.0,
            max_value=1000000.0,
            value=0.0,
            step=100.0,
            help="Set the minimum CIU delta to filter the records."
        )

    with st.expander("Filters", expanded=False):
        filter_columns = st.multiselect(
            "Select Columns to Filter",
            options=st.session_state.get('ai_recommendation_features', []),
            help="Select columns to filter the records based on specific criteria."
        )
        matching_columns = st.session_state.get('ai_recommendation_matching_features', [])
        filter_values = {}
        for col in filter_columns:
            if col not in matching_columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Record 1 Filters")
                    filter_values[f"{col}_record1"] = st.selectbox(f"Select value for {col} in Record 1", options=df[col].unique(), key=f"filter_{col}_record1")
                with col2:
                    st.write(f"Record 2 Filters")
                    filter_values[f"{col}_record2"] = st.selectbox(f"Select value for {col} in Record 2", options=df[col].unique(), key=f"filter_{col}_record2")
            else:
                st.write(f"Matching Feature: {col}. Selecting a value for this column will filter both Record 1 and Record 2.")
                filter_values[f"{col}_record1"] = st.selectbox(
                    f"Select value for {col} in Record 1 and Record 2",
                    options=df[col].unique(),
                    key=f"filter_{col}_record1"
                )
                filter_values[f"{col}_record2"] = filter_values[f"{col}_record1"]

    
    if similarity_df_records is not None:
        top_opps: pd.DataFrame = similarity_df_records.copy()
        top_opps['similarity'] = top_opps['similarity'] * 100
        top_opps = top_opps[top_opps['similarity'] >= similarity]
        top_opps = top_opps[top_opps['perc_ciu_delta'] >= ciu_delta]
        if filter_columns:
            for col in filter_columns:
                top_opps = top_opps[(top_opps[f"{col}_record1"] == filter_values[f"{col}_record1"]) & (top_opps[f"{col}_record2"] == filter_values[f"{col}_record2"])]
        if top_opps.empty:
            st.warning("No records found matching the selected filters and thresholds.")
            return
        number_of_rows = st.number_input(
            "Number of rows to display",
            min_value=1,
            max_value=top_opps.shape[0],
            value=10 if top_opps.shape[0] > 10 else top_opps.shape[0],
            step=1,
            help="Select the number of top opportunities to display based on similarity and CIU delta."
        )
        top_opps = top_opps.sort_values('perc_ciu_delta', ascending=False).head(number_of_rows)
        top_opps['perc_ciu_delta'] = top_opps['perc_ciu_delta'].apply(lambda x: f"{x:.2f}%")

        display_df = top_opps.copy()
        display_df.rename(columns={
            'record1': 'Record 1',
            'record2': 'Record 2',
            'similarity': 'Similarity (%)',
            'total_ciu_delta': 'Total CIU Delta',
            'perc_ciu_delta': 'Percentage CIU Delta (%)'
        }, inplace=True)

        selected_entry = st.dataframe(
            display_df[['Record 1', 'Record 2', 'Similarity (%)', 'Total CIU Delta', 'Percentage CIU Delta (%)']],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            hide_index=True,
        )

        if len(selected_entry['selection']['rows']) != 0:
            selected_row = pd.DataFrame(top_opps.iloc[selected_entry['selection']['rows'][0]].to_dict(), index=[0])
            selected_row['similarity'] = selected_row['similarity'].astype(float)

            record1_data = selected_row[[col for col in selected_row.columns.to_list() if col.endswith('_record1')]]
            record1_data.rename(columns={col: col.replace('_record1', '') for col in record1_data.columns}, inplace=True)
            record1_data.rename(columns={'index': 'Index'}, inplace=True)
            record1_data = record1_data.reset_index(drop=True).T
            record1_data.columns = ['Record 1 Value']
            record2_data = selected_row[[col for col in selected_row.columns.to_list() if col.endswith('_record2')]]
            record2_data.rename(columns={col: col.replace('_record2', '') for col in record2_data.columns}, inplace=True)
            record2_data.rename(columns={'index': 'Index'}, inplace=True)
            record2_data = record2_data.reset_index(drop=True).T
            record2_data.columns = ['Record 2 Value']

            selected_row = selected_row.to_dict(orient='records')[0]

            st.subheader("Selected Record Pair:")
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Similarity", selected_row['similarity'])
            with c2:
                difference = float(selected_row['total_ciu_delta'])
                st.metric(
                    "Total CIU Difference",
                    value=f"{difference:.2f}",
                    delta=f"{selected_row['perc_ciu_delta']}",
                )

            st.subheader("Data Comparison")
            comparison_df = pd.merge(
                record1_data,
                record2_data,
                left_index=True,
                right_index=True,
                suffixes=('_record1', '_record2')
            )
            comparison_df.reset_index(inplace=True)
            comparison_df.columns = ["Column", "Record 1 Value", "Record 2 Value"]
            comparison_df['Match'] = comparison_df['Record 1 Value'] == comparison_df['Record 2 Value']
            st.dataframe(comparison_df, use_container_width=True)
