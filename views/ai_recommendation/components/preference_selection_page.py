import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def main():
    df: pd.DataFrame = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing AI Recommendations.")
        return

    feature_set = st.multiselect(
        "Select Features for AI Recommendation",
        options=df.columns.tolist(),
        default=st.session_state.get('ai_recommendation_features', []),
    )

    matching_features = st.multiselect(
        "Select Matching Features for AI Recommendation",
        options=feature_set,
        default=st.session_state.get('ai_recommendation_matching_features', []),
        help="AI Recommendation will use these features to find similar records.",
    )

    if not feature_set or len(feature_set) == 0:
        st.warning("Please select at least one feature to proceed with AI Recommendation analysis.")
        return
    
    if st.session_state.get('similarity_df_records', None) is not None:
        st.info("AI Recommendation model has been trained.")
        st.write("You can proceed to the Recommendations page to view the results.")

    if st.button("Train AI"):
        st.session_state['ai_recommendation_features'] = feature_set
        st.session_state['ai_recommendation_matching_features'] = matching_features
        
        with st.spinner("Training AI... This may take a few minutes depending on the dataset size."):
            groupings = df.groupby(matching_features)

            groups = []
            for name, group in groupings:
                if len(group) < 2:
                    continue
                X = group[feature_set]
                X = pd.get_dummies(X, drop_first=True)
                X = X.replace({True: 1, False: 0})

                similarity_matrix = cosine_similarity(X)
                similarity_df = pd.DataFrame(similarity_matrix)

                similarity_df.columns = group.index
                similarity_df.index = group.index

                groups.append(similarity_df)

            df = df.reset_index()
            group_list = []
            progress_bar = st.progress(0)
            for idx, group in enumerate(groups):
                group = pd.DataFrame(group)
                # I have matrix of n x n where n is number of records in group
                # I want to convert this matrix into a long format DataFrame with columns:
                # 'record1', 'record2', 'similarity'
                group = group.stack().reset_index()
                group.columns = ['record1', 'record2', 'similarity']
                group = group[group['record1'] != group['record2']]

                group = group.merge(df, left_on='record1', right_on='index', suffixes=('', '_record1'))
                group.rename(columns={col: f'{col}_record1' for col in df.columns}, inplace=True)
                group = group.merge(df, left_on='record2', right_on='index', suffixes=('', '_record2'))
                group.rename(columns={col: f'{col}_record2' for col in df.columns}, inplace=True)

                group['total_ciu_delta'] = group['Total CIU curr / vol_record1'] - group['Total CIU curr / vol_record2']
                group['perc_ciu_delta'] = (group['total_ciu_delta'] / group['Total CIU curr / vol_record2']) * 100

                group_list.append(group)
                # break
                
            progress_bar.progress(1.0)
            
            similarity_df_records = pd.concat(group_list)
            st.session_state['similarity_df_records'] = similarity_df_records
            st.success("AI Training completed successfully!")
