import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    st.title("Clustering Analysis")

    df: pd.DataFrame = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing AI Recommendations.")
        return
    
    matching_features = st.multiselect(
        "Select Grouping Features",
        options=df.columns.tolist(),
        default=st.session_state.get('clustering_matching_features', []),
        help="These features will be used to group the data before clustering analysis.",
    )

    if not matching_features or len(matching_features) == 0:
        st.warning("Please select at least one grouping feature to proceed with Clustering Analysis.")
        return

    feature_set = st.multiselect(
        "Select Features for Clustering Analysis",
        options=df.columns.tolist(),
        default=st.session_state.get('clustering_features', []),
        help="These features will be used for clustering analysis. Ensure to select relevant features that can help in identifying patterns in the data."
    )

    if not feature_set or len(feature_set) == 0:
        st.warning("Please select at least one feature to proceed with Clustering Analysis.")
        return

    if st.session_state.get('clustering_analysis_data', None) is not None:
        st.info("Clustering ML model has been trained.")
    
    if st.button("Train Clustering Model"):
        st.session_state['clustering_features'] = feature_set
        st.session_state['clustering_matching_features'] = matching_features

        with st.spinner("Training Clustering Model... This may take a few minutes depending on the dataset size."):
            feature_set = ['Segment', 'Brand', 'Application', 'Product', 'Flavor', 'Supplier', 'Region', 'Process', 'Format', 'Country', 'RM code', 'Multiple Flavors']
            X = df[feature_set]
            X = pd.get_dummies(X, drop_first=True)
            X = X.replace({True: 1, False: 0})

            groups = df.groupby(matching_features)

            group_list = []
            progress_bar = st.progress(0)
            total_groups = len(groups)

            for idx, (name, group) in enumerate(groups):
                progress_bar.progress((idx + 1) / total_groups, text=f"Processing group {idx + 1} of {total_groups}: {name} | {(idx + 1) / total_groups:.2%} complete")
                if len(group) > 3:
                    group['Group'] = idx
                    training_data = X.loc[group.index]
                    k_clusters = range(2, min(11, len(training_data) // 2 + 1))
                    inertias = []
                    for k in k_clusters:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(training_data)
                        inertias.append(kmeans.inertia_)
                    optimal_k = k_clusters[inertias.index(min(inertias))]
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                    group['Cluster'] = kmeans.fit_predict(training_data)
                    group['Group'] = group['Group'].apply(lambda x: f"Group {x}")
                    group['Cluster'] = group['Cluster'].apply(lambda x: f"Cluster {x}")
                    group_list.append(group)
                else:
                    group['Group'] = f"Group {idx}"
                    group['Cluster'] = 'Cluster 0'
                    group_list.append(group)

            progress_bar.progress(1.0, text="Processing complete. Compiling results...")
            group_list = pd.concat(group_list, ignore_index=True)

            st.session_state['clustering_analysis_data'] = group_list
            st.success("Clustering Model trained successfully!")

    if st.session_state.get('clustering_analysis_data', None) is not None:
        st.write("Clustering Analysis Results:")
        group_list: pd.DataFrame = st.session_state['clustering_analysis_data']

        group_list_grouped = group_list.groupby(['Group', 'Cluster']).agg({
            **{matching_feature: 'first' for matching_feature in matching_features},
            'CIU curr / vol': ['count', 'mean', 'std', 'min', 'max'],
            # 'FG volume / year': ['mean', 'std', 'min', 'max'],
            # 'Total CIU curr / vol': ['mean', 'std', 'min', 'max'],
            # 'Flavor Spend': ['mean', 'std', 'min', 'max'],
        })

        group_list_grouped = group_list_grouped[group_list_grouped[('CIU curr / vol', 'count')] > 3]

        st.dataframe(group_list_grouped)

        st.divider()

        st.subheader("Clusters with Highest Deviation in CIU curr / vol")
        top_clusters = group_list_grouped.sort_values(('CIU curr / vol', 'std'), ascending=False).head(10).reset_index()

        top_clusters.columns = ['Group', 'Cluster'] + [f'{col[0]} {str(col[1]).capitalize()}' for col in top_clusters.columns[2:]]

        selected_row = st.dataframe(top_clusters, selection_mode="single-row", on_select="rerun")

        if selected_row['selection'].get('rows') != []:
            selected_row = top_clusters.iloc[selected_row['selection']['rows']].to_dict(orient='records')[0]
            selected_group = selected_row['Group']
            selected_cluster = selected_row['Cluster']

            cluster_data = group_list[(group_list['Group'] == selected_group) & (group_list['Cluster'] == selected_cluster)]
            st.dataframe(cluster_data)
