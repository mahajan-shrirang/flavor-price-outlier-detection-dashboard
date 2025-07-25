import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

def calculate_hhi_index(df, column):
    """
    Calculate Herfindahl-Hirschman Index (HHI) for supplier concentration
    HHI is the sum of squared market shares (0-10000 scale)
    HHI < 1500: Low concentration
    1500 < HHI < 2500: Moderate concentration
    HHI > 2500: High concentration
    """
    total_spend = df['Flavor Spend'].sum()
    supplier_spend = df.groupby(column)['Flavor Spend'].sum().reset_index()
    supplier_spend['Market Share'] = supplier_spend['Flavor Spend'] / total_spend
    supplier_spend['HHI Contribution'] = supplier_spend['Market Share'] ** 2 * 10000
    
    hhi = supplier_spend['HHI Contribution'].sum()
    
    # Categorize HHI
    if hhi < 1500:
        concentration_level = "Low"
        risk_score = 1
    elif 1500 <= hhi <= 2500:
        concentration_level = "Moderate"
        risk_score = 2
    else:
        concentration_level = "High"
        risk_score = 3
        
    return hhi, concentration_level, risk_score, supplier_spend

def calculate_price_volatility(df, group_columns):
    """
    Calculate price volatility score based on CIU variation
    Uses coefficient of variation (CV) = standard deviation / mean
    """
    grouped = df.groupby(group_columns)['Total CIU curr / vol'].agg(['mean', 'std']).reset_index()
    grouped['CV'] = grouped['std'] / grouped['mean']
    grouped = grouped.dropna()
    grouped['Volatility Score'] = grouped['CV'].apply(lambda x: 3 if x > 0.3 else (2 if x > 0.15 else 1))
    
    return grouped

def calculate_supply_chain_risk(df):
    """
    Calculate supply chain disruption risk based on:
    1. Geographic concentration
    2. Supplier count
    3. Multiple flavors indicator
    """
    # Regional concentration
    region_count = df['Region'].nunique()
    country_count = df['Country'].nunique()
    supplier_count = df['Supplier'].nunique()
    
    # Risk factors
    geo_risk = 3 if region_count <= 2 else (2 if region_count <= 4 else 1)
    supplier_risk = 3 if supplier_count <= 3 else (2 if supplier_count <= 7 else 1)
    
    # Calculate multi-sourcing percentage
    multi_flavors = df['Multiple Flavors'].mean() * 100
    multi_flavor_risk = 3 if multi_flavors < 20 else (2 if multi_flavors < 50 else 1)
    
    # Overall risk score (scale 3-9, where 9 is highest risk)
    overall_risk = geo_risk + supplier_risk + multi_flavor_risk
    risk_level = "High" if overall_risk >= 7 else ("Medium" if overall_risk >= 5 else "Low")
    
    risk_data = {
        "Region Count": region_count,
        "Country Count": country_count,
        "Supplier Count": supplier_count,
        "Multi-flavor %": f"{multi_flavors:.1f}%",
        "Geographic Risk": geo_risk,
        "Supplier Risk": supplier_risk,
        "Multi-flavor Risk": multi_flavor_risk,
        "Overall Risk Score": overall_risk,
        "Risk Level": risk_level
    }
    
    return risk_data

def simulate_disruption_impact(df, disruption_type, impact_percentage):
    """
    Simulate the impact of a supply chain disruption
    """
    if disruption_type == "Supplier Disruption":
        # Sort suppliers by spend and select top ones by impact percentage
        supplier_spend = df.groupby('Supplier')['Flavor Spend'].sum().reset_index()
        supplier_spend = supplier_spend.sort_values('Flavor Spend', ascending=False)
        cumulative_percent = (supplier_spend['Flavor Spend'].cumsum() / supplier_spend['Flavor Spend'].sum()) * 100
        affected_suppliers = supplier_spend[cumulative_percent <= impact_percentage]['Supplier'].tolist()
        
        # Calculate affected spend and volume
        affected_df = df[df['Supplier'].isin(affected_suppliers)]
        
    elif disruption_type == "Regional Disruption":
        # Sort regions by spend and select top ones by impact percentage
        region_spend = df.groupby('Region')['Flavor Spend'].sum().reset_index()
        region_spend = region_spend.sort_values('Flavor Spend', ascending=False)
        cumulative_percent = (region_spend['Flavor Spend'].cumsum() / region_spend['Flavor Spend'].sum()) * 100
        affected_regions = region_spend[cumulative_percent <= impact_percentage]['Region'].tolist()
        
        # Calculate affected spend and volume
        affected_df = df[df['Region'].isin(affected_regions)]
    
    # Calculate impact metrics
    total_spend = df['Flavor Spend'].sum()
    total_volume = df['FG volume / year'].sum()
    
    affected_spend = affected_df['Flavor Spend'].sum()
    affected_volume = affected_df['FG volume / year'].sum()
    
    impact_metrics = {
        "Affected Spend": affected_spend,
        "Affected Spend %": (affected_spend / total_spend) * 100,
        "Affected Volume": affected_volume,
        "Affected Volume %": (affected_volume / total_volume) * 100,
        "Number of Products Affected": affected_df['Product'].nunique(),
        "Number of Flavors Affected": affected_df['Flavor'].nunique()
    }
    
    return impact_metrics, affected_df

def main():
    st.title("Risk Analysis")
    
    df = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing Risks.")
        return
    
    tabs = st.tabs(["Supplier Concentration Risk", "Price Volatility Risk"])

    with tabs[0]:
        st.header("Supplier Concentration Risk Assessment")
        st.markdown("""
        This analysis uses the Herfindahl-Hirschman Index (HHI) to measure supplier concentration:
        - **HHI < 1500**: Low concentration risk
        - **1500 < HHI < 2500**: Moderate concentration risk
        - **HHI > 2500**: High concentration risk
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            concentration_dimension = st.selectbox(
                "Select dimension to analyze concentration",
                options=["Supplier", "Region", "Country", "Flavor"],
                index=0
            )
            
        with col2:
            grouping_dimension = st.multiselect(
                "Optional: Group by additional dimensions",
                options=["Product", "Flavor", "Brand", "Segment"],
                default=[]
            )
        
        if not grouping_dimension:
            # Calculate overall concentration
            hhi, concentration_level, risk_score, supplier_data = calculate_hhi_index(
                df, concentration_dimension
            )
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("HHI Index", f"{hhi:.1f}")
            with metrics_col2:
                st.metric("Concentration Level", concentration_level)
            
            # Display market share visualization
            top_suppliers = supplier_data.sort_values('Flavor Spend', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_suppliers[concentration_dimension],
                    y=top_suppliers['Market Share'] * 100,
                    marker=dict(
                        color='green',
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                    ),
                    text=[f"{x:.1f}%" for x in top_suppliers['Market Share'] * 100],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Top 10 {concentration_dimension}s by Market Share",
                xaxis_title=concentration_dimension,
                yaxis_title="Market Share (%)",
                template='plotly_white'
            )
            
            st.plotly_chart(fig)
            
            # Pareto Chart
            supplier_data = supplier_data.sort_values('Flavor Spend', ascending=False)
            supplier_data['Cumulative %'] = supplier_data['Market Share'].cumsum() * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=supplier_data[concentration_dimension],
                y=supplier_data['Market Share'] * 100,
                name='Market Share',
                marker=dict(color='lightgreen')
            ))
            
            fig.add_trace(go.Scatter(
                x=supplier_data[concentration_dimension],
                y=supplier_data['Cumulative %'],
                name='Cumulative %',
                marker=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"Pareto Chart: {concentration_dimension} Concentration",
                xaxis_title=concentration_dimension,
                yaxis_title='Market Share (%)',
                yaxis2=dict(
                    title='Cumulative %',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                template='plotly_white'
            )
            
            st.plotly_chart(fig)
            
            # Data table with full details
            st.subheader(f"{concentration_dimension} Concentration Details")
            detail_df = supplier_data.copy()
            detail_df['Market Share'] = (detail_df['Market Share'] * 100).round(2).astype(str) + '%'
            detail_df['HHI Contribution'] = detail_df['HHI Contribution'].round(2)
            st.dataframe(detail_df.reset_index(drop=True))
            
        else:
            # Group by selected dimensions and show concentration by group
            # group_cols = grouping_dimension + [concentration_dimension]
            
            # grouped_df = df.groupby(group_cols)['Flavor Spend'].sum().reset_index()
            # group_totals = df.groupby(grouping_dimension)['Flavor Spend'].sum().reset_index()
            
            concentration_results = []
            
            for group_name, group_data in df.groupby(grouping_dimension):
                if isinstance(group_name, tuple):
                    group_id = "/".join(str(g) for g in group_name)
                else:
                    group_id = str(group_name)
                    
                hhi, level, score, _ = calculate_hhi_index(group_data, concentration_dimension)
                
                group_total = group_data['Flavor Spend'].sum()
                overall_total = df['Flavor Spend'].sum()
                
                concentration_results.append({
                    'Group': group_id,
                    'HHI': hhi,
                    'Concentration Level': level,
                    'Risk Score': score,
                    'Group Spend': group_total,
                    'Percent of Total': (group_total / overall_total) * 100
                })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(concentration_results)
            
            # Sort by HHI descending
            results_df = results_df.sort_values('HHI', ascending=False)
            
            # Format columns
            results_df['HHI'] = results_df['HHI'].round(2)
            results_df['Group Spend'] = results_df['Group Spend'].round(2)
            results_df['Percent of Total'] = results_df['Percent of Total'].round(2).astype(str) + '%'
            
            st.subheader(f"Concentration by {', '.join(grouping_dimension)}")
            st.dataframe(results_df)
            
            # Visualization of HHI by group
            fig = px.bar(
                results_df,
                x='Group',
                y='HHI',
                color='Concentration Level',
                color_discrete_map={
                    'Low': 'green',
                    'Moderate': 'orange',
                    'High': 'red'
                },
                height=500
            )
            
            fig.add_hline(y=1500, line_dash="dash", line_color="orange", annotation_text="Moderate Concentration")
            fig.add_hline(y=2500, line_dash="dash", line_color="red", annotation_text="High Concentration")
            
            fig.update_layout(
                title=f"HHI Index by {', '.join(grouping_dimension)}",
                xaxis_title=f"{', '.join(grouping_dimension)}",
                yaxis_title="HHI Index",
                template='plotly_white'
            )
            
            st.plotly_chart(fig)

    with tabs[1]:
        st.header("Price Volatility Risk Analysis")
        st.markdown("""
        This analysis examines price volatility based on the Coefficient of Variation (CV) in CIU prices.
        - **CV < 0.15**: Low volatility
        - **0.15 < CV < 0.30**: Moderate volatility
        - **CV > 0.30**: High volatility
        """)

        # Write a warning that when there is only single row is present, CV cannot be calculated. Thus these records will be excluded from the analysis.
        st.warning("""
        Note: When there is only a single row present for a group, CV cannot be calculated. These records will be excluded from the analysis.
        """)
        
        
        volatility_dims = st.multiselect(
            "Select dimensions to analyze price volatility",
            options=["Product", "Flavor", "Supplier", "Brand", "Country", "Region"],
            default=["Product", "Supplier"]
        )
        
        if volatility_dims:
            volatility_data = calculate_price_volatility(df, volatility_dims)
            
            # Format data for display
            display_data = volatility_data.copy()
            display_data['CV'] = (display_data['CV'] * 100).round(2).astype(str) + '%'
            display_data['mean'] = display_data['mean'].round(2)
            display_data['std'] = display_data['std'].round(2)
            
            # Risk breakdown
            risk_counts = volatility_data['Volatility Score'].value_counts().reset_index()
            risk_counts.columns = ['Risk Score', 'Count']
            risk_counts['Percentage'] = (risk_counts['Count'] / risk_counts['Count'].sum() * 100).round(2)
            
            # Display metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                high_vol = len(volatility_data[volatility_data['Volatility Score'] == 3])
                st.metric("High Volatility Items", high_vol)
            with metrics_col2:
                med_vol = len(volatility_data[volatility_data['Volatility Score'] == 2])
                st.metric("Medium Volatility Items", med_vol)
            with metrics_col3:
                low_vol = len(volatility_data[volatility_data['Volatility Score'] == 1])
                st.metric("Low Volatility Items", low_vol)
            
            # Visualization - Distribution of volatility
            fig = px.histogram(
                volatility_data,
                x='CV',
                nbins=20,
                color_discrete_sequence=['lightgreen']
            )
            
            fig.add_vline(x=0.15, line_dash="dash", line_color="orange", annotation_text="Moderate Volatility")
            fig.add_vline(x=0.30, line_dash="dash", line_color="red", annotation_text="High Volatility")
            
            fig.update_layout(
                title="Distribution of Price Volatility",
                xaxis_title="Coefficient of Variation (CV)",
                yaxis_title="Count",
                template='plotly_white'
            )
            
            st.plotly_chart(fig)

            # Most Volatile Items
            st.subheader("Most Volatile Items")

            count = st.slider(
                "Number of Top Volatile Items to Display",
                min_value=5,
                max_value=len(volatility_data),
                value=10,
                step=5,
                help="Select how many top volatile items to display"
            )

            top_volatile = volatility_data.sort_values('CV', ascending=False).head(count)
            
            formatted_top = top_volatile.copy()
            formatted_top['CV'] = (formatted_top['CV'] * 100).round(2).astype(str) + '%'
            formatted_top['mean'] = formatted_top['mean'].round(2)
            formatted_top['std'] = formatted_top['std'].round(2)

            formatted_top = formatted_top.reset_index(drop=True)
            
            st.dataframe(formatted_top)
            
            # Full volatility data table
            # with st.expander("View Complete Volatility Analysis"):
            #     st.dataframe(display_data.sort_values('CV', ascending=False))

    # with tabs[2]:
    #     st.header("Supply Chain Disruption Impact Analysis")
        
    #     # Overall supply chain risk assessment
    #     risk_data = calculate_supply_chain_risk(df)
        
    #     st.subheader("Supply Chain Risk Assessment")
        
    #     metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    #     with metrics_col1:
    #         st.metric("Overall Risk Level", risk_data["Risk Level"])
    #     with metrics_col2:
    #         st.metric("Risk Score", f"{risk_data['Overall Risk Score']}/9")
    #     with metrics_col3:
    #         st.metric("Geographic Risk", f"{risk_data['Geographic Risk']}/3")
        
    #     # Risk factors table
    #     risk_factors_df = pd.DataFrame({
    #         "Risk Factor": [
    #             "Geographic Diversity", 
    #             "Supplier Diversity",
    #             "Multi-flavor Sourcing"
    #         ],
    #         "Metric": [
    #             f"{risk_data['Region Count']} regions, {risk_data['Country Count']} countries",
    #             f"{risk_data['Supplier Count']} suppliers",
    #             risk_data["Multi-flavor %"]
    #         ],
    #         "Risk Score": [
    #             f"{risk_data['Geographic Risk']}/3",
    #             f"{risk_data['Supplier Risk']}/3",
    #             f"{risk_data['Multi-flavor Risk']}/3"
    #         ]
    #     })
        
    #     st.dataframe(risk_factors_df, hide_index=True)
        
    #     # # Disruption simulation
    #     # st.subheader("Disruption Scenario Simulation")
    #     # st.markdown("""
    #     # Simulate the impact of supply chain disruptions by selecting a disruption type and impact level.
    #     # """)
        
    #     # sim_col1, sim_col2 = st.columns(2)
    #     # with sim_col1:
    #     #     disruption_type = st.selectbox(
    #     #         "Disruption Type",
    #     #         options=["Supplier Disruption", "Regional Disruption"],
    #     #         index=0
    #     #     )
    #     # with sim_col2:
    #     #     impact_percentage = st.slider(
    #     #         "Impact Percentage", 
    #     #         min_value=10,
    #     #         max_value=90,
    #     #         value=50,
    #     #         step=10,
    #     #         help="Percentage of supply affected by the disruption"
    #     #     )
        
    #     # # Run simulation
    #     # impact_metrics, affected_df = simulate_disruption_impact(df, disruption_type, impact_percentage)
        
    #     # # Display impact metrics
    #     # metrics_col1, metrics_col2 = st.columns(2)
    #     # with metrics_col1:
    #     #     st.metric("Affected Spend", f"${impact_metrics['Affected Spend']:,.2f}")
    #     #     st.metric("Affected Volume", f"{impact_metrics['Affected Volume']:,.2f}")
    #     # with metrics_col2:
    #     #     st.metric("Affected Spend %", f"{impact_metrics['Affected Spend %']:.2f}%")
    #     #     st.metric("Affected Volume %", f"{impact_metrics['Affected Volume %']:.2f}%")
        
    #     # impact_col1, impact_col2 = st.columns(2)
    #     # with impact_col1:
    #     #     st.metric("Products Affected", impact_metrics['Number of Products Affected'])
    #     # with impact_col2:
    #     #     st.metric("Flavors Affected", impact_metrics['Number of Flavors Affected'])
        
    #     # # Visualize affected products/flavors
    #     # if disruption_type == "Supplier Disruption":
    #     #     dimension = "Supplier"
    #     # else:
    #     #     dimension = "Region"
        
    #     # affected_summary = affected_df.groupby("Product")["Flavor Spend"].sum().reset_index()
    #     # affected_summary = affected_summary.sort_values("Flavor Spend", ascending=False)
        
    #     # fig = px.bar(
    #     #     affected_summary.head(10),
    #     #     x="Product",
    #     #     y="Flavor Spend",
    #     #     title=f"Top 10 Affected Products by {disruption_type}",
    #     #     color_discrete_sequence=['red']
    #     # )
        
    #     # fig.update_layout(
    #     #     xaxis_title="Product",
    #     #     yaxis_title="Affected Spend ($)",
    #     #     template='plotly_white'
    #     # )
        
    #     # st.plotly_chart(fig)
        
    #     # # Display affected items
    #     # with st.expander("View Affected Items Details"):
    #     #     st.dataframe(affected_df)
