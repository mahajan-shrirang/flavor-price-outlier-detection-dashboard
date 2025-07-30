import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

def calculate_hhi(shares):
    """Calculate the Herfindahl-Hirschman Index (HHI) for concentration"""
    # Normalize to ensure shares sum to 1.0
    total = sum(shares.values())
    if total > 0:
        normalized_shares = {k: v/total for k, v in shares.items()}
    else:
        normalized_shares = shares
    
    # Calculate HHI (on 10000 scale)
    hhi = sum(share**2 for share in normalized_shares.values()) * 10000
    return hhi

def simple_optimize_portfolio(df, analysis_dimension, cost_weight=0.5, diversity_weight=0.5, spend_target=None, min_percent=0, max_percent=100):
    """
    A simplified portfolio optimization function
    
    Args:
        df (pd.DataFrame): The dataframe with supplier data
        analysis_dimension (str): The column to analyze (e.g., 'Supplier')
        cost_weight (float): Weight for cost minimization (0-1)
        diversity_weight (float): Weight for diversity maximization (0-1)
        spend_target (float): Target total spend (if None, use current total)
        min_percent (float): Minimum percentage allocation per supplier (0-100)
        max_percent (float): Maximum percentage allocation per supplier (0-100)
        
    Returns:
        dict: Optimized allocation of spend to suppliers
        dict: Performance metrics of the optimized portfolio
    """
    # Calculate current allocation
    current_df = df.groupby(analysis_dimension).agg({
        'Flavor Spend': 'sum',
        'Total CIU curr / vol': 'mean'
    }).reset_index()
    
    # Create supplier lookup with average CIU
    suppliers = current_df[analysis_dimension].tolist()
    supplier_ciu = dict(zip(current_df[analysis_dimension], current_df['Total CIU curr / vol']))
    
    # Calculate current spend and allocation
    current_spend = current_df['Flavor Spend'].sum()
    current_allocation = dict(zip(current_df[analysis_dimension], current_df['Flavor Spend']))
    
    # Set target spend (use current if not specified)
    if spend_target is None:
        spend_target = current_spend
    
    # Convert min/max percentages to values
    min_allocation = spend_target * (min_percent / 100)
    max_allocation = spend_target * (max_percent / 100)
    
    # Calculate current HHI
    current_hhi = calculate_hhi(current_allocation)
    
    # Define the objective function to minimize
    def objective(x):
        # Convert array to dict for easier handling
        allocation = {suppliers[i]: x[i] for i in range(len(suppliers))}
        
        # Cost component - weighted average CIU
        total = sum(x)
        if total > 0:
            cost_component = sum(supplier_ciu[suppliers[i]] * x[i] for i in range(len(suppliers))) / total
            # Normalize cost by max CIU
            max_ciu = max(supplier_ciu.values())
            cost_component = cost_component / max_ciu if max_ciu > 0 else 0
        else:
            cost_component = 0
        
        # Diversity component - HHI (normalized to 0-1)
        hhi = calculate_hhi(allocation)
        diversity_component = hhi / 10000
        
        # The objective function is set up so that we want to MINIMIZE both components
        # For cost: lower is better already
        # For diversity: higher HHI means worse diversity, so this works as is
        
        # Apply weights with exponential emphasis to make differences more pronounced
        # This ensures that even small changes in weights have a noticeable impact
        if cost_weight == 0:
            cost_impact = 0
        else:
            cost_impact = cost_component ** (1 / cost_weight) if cost_weight > 0.1 else 0
            
        if diversity_weight == 0:
            diversity_impact = 0
        else:
            diversity_impact = diversity_component ** (1 / diversity_weight) if diversity_weight > 0.1 else 0
        
        # Combined objective (using weighted sum)
        # Add a small value to avoid division by zero or numerical issues
        return (cost_weight * cost_impact + diversity_weight * diversity_impact) + 0.001
    
    # Define constraint that sum equals target spend
    def constraint_total_spend(x):
        return sum(x) - spend_target
    
    # Initial guess - weighted by inverse of CIU (cheaper suppliers get more initial allocation)
    # This helps the optimizer find better solutions by starting from a cost-sensitive point
    total_inverse_ciu = sum(1/ciu if ciu > 0 else 0 for ciu in supplier_ciu.values())
    if total_inverse_ciu > 0:
        x0 = np.array([(1/supplier_ciu[supplier] if supplier_ciu[supplier] > 0 else 1) 
                      for supplier in suppliers])
        x0 = x0 * (spend_target / sum(x0))
    else:
        # Fallback to equal distribution
        x0 = np.ones(len(suppliers)) * (spend_target / len(suppliers))
    
    # Bounds for each supplier
    bounds = [(min_allocation, max_allocation) for _ in suppliers]
    
    # Constraint: total allocation must equal spend target
    constraints = [{'type': 'eq', 'fun': constraint_total_spend}]
    
    # Run the optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'disp': False, 'maxiter': 1000000}
    )
    
    if result.success:
        st.success("Optimization successful!")
        # Convert results back to dictionary
        optimized_allocation = {suppliers[i]: result.x[i] for i in range(len(suppliers))}
        
        # Calculate performance metrics
        weighted_avg_ciu = sum(supplier_ciu[s] * amt for s, amt in optimized_allocation.items()) / spend_target
        optimized_hhi = calculate_hhi(optimized_allocation)
        
        # Calculate changes
        current_weighted_avg_ciu = sum(supplier_ciu[s] * amt for s, amt in current_allocation.items()) / current_spend
        ciu_change = (current_weighted_avg_ciu - weighted_avg_ciu) / current_weighted_avg_ciu * 100 if current_weighted_avg_ciu > 0 else 0
        hhi_change = (current_hhi - optimized_hhi) / current_hhi * 100 if current_hhi > 0 else 0
        
        # Add more detailed metrics
        cost_impact = cost_weight * (weighted_avg_ciu / max(supplier_ciu.values()) if supplier_ciu.values() else 1)
        diversity_impact = diversity_weight * (optimized_hhi / 10000)
        
        metrics = {
            'weighted_avg_ciu': weighted_avg_ciu,
            'hhi': optimized_hhi,
            'ciu_change': ciu_change,
            'hhi_change': hhi_change,
            'cost_impact': cost_impact,
            'diversity_impact': diversity_impact
        }
        
        return optimized_allocation, metrics
    else:
        st.error("Optimization failed. Please adjust your parameters and try again.")
        # If optimization failed, return current allocation
        return current_allocation, {}

def run_simple_scenario(df, analysis_dimension, scenario_type, change_value, current_allocation=None, 
                      cost_weight=0.5, diversity_weight=0.5):
    """
    Run a simplified scenario analysis
    
    Args:
        df (pd.DataFrame): The dataframe with supplier data
        analysis_dimension (str): The column to analyze (e.g., 'Supplier')
        scenario_type (str): Type of scenario ('price_change', 'budget_change')
        change_value (dict or float): The change to apply
        current_allocation (dict): Current allocation of spend to suppliers
        cost_weight (float): Weight for cost optimization
        diversity_weight (float): Weight for diversity optimization
        
    Returns:
        dict: Optimized allocation for this scenario
        dict: Performance metrics
    """
    scenario_df = df.copy()
    
    # If current allocation not provided, calculate it
    if current_allocation is None:
        allocation_df = df.groupby(analysis_dimension)['Flavor Spend'].sum().reset_index()
        current_allocation = dict(zip(allocation_df[analysis_dimension], allocation_df['Flavor Spend']))
    
    current_spend = sum(current_allocation.values())
    spend_target = current_spend
    
    # Apply scenario adjustments
    if scenario_type == 'price_change' and isinstance(change_value, dict):
        # Apply price changes to specific suppliers
        for supplier, change_pct in change_value.items():
            mask = scenario_df[analysis_dimension] == supplier
            scenario_df.loc[mask, 'Total CIU curr / vol'] *= (1 + change_pct/100)
            
    elif scenario_type == 'budget_change' and isinstance(change_value, (int, float)):
        # Adjust spend target based on percentage change
        spend_target = current_spend * (1 + change_value/100)
    
    # Run optimization with the scenario
    return simple_optimize_portfolio(
        scenario_df,
        analysis_dimension,
        cost_weight,
        diversity_weight,
        spend_target
    )

def main():
    df: pd.DataFrame = st.session_state.get('data', None)
    if df is None:
        st.title("Portfolio Optimization")
        st.warning("Please upload your data on the Data Upload page to start Portfolio Optimization.")
        return
    
    tabs = st.tabs(["Portfolio Optimization", "Flavor Spend Optimization"])
    
    with tabs[0]:
        st.title("Portfolio Optimization")

        st.markdown("""
        This tool helps you optimize your supplier allocation to balance cost, diversity, and risk.
        """)
        
        # Get primary dimension for analysis (usually Supplier)
        analysis_dimension = st.selectbox(
            "Select dimension to optimize",
            options=["Supplier", "Region", "Country"],
            index=0,
            help="Choose which dimension you want to optimize allocation across"
        )
        
        # Optional product filter
        product_filter = st.multiselect(
            "Filter by Product (optional)",
            options=df['Product'].unique(),
            default=[],
            help="Select specific products to focus the optimization on"
        )
        
        filtered_df = df
        if product_filter:
            filtered_df = df[df['Product'].isin(product_filter)]
            
        if filtered_df.empty:
            st.warning("No data matches your filter criteria.")
            return
        
        # Calculate current allocation
        allocation_df = filtered_df.groupby(analysis_dimension).agg({
            'Flavor Spend': 'sum',
            'Total CIU curr / vol': ['mean', 'std']
        })
        
        # Format column names for readability
        allocation_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in allocation_df.columns]
        
        allocation_df = allocation_df.reset_index()

        # Calculate totals and percentages
        total_spend = allocation_df['Flavor Spend_sum'].sum()
        allocation_df['Percentage'] = (allocation_df['Flavor Spend_sum'] / total_spend) * 100
        
        # Calculate HHI for concentration
        current_shares = dict(zip(allocation_df[analysis_dimension], allocation_df['Percentage'] / 100))
        hhi = calculate_hhi(current_shares)
        
        # Display current metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spend", f"${total_spend:,.2f}")
        with col2:
            weighted_ciu = (allocation_df['Total CIU curr / vol_mean'] * allocation_df['Flavor Spend_sum']).sum() / total_spend
            st.metric("Weighted Avg CIU", f"${weighted_ciu:.2f}")
        with col3:
            st.metric("HHI Index", f"{hhi:.1f}", help="Herfindahl-Hirschman Index - measures concentration")
        
        # Show current allocation in a chart
        st.subheader(f"Current {analysis_dimension} Allocation")
        
        # Format data for display
        display_df = allocation_df.copy()
        display_df['Flavor Spend_sum'] = display_df['Flavor Spend_sum'].round(2)
        display_df['Percentage'] = display_df['Percentage'].round(2)
        display_df['Total CIU curr / vol_mean'] = display_df['Total CIU curr / vol_mean'].round(2)
        
        fig = px.pie(
            allocation_df, 
            values='Flavor Spend_sum', 
            names=analysis_dimension,
            title=f'Current {analysis_dimension} Allocation',
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig)
        
        # Show allocation table
        st.dataframe(
            display_df[[analysis_dimension, 'Flavor Spend_sum', 'Percentage', 'Total CIU curr / vol_mean']].rename(
                columns={
                    'Flavor Spend_sum': 'Total Spend',
                    'Percentage': 'Allocation %',
                    'Total CIU curr / vol_mean': 'Avg CIU'
                }
            )
        )
        
        # Optimization parameters
        st.subheader("Optimization Settings")
        st.markdown("Adjust these settings to balance your optimization goals:")
        
        # Simple sliders for weights
        col1, col2 = st.columns(2)
        with col1:
            cost_weight = st.slider(
                "Cost Importance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values prioritize cost reduction"
            )
        with col2:
            diversity_weight = 1.0 - cost_weight
            st.slider(
                "Diversity Importance",
                min_value=0.0,
                max_value=1.0,
                value=diversity_weight,
                step=0.1,
                disabled=True,
                help="Automatically adjusted to ensure weights sum to 1.0"
            )
        
        # Budget settings
        budget_col1, budget_col2 = st.columns(2)
        with budget_col1:
            budget_change = st.slider(
                "Budget Change",
                min_value=-20,
                max_value=20,
                value=0,
                step=5,
                format="%d%%",
                help="Adjust total budget up or down"
            )
        with budget_col2:
            target_spend = total_spend * (1 + budget_change/100)
            st.metric(
                "Target Spend", 
                f"${target_spend:,.2f}",
                f"{budget_change:+.1f}%" if budget_change != 0 else None
            )
        
        # Allocation constraints
        st.subheader("Allocation Constraints")
        constraint_col1, constraint_col2 = st.columns(2)
        with constraint_col1:
            minimum_allocation_upper_limit = (100 // display_df[analysis_dimension].nunique())
            min_allocation = st.slider(
                "Minimum Allocation per Supplier",
                min_value=0,
                max_value=minimum_allocation_upper_limit,
                value=0,
                step=1,
                format="%d%%",
                help="Minimum percentage allocation per supplier"
            )
        with constraint_col2:
            maximum_allocation_lower_limit = 100 - ((display_df[analysis_dimension].nunique() - 1) * minimum_allocation_upper_limit)
            max_allocation = st.slider(
                "Maximum Allocation per Supplier",
                min_value=maximum_allocation_lower_limit,
                max_value=100,
                value=(100 - min_allocation),
                step=5,
                format="%d%%",
                help="Maximum percentage allocation to any single supplier"
            )
        
        # Run optimization button
        if st.button("Optimize Portfolio", type="primary"):
            with st.spinner("Running optimization..."):
                optimized_allocation, metrics = simple_optimize_portfolio(
                    filtered_df,
                    analysis_dimension,
                    cost_weight,
                    diversity_weight,
                    target_spend,
                    min_allocation,
                    max_allocation
                )
                
                if not metrics:
                    st.error("Optimization failed. Please adjust your parameters and try again.")
                else:
                    st.session_state['optimized_allocation'] = optimized_allocation
                    st.session_state['metrics'] = metrics
                    # Display optimization results
                    st.subheader("Optimization Results")

                    # Create comparison table
                    comparison_data = []
                    current_allocation = dict(zip(allocation_df[analysis_dimension], allocation_df['Flavor Spend_sum']))
                    
                    for entity in set(list(current_allocation.keys()) + list(optimized_allocation.keys())):
                        current = current_allocation.get(entity, 0)
                        optimized = optimized_allocation.get(entity, 0)
                        change = optimized - current
                        change_pct = (change / current * 100) if current > 0 else 0
                        
                        comparison_data.append({
                            analysis_dimension: entity,
                            'Current': current,
                            'Optimized': optimized,
                            'Change': change,
                            'Change %': change_pct
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Optimized', ascending=False)
                    
                    # Calculate percentages for visualization
                    comparison_df['Current %'] = comparison_df['Current'] / comparison_df['Current'].sum() * 100
                    comparison_df['Optimized %'] = comparison_df['Optimized'] / comparison_df['Optimized'].sum() * 100
                    
                    # Format for display
                    comparison_df['Current'] = comparison_df['Current'].round(2)
                    comparison_df['Optimized'] = comparison_df['Optimized'].round(2)
                    comparison_df['Change'] = comparison_df['Change'].round(2)
                    comparison_df['Change %'] = comparison_df['Change %'].round(2)
                    comparison_df['Current %'] = comparison_df['Current %'].round(2)
                    comparison_df['Optimized %'] = comparison_df['Optimized %'].round(2)

                    # Metrics comparison
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Total Spend ($)", comparison_df['Optimized'].sum().round(2),
                                delta=f"{comparison_df['Optimized'].sum() - comparison_df['Current'].sum():,.2f} ({(comparison_df['Optimized'].sum() - comparison_df['Current'].sum()) / comparison_df['Current'].sum() * 100:.1f}%)",
                                delta_color="inverse" if comparison_df['Optimized'].sum() >= comparison_df['Current'].sum() else "normal")
                    with metric_col2:
                        st.metric(
                            "Average CIU",
                            f"${metrics['weighted_avg_ciu']:.2f}",
                            f"{metrics['ciu_change']:.1f}%",
                            delta_color="normal" if metrics['ciu_change'] >= 0 else "inverse"
                        )
                        st.caption(f"Cost Weight: {cost_weight:.1f}")
                    with metric_col3:
                        st.metric(
                            "HHI Index", 
                            f"{metrics['hhi']:.1f}",
                            f"{metrics['hhi_change']:.1f}%",
                            delta_color="normal" if metrics['hhi_change'] >= 0 else "inverse"
                        )
                        st.caption(f"Diversity Weight: {diversity_weight:.1f}")
                    with metric_col4:
                        # Overall optimization metric
                        optimization_score = (metrics['ciu_change'] + metrics['hhi_change'])/2
                        st.metric(
                            "Overall Improvement", 
                            f"{optimization_score:.1f}%",
                            delta_color="normal" if optimization_score >= 0 else "inverse"
                        )
                    
                    # Show comparison chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df[analysis_dimension],
                        y=comparison_df['Current %'],
                        name='Current Allocation (%)',
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df[analysis_dimension],
                        y=comparison_df['Optimized %'],
                        name='Optimized Allocation (%)',
                        marker_color='lightgreen'
                    ))
                    
                    fig.update_layout(
                        title='Current vs. Optimized Allocation (%)',
                        xaxis_title=analysis_dimension,
                        yaxis_title='Allocation (%)',
                        barmode='group',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Show comparison table
                    st.subheader("Allocation Comparison")
                    st.dataframe(comparison_df.reset_index(drop=True).rename(
                        columns={
                            'Current': 'Current Spend ($)',
                            'Optimized': 'Optimized Spend ($)',
                            'Change': 'Change ($)',
                            'Change %': 'Change (%)'
                        }
                    ))
                    
                    # Key recommendations
                    st.subheader("Key Recommendations")
                    
                    # Find top increases and decreases
                    top_changes = comparison_df.sort_values(by='Change', key=abs, ascending=False)
                    
                    for _, row in top_changes.iterrows():
                        if row['Change'] > 0:
                            st.success(f"Increase allocation to {row[analysis_dimension]} by ${row['Change']:,.2f} ({row['Change %']:.1f}%)")
                        elif row['Change'] < 0:
                            st.error(f"Decrease allocation to {row[analysis_dimension]} by ${abs(row['Change']):,.2f} ({abs(row['Change %']):.1f}%)")

    with tabs[1]:
        st.title("Flavor Spend Optimization")
        
        st.markdown("""
        This tool helps you optimize your flavor spend to balance cost based on historical spend and current CIU.
        """)

        # Get primary dimension for analysis (usually Supplier)
        analysis_dimensions = st.multiselect(
            "Select dimensions to optimize",
            options=df.columns.tolist(),
            default=[col for col in df.columns.tolist() if pd.api.types.is_object_dtype(df[col])],
            help="Choose which dimensions you want to optimize allocation across"
        )

        ciu_aggregation_method = st.selectbox(
            "CIU Aggregation Method",
            options=["Minimum", "25th Percentile", "Median", "Mean"],
            index=3,
            help="Select the CIU aggregation method to use for optimization"
        )

        if ciu_aggregation_method == "Minimum":
            constraint_function = lambda x: pd.Series.min(x)
        elif ciu_aggregation_method == "25th Percentile":
            constraint_function = lambda x: pd.Series.quantile(x, 0.25)
        elif ciu_aggregation_method == "Median":
            constraint_function = lambda x: pd.Series.median(x)
        elif ciu_aggregation_method == "Mean":
            constraint_function = lambda x: pd.Series(x).mean()

        if not analysis_dimensions or len(analysis_dimensions) == 0:
            st.warning("Please select at least one dimension to optimize.")
            return
        
        with st.expander("Filters", expanded=False):
            for dim in analysis_dimensions:
                unique_values = df[dim].unique()
                selected_values = st.multiselect(
                    f"Filter by {dim}",
                    options=unique_values,
                    default=unique_values.tolist(),
                    help=f"Select specific {dim} values to focus the optimization on"
                )
                if selected_values:
                    df = df[df[dim].isin(selected_values)]
        if df.empty:
            st.warning("No data matches your filter criteria.")
            return

        total_spend = df['Flavor Spend'].sum()

        optimized_df = df.groupby(analysis_dimensions).agg({'CIU curr / vol': lambda x: constraint_function(x), 'Flavor Spend': 'size'}).reset_index() # 25th percentile of CIU curr / vol
        optimized_df.rename(columns={'CIU curr / vol': 'Optimized CIU [$]', 'Flavor Spend': 'Recond Count'}, inplace=True)
        optimized_df = optimized_df.sort_values(by='Recond Count', ascending=False)

        # st.subheader("Count Display")
        # st.write(df.groupby(analysis_dimensions).size().reset_index(name='Count').sort_values(by='Count', ascending=False))
        
        st.write("Select a row to see its details:")
        selected_row = st.dataframe(optimized_df, selection_mode="single-row", on_select="rerun", use_container_width=True, hide_index=True)

        if len(selected_row['selection']['rows']) > 0:
            selected_row_data = selected_row['selection']['rows'][0]
            optimized_values = optimized_df.iloc[selected_row_data]

            temp_show = optimized_values.reset_index()
            temp_show.columns = ['Dimension', 'Optimized Value']
            st.subheader("Selected Row Details")
            st.write(temp_show)

            updated_df = df.copy()
            updated_df = updated_df.merge(
                optimized_df,
                on=analysis_dimensions,
                how='left',
                suffixes=('', '_optimized')
            )

            selected_row_details_df = updated_df.groupby(analysis_dimensions).get_group(tuple([optimized_values[dim] for dim in analysis_dimensions])).reset_index(drop=True)
            selected_row_details_df['Flavor Spend'] = selected_row_details_df['FG volume / year'] * selected_row_details_df['CIU curr / vol']
            selected_row_details_df['Optimized Flavor Spend'] = selected_row_details_df['FG volume / year'] * selected_row_details_df['Optimized CIU [$]']

            st.subheader("Optimization Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Spend", f"${selected_row_details_df['Flavor Spend'].sum():,.2f}")
            with c2:
                st.metric("Optimized Flavor Spend", f"${selected_row_details_df['Optimized Flavor Spend'].sum():,.2f}")
            with c3:
                st.metric("Spend Change", f"${np.abs(selected_row_details_df['Optimized Flavor Spend'].sum() - selected_row_details_df['Flavor Spend'].sum()):,.2f}",
                          delta=f"{((selected_row_details_df['Optimized Flavor Spend'].sum() - selected_row_details_df['Flavor Spend'].sum()) / selected_row_details_df['Flavor Spend'].sum() * 100).round(2)} %",
                          delta_color="inverse")
            

            st.write("Optimized Details")
            st.write(selected_row_details_df)
        else:
            st.info("Please select a row to see its details.")
