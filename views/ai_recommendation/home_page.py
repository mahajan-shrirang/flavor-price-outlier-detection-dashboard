import streamlit as st
import pandas as pd
from views.ai_recommendation.components import preference_selection_page, recommendation_page

def main():
    df = st.session_state.get('data', None)
    if df is None:
        st.warning("Please upload your data on the Data Upload page to start analyzing AI Recommendations.")
        return

    page = st.session_state.get("ai_recommendation_page", 0)

    pages = {
        0: "🤖 AI Recommendation - Setup and Training",
        1: "🤖 AI Recommendation - View Recommendations",
    }

    st.title(pages[page])
    
    # Instructions for users (only show on first page)
    if page == 0:
        with st.expander("ℹ️ How to use this tab", expanded=False):
            st.markdown("""
            **What it does:** Uses artificial intelligence to find similar suppliers or products with different prices, helping you identify cost-saving opportunities.
            
            **How to use it:**
            
            **Page 1: Setup and Training**
            1. **Select features** that the AI should consider (e.g., Country, Product Type, Application)
            2. **Choose matching features** for finding similar items
            3. **Click "Train AI"** and wait for processing to complete
            4. **Click "Next"** to see recommendations
            
            **Page 2: View Recommendations**
            1. **Set your preferences**:
               - Minimum similarity (how similar items should be)
               - Minimum cost difference (how much savings to show)
            2. **Apply filters** to focus on specific categories
            3. **Select a recommendation** from the table to see details
            4. **Review the comparison** between similar items
            
            **What you'll learn:**
            - Alternative suppliers with similar products but lower prices
            - Potential cost savings opportunities
            - Why certain suppliers are more expensive than others
            - Data-driven negotiation points
            
            **When to use this:**
            - Looking for alternative suppliers
            - Preparing for price negotiations
            - Benchmarking current suppliers
            - Cost optimization projects
            """)

    if page == 0:
        preference_selection_page.main()
    elif page == 1:
        recommendation_page.main()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        if page == 1:
            if st.button("⬅️ Back", disabled=page == 0, use_container_width=True):
                st.session_state["ai_recommendation_page"] = page - 1
                st.rerun()
    with col2:
        if page == 0:
            if st.button("Next ➡️", disabled=page == 1, use_container_width=True):
                st.session_state["ai_recommendation_page"] = page + 1
                st.rerun()
