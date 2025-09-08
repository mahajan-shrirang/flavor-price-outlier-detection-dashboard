import streamlit as st
from streamlit_option_menu import option_menu

from views import (
    ciu_analysis,
    data_upload,
    spend_impact
)
from views.ai_recommendation import home_page
from views.clustering_analysis import cluster_analysis_page
from views.risk_analysis import risk_analysis_page
from views.portfolio_optimization import portfolio_optimization_page

st.set_page_config(page_title="Flavor Data Dashboard", page_icon=":bar_chart:", layout="wide")


with st.sidebar:
    st.title("Flavor Data Dashboard")
    st.write("Select a page to navigate:")
    
    # Quick help section
    with st.expander("💡 Quick Start Guide"):
        st.markdown("""
        **New to the dashboard?**
        1. Start with **📤 Data Upload**
        2. Try **💰 Spend Impact** for quick insights
        3. Use **📊 CIU Analysis** to find pricing errors
        4. Explore **🤖 AI Recommendations** for savings
        5. Check **⚠️ Risk Analysis** quarterly
        6. Use **⚖️ Portfolio Optimization** for strategy

        💡 *Tip: Click the ℹ️ icon in each tab for detailed instructions*
        """)
    
    page = option_menu(
        menu_title=None,
        options=["Data Upload", 'CIU Analysis', "Spend Impact", "AI Recommendation", "Clustering Analysis", "Risk Analysis", "Portfolio Optimization"], 
        icons=['cloud-upload', 'bar-chart', 'bag', 'robot', 'search', 'shield-exclamation', 'briefcase'],
        menu_icon="list", default_index=0
    )

if page == "Data Upload":
    data_upload.main()
elif page == "CIU Analysis":
    ciu_analysis.main()
elif page == "Spend Impact":
    spend_impact.main()
elif page == "AI Recommendation":
    home_page.main()
elif page == "Clustering Analysis":
    cluster_analysis_page.main()
elif page == "Risk Analysis":
    risk_analysis_page.main()
elif page == "Portfolio Optimization":
    portfolio_optimization_page.main()
else:
    st.error("Page not found. Please select a valid page from the sidebar.")
