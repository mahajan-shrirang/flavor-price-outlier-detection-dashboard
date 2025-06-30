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
        0: "Preference Selection and AI Training",
        1: "Recommendations",
    }

    st.title(pages[page])

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
