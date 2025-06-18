import streamlit as st
from streamlit_option_menu import option_menu

from views import (
    ciu_analysis,
    data_upload,
    spend_impact
)

st.set_page_config(page_title="Data Analyser", page_icon=":bar_chart:", layout="wide")


with st.sidebar:
    st.title("Data Analyser")
    st.write("Select a page to navigate:")
    page = option_menu(
        menu_title=None,
        options=["Data Upload", 'CIU Analysis', "Spend Impact"], 
        icons=['cloud-upload', 'bar-chart', 'bag'],
        menu_icon="list", default_index=0
    )

if page == "Data Upload":
    data_upload.main()
elif page == "CIU Analysis":
    ciu_analysis.main()
elif page == "Spend Impact":
    spend_impact.main()
else:
    st.error("Page not found. Please select a valid page from the sidebar.")
