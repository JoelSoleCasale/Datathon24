import streamlit as st
from navigation import render_navigation_bar

def app():
    # Render navigation bar
    render_navigation_bar()

    # Add page content
    st.markdown(
        """
        <style>
        /* Hide Streamlit's default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Add Page")
    st.write("This is where users can add new items.")
