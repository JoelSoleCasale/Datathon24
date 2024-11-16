import streamlit as st
from navigation import render_navigation_bar

def app():
    # Render navigation bar
    render_navigation_bar()

    # Search page content
    st.markdown(
        
    )
    st.title("Search Page")
    st.write("This is where users can search for items.")
