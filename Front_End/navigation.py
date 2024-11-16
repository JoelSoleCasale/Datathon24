import streamlit as st

def render_navigation_bar():
    # Custom CSS for navigation bar
    st.markdown(
        """
        <style>
        /* Remove Streamlit default padding and margin */
        .css-18e3th9 {
            padding: 0;
        }
        .css-1d391kg {
            padding: 0;
            margin: 0;
        }

        /* Hide Streamlit's default hamburger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Navigation Bar Styling */
        .nav-bar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: white;
            color: black;
            height: 60px;
            display: flex;
            justify-content: center;
            align-items: center;
            border-bottom: 2px solid grey;
            z-index: 1000;
            font-family: Arial, sans-serif;
        }

        /* Logo Styling */
        .nav-logo {
            font-size: 24px;
            font-weight: bold;
        }

        /* Navigation Links */
        .nav-links {
            position: absolute;
            right: 20px;
        }

        .nav-links a {
            color: black;
            text-decoration: none;
            font-weight: bold;
            margin: 0 15px;
        }

        .nav-links a:hover {
            color: grey;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Navigation bar HTML
    st.markdown(
        """
        <div class="nav-bar">
            <div class="nav-logo">MANGO</div>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/Add">Add</a>
                <a href="/Search">Search</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
