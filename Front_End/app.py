import streamlit as st
import os

# Streamlit configuration
st.set_page_config(layout="wide")

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
plus_icon_path = os.path.join(current_dir, "plus-icon.png")
search_icon_path = os.path.join(current_dir, "magnifying-glass-icon.png")

# Define session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Function to handle navigation
def navigate_to(page):
    st.session_state.page = page

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    /* Hide Streamlit's default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Navigation Bar Styling */
    .nav-bar {{
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
    }}

    .nav-logo {{
        font-size: 24px;
        font-weight: bold;
    }}

    /* Round Buttons Styling */
    .custom-button {{
        margin-top: 250px;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        background-color: grey;
        border: none;
        cursor: pointer;
        outline: none;
        transition: transform 0.2s ease;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .custom-button:hover {{
        transform: scale(1.1);
    }}

    .button-container {{
        display: flex;
        justify-content: space-around;
        margin-top: 30px;
        margin-bottom: 120px; /* Add white space below */
    }}

    .button-icon {{
        width: 100px;
        height: 100px;
        position: absolute;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-logo">PAPA.IA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Page rendering logic
if st.session_state.page == "home":
    # Full-width image
    st.image("ModelAIcrop.jpg", use_column_width=True, output_format="auto")

    # HTML buttons with icons
    st.markdown(
        f"""
        <div class="button-container">
            <form method="post">
                <button type="submit" name="add" class="custom-button">
                    <img src="file://{plus_icon_path}" class="button-icon" alt="Add Icon">
                </button>
            </form>
            <form method="post">
                <button type="submit" name="search" class="custom-button">
                    <img src="file://{search_icon_path}" class="button-icon" alt="Search Icon">
                </button>
            </form>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check which button was pressed
    if "add" in st.query_params:
        navigate_to("add")
    elif "search" in st.query_params:
        navigate_to("search")

elif st.session_state.page == "add":
    st.title("Add Page")
    st.write("This is where users can add new items.")
    if st.button("Go Back"):
        navigate_to("home")

elif st.session_state.page == "search":
    st.title("Search Page")
    st.write("This is where users can search for items.")
    if st.button("Go Back"):
        navigate_to("home")
