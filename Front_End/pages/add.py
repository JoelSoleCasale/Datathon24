import streamlit as st

st.markdown(
    f"""
    <style>
    /* Remove Streamlit default padding and margin */
    .css-18e3th9 {{
        padding: 0;
    }}
    .css-1d391kg {{
        padding: 0;
        margin: 0;
    }}

    /* Hide Streamlit's default hamburger menu and footer */
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

    /* Logo Styling */
    .nav-logo {{
        font-size: 24px;
        font-weight: bold;
    }}

    /* Navigation Links */
    .nav-links {{
        position: absolute;
        right: 20px;
    }}

    .nav-links a {{
        color: black;
        text-decoration: none;
        font-weight: bold;
        margin: 0 15px;
    }}

    .nav-links a:hover {{
        color: grey;
    }}

    /* Full-width image */
    .full-width-img {{
        width: 100%;
        height: calc(100vh - 60px); /* Full height minus the nav bar height */
        object-fit: cover;
        margin-top: 60px; /* Push below the nav bar */
    }}

    /* Larger Buttons Below Image */
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
        width: 100px; /* Adjust the icon size */
        height: 100px;
        position: absolute;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-logo">PAPA.IA</div>
        <div class="nav-links">
            <a href="add">Add</a>
            <a href="search">Search</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.image("pages\greenJ_model.jpg", use_column_width=True, output_format="auto")

def analyze (img):
    return "FUNCIONA!! :D"

img = st.file_uploader(type = 'jpg', label = "upload a picture of the article to analyze it:")
if img:
    st.image(img, use_column_width=True, output_format="auto")
    labels = analyze(img)
    st.write(labels)

