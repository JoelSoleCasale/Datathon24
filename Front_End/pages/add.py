import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
import pandas as pd
from PIL import Image
import sys
import os

# Add the ../Back_End directory to the Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(backend_path)

from Back_End import backend

# CSS Styling for Coherence
st.markdown(
    """
    <style>
    /* General App Styling */
    body {
        font-family: Arial, sans-serif;
        color: #333;
    }

    /* Remove Streamlit default padding and margin */
    .css-18e3th9 { padding: 0; }
    .css-1d391kg { padding: 0; margin: 0; }

    /* Hide Streamlit's default hamburger menu and footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

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
    .nav-logo a {
        text-decoration: none;
        font-size: 24px;
        font-weight: bold;
        color: black;
    }
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
    .nav-links a:hover { color: grey; }

    /* Section Headers */
    h1, h2, h3 {
        font-family: Arial, sans-serif;
        font-weight: bold;
        color: #333;
    }

    /* Button Styling */
    button {
        font-family: Arial, sans-serif;
        font-size: 14px;
        font-weight: bold;
        color: white;
        background-color: grey;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    button:hover {
        background-color: #555;
    }

    /* AgGrid Table Styling */
    .ag-theme-streamlit {
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
st.markdown(
    """
    <div class="nav-bar">
        <div class="nav-logo"><a href="app">PAPA.IA</a></div>
        <div class="nav-links">
            <a href="add">Add</a>
            <a href="search">Search</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Functions to initialize tables
def initialize_metadata_table(image):
    df = pd.read_csv('../data/product_data.csv')
    return pd.DataFrame({"Metadata": df.columns, "Values": df.iloc[1, :]})

def initialize_attributes_table(image, metadata):
    df = backend.predict_attributes(image, metadata)
    return pd.DataFrame({"Attributes": df.columns, "Values": df.iloc[1, :]})

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["uploaded_image"] = None
    st.session_state["metadata"] = None

# Handlers for button actions
def go_to_step(step):
    st.session_state["step"] = step

# Add spacing between steps
st.divider()

# Step 1: Upload an image
if st.session_state["step"] == 1:
    st.write("### Step 1: Upload your image")
    st.image("pages/greenJ_model.jpg", use_column_width=True, output_format="auto")

    # File uploader inside a form
    with st.form(key="form1"):
        img = st.file_uploader(
            label="Upload a picture of the article to analyze it:", type=["jpg", "jpeg", "png"]
        )
        submit_image_button = st.form_submit_button(label="Submit image")

    # Handle image upload and validation
    if submit_image_button:
        if img:
            try:
                # Validate the image file with Pillow
                image = Image.open(img)
                image.verify()  # Raise an exception if the file is invalid

                # Save the validated image in session state
                st.session_state["uploaded_image"] = img
                st.session_state["step"] = 2  # Move to the next step
                st.experimental_rerun()  # Force a rerun to reflect the updated step immediately
            except Exception as e:
                st.error(f"Invalid image file: {e}")
        else:
            st.error("Please upload an image before submitting.")

# Step 2: Display metadata table
elif st.session_state["step"] == 2:
    st.write("### Step 2: Check the validity of the metadata")
    st.image(st.session_state["uploaded_image"], use_column_width=True, output_format="auto")

    df = initialize_metadata_table(st.session_state["uploaded_image"])
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True)
    grid_options = gb.build()

    st.write("### Interactive Metadata Table")
    response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
    )
    st.session_state["metadata"] = pd.DataFrame(response["data"])

    col1, col2 = st.columns(2)
    col1.button("Previous Step", on_click=go_to_step, args=(1,))
    col2.button("Next Step", on_click=go_to_step, args=(3,))

# Step 3: Display attributes table
elif st.session_state["step"] == 3:
    st.write("### Step 3: Check the attributes and their values")
    st.image(st.session_state["uploaded_image"], use_column_width=True, output_format="auto")

    attributes_df = initialize_attributes_table(st.session_state["uploaded_image"], st.session_state["metadata"])
    gb = GridOptionsBuilder.from_dataframe(attributes_df)
    gb.configure_default_column(editable=True)
    grid_options = gb.build()

    st.write("### Interactive Attributes Table")
    response = AgGrid(
        attributes_df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
    )

    col1, col2 = st.columns(2)
    col1.button("Previous Step", on_click=go_to_step, args=(2,))
    col2.button("Restart", on_click=go_to_step, args=(1,))
