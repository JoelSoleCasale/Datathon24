import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
import pandas as pd
from PIL import Image

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

# Function to calculate initial values for the table
def calculate_initial_values(image):
    # Example of dynamically generated data (can be replaced with actual metadata extraction logic)
    df = pd.read_csv('../data/product_data.csv')
    data = {
        "Parameter": df.columns,
        "Value": df.iloc[1,:]
    }
    return pd.DataFrame(data)

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 1  # Tracks the current step
    st.session_state["uploaded_image"] = None
    st.session_state["metadata"] = None

# Add some spacing between steps
st.divider()

# Step 1: Upload an image
if st.session_state["step"] == 1:
    st.write("### Step 1: Upload your image")
    st.image("pages/greenJ_model.jpg", use_column_width=True, output_format="auto")

    with st.form(key="form1"):
        img = st.file_uploader(
            label="Upload a picture of the article to analyze it:", type=["jpg", "jpeg", "png"]
        )
        submit_image_button = st.form_submit_button(label="Submit image")

    if submit_image_button and img:
        try:
            # Validate the image file with Pillow
            image = Image.open(img)
            image.verify()  # Raise an exception if the file is invalid

            # Save the validated image in session state
            st.session_state["uploaded_image"] = img
            st.session_state["step"] = 2  # Move to the next step
        except Exception as e:
            st.error(f"Invalid image file: {e}")

# Step 2: Display interactive table
elif st.session_state["step"] == 2:
    st.write("### Step 2: Check the validity of the metadata")
    st.image(st.session_state["uploaded_image"], use_column_width=True, output_format="auto")

    # Generate initial table values
    df = calculate_initial_values(st.session_state["uploaded_image"])

    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True)  # Make all columns editable
    grid_options = gb.build()

    # Display the interactive table
    st.write("### Interactive Table")
    response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True
    )

    # Get the updated data
    st.session_state["metadata"] = pd.DataFrame(response["data"])

    # Navigation buttons
    col1, col2 = st.columns(2)
    if col1.button("Previous Step"):
        st.session_state["step"] = max(1, st.session_state["step"] - 1)
    if col2.button("Next Step"):
        st.session_state["step"] = 3  # Move to the next step

# Step 3: Display results
elif st.session_state["step"] == 3:
    st.write("### Step 3: Check the validity of the processed results")
    st.image(st.session_state["uploaded_image"], use_column_width=True, output_format="auto")

    st.write("### Processed Results")
    st.write("Metadata Table:")
    st.write(st.session_state["metadata"])

    # Save metadata to a file
    if st.button("Save Metadata"):
        st.session_state["metadata"].to_csv("metadata_output.csv", index=False)
        st.success("Metadata saved successfully!")

    # Navigation buttons
    col1, col2 = st.columns(2)
    if col1.button("Previous Step"):
        st.session_state["step"] = max(1, st.session_state["step"] - 1)
    if col2.button("Restart"):
        st.session_state["step"] = 1
        st.session_state["uploaded_image"] = None
        st.session_state["metadata"] = None