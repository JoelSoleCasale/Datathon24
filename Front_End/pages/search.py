import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(layout="wide")

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
    .nav-logo a {{
        text-decoration: none;
        font-size: 24px;
        font-weight: bold;
        color: black;
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

    </style>
    """,
    unsafe_allow_html=True,
)

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


# Define paths
DATA_PATH = '../Front_End/imageDB/merged_dataset.csv'  # Replace with the path to your CSV file
IMAGE_DIR = '../../datathon-fme-mango/archive/images/images'  # Replace with the directory containing images

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Function to load an image
def load_image(filename):
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(filepath):
        return Image.open(filepath)
    return None

# JavaScript to scroll to the top of the page
def scroll_to_top():
    st.components.v1.html(
        """
        <script>
            document.querySelector("button[aria-label='Next']").addEventListener("click", function() {
                window.scrollTo({top: 0, behavior: 'smooth'});
            });
            document.querySelector("button[aria-label='Previous']").addEventListener("click", function() {
                window.scrollTo({top: 0, behavior: 'smooth'});
            });
        </script>
        """,
        height=0,
    )

# Streamlit App
st.title("Product Search Engine with Pagination")

# Load dataset
data = load_data()

# Search term input
search_term = st.text_input("Search for a product (e.g., color, category, etc.):")

# Filter attributes
st.sidebar.title("Filter Options")
sex_filter = st.sidebar.selectbox("Filter by Gender:", ["All"] + data['des_sex'].dropna().unique().tolist())
age_filter = st.sidebar.selectbox("Filter by Age:", ["All"] + data['des_age'].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Filter by Product Category:", ["All"] + data['des_product_category'].dropna().unique().tolist())

# Filter data based on user input
filtered_data = data

if sex_filter != "All":
    filtered_data = filtered_data[filtered_data['des_sex'] == sex_filter]

if age_filter != "All":
    filtered_data = filtered_data[filtered_data['des_age'] == age_filter]

if category_filter != "All":
    filtered_data = filtered_data[filtered_data['des_product_category'] == category_filter]

if search_term:
    # Search term filtering across all columns
    filtered_data = filtered_data[
        filtered_data.apply(
            lambda row: row.astype(str).str.contains(search_term, case=False).any(),
            axis=1
        )
    ]

# Pagination
rows_per_page = st.sidebar.number_input("Results per page:", min_value=1, max_value=50, value=10)
total_results = len(filtered_data)
total_pages = (total_results - 1) // rows_per_page + 1

# Current page
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Paginate the data
start_row = (st.session_state.current_page - 1) * rows_per_page
end_row = start_row + rows_per_page
paginated_data = filtered_data.iloc[start_row:end_row]

# Display results
st.write(f"Showing results {start_row + 1} to {min(end_row, total_results)} of {total_results}:")
for _, row in paginated_data.iterrows():
    # Display product information
    st.subheader(f"Product ID: {row['cod_modelo_color']}")
    st.write(f"**Category:** {row['des_product_category']}")
    st.write(f"**Color:** {row['des_color']}")
    st.write(f"**Gender:** {row['des_sex']}")
    st.write(f"**Age:** {row['des_age']}")

    # Load and display the image
    image_filename = f"{row['des_filename']}"
    image = load_image(image_filename)
    if image:
        st.image(image, caption=image_filename, use_column_width=False, width=200)  # Adjust width as needed
    else:
        st.write(f"Image not found for `{image_filename}`.")
    st.markdown("---")

# Footer
st.write("Use the sidebar to filter products or the search box to find specific items.")

# Pagination controls
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("Previous", key = "previous") and st.session_state.current_page > 1:
        st.session_state.current_page -= 1

with col3:
    if st.button("Next", key = "next") and st.session_state.current_page < total_pages:
        st.session_state.current_page += 1

# Display current page number
with col2:
    st.write(f"Page {st.session_state.current_page} of {total_pages}")

    # Add scroll-to-top functionality
scroll_to_top()