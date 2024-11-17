import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(layout="wide")

# Global Styling for Uniformity
st.markdown(
    """
    <style>
    /* General Styling */
    body {
        font-family: Arial, sans-serif;
        color: #333;
    }
    .css-18e3th9, .css-1d391kg {
        padding: 0;
        margin: 0;
    }
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

    /* Pagination Button Styling */
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

    /* Product Display Styling */
    .product-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .product-title {
        font-size: 18px;
        font-weight: bold;
    }
    .product-details {
        font-size: 14px;
    }
    .product-image {
        display: block;
        margin: 10px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
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

# Define paths
DATA_PATH = '../Front_End/imageDB/merged_dataset.csv'
IMAGE_DIR = '../../datathon-fme-mango/archive/images/images'

# Load data
def load_data():
    return pd.read_csv(DATA_PATH)

# Load an image
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
st.title("Product Search Engine")
st.write("Search for products by keywords or apply filters from the sidebar.")

# Load dataset
data = load_data()

# Sidebar filters
st.sidebar.title("Filter Products")
sex_filter = st.sidebar.selectbox("Gender", ["All"] + data['des_sex'].dropna().unique().tolist())
age_filter = st.sidebar.selectbox("Age", ["All"] + data['des_age'].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Category", ["All"] + data['des_product_category'].dropna().unique().tolist())
rows_per_page = st.sidebar.slider("Results per Page", 5, 50, 10)

# Filter data based on user input
filtered_data = data.copy()
if sex_filter != "All":
    filtered_data = filtered_data[filtered_data['des_sex'] == sex_filter]
if age_filter != "All":
    filtered_data = filtered_data[filtered_data['des_age'] == age_filter]
if category_filter != "All":
    filtered_data = filtered_data[filtered_data['des_product_category'] == category_filter]

search_term = st.text_input("Search Products", placeholder="Enter keywords like color, category...")
if search_term:
    filtered_data = filtered_data[
        filtered_data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
    ]

# Pagination
total_results = len(filtered_data)
total_pages = (total_results - 1) // rows_per_page + 1
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
start_row = (st.session_state.current_page - 1) * rows_per_page
end_row = start_row + rows_per_page
paginated_data = filtered_data.iloc[start_row:end_row]

# Display results
st.write(f"Showing {start_row + 1} to {min(end_row, total_results)} of {total_results} results.")
for _, row in paginated_data.iterrows():
    with st.container():
        # Display product information
        st.markdown(
            f"""
            <div class="product-card">
                <div class="product-title"><strong>Product ID:</strong> {row['cod_modelo_color']}</div>
                <div class="product-details">
                    <strong>Category:</strong> {row['des_product_category']}<br>
                    <strong>Color:</strong> {row['des_color']}<br>
                    <strong>Gender:</strong> {row['des_sex']}<br>
                    <strong>Age:</strong> {row['des_age']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Retrieve all associated image filenames for the product
        product_images = row['des_filename'].split(';')  # Assuming filenames are stored as a semicolon-separated string
        
        # Display images side by side while keeping their original size
        image_columns = st.columns(len(product_images))
        for col, image_filename in zip(image_columns, product_images):
            image = load_image(image_filename)
            if image:
                with col:
                    st.image(image, caption=image_filename, use_column_width=False, width=200)  # Fixed width for uniformity
            else:
                with col:
                    st.write(f"Image not found for `{image_filename}`.")
    st.markdown("---")


# Pagination controls
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous") and st.session_state.current_page > 1:
        st.session_state.current_page -= 1
with col3:
    if st.button("Next") and st.session_state.current_page < total_pages:
        st.session_state.current_page += 1
with col2:
    st.write(f"Page {st.session_state.current_page} of {total_pages}")

# Scroll to top
scroll_to_top()
