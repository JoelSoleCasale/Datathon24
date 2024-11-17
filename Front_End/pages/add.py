import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
import pandas as pd
from PIL import Image
import sys
import os
from rembg import remove
import io

# Function to remove the background and set it to white
def remove_background(image: Image.Image) -> Image.Image:
    # Convert PIL image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    # Use rembg to remove the background
    no_bg_bytes = remove(img_bytes)

    # Convert the result back to a PIL Image
    no_bg_image = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")

    # Create a white background
    white_bg = Image.new("RGBA", no_bg_image.size, (255, 255, 255, 255))

    # Composite the no-background image over the white background
    final_image = Image.alpha_composite(white_bg, no_bg_image).convert("RGB")

    return final_image

# Add the ../Back_End directory to the Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(backend_path)

from backend import backend

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

# Allowed values for attributes
ATTRIBUTE_ALLOWED_VALUES = {
    "cane_height_type": ["INVALID", "Cuña abotinada", "Alta", "Bloque", "Cuña", "Baja", "Media"],
    "closure_placement": ["INVALID", "Cierre Delantero", "Sin cierre", "Cuello", "Lateral", "Cierre Hombro", "Cierre Trasero"],
    "heel_shape_type": [
        "INVALID", "Kitten", "Plano", "Bloque", "Embudo", "Rectangular", "Plataforma", 
        "De aguja", "Cuña", "Plataforma plana", "Trompeta", 
        "Plataforma en la parte delantera"
    ],
    "knit_structure": ["INVALID", "Punto fino", "Punto medio", "Punto grueso", "UNKNOWN", "Hecho a mano"],
    "length_type": [
        "INVALID", "Crop", "Medio", "Largo", "Standard", "Corto", "Midi", "Mini/Micro", 
        "Tobillero", "Maxi", "Capri", "Tres Cuartos", "Asimétrico"
    ],
    "neck_lapel_type": [
        "INVALID", "Redondo", "Pico", "Regular", "Caja", "Chimenea", "Perkins", "Capucha", 
        "Camisero", "Espalda Abierta", "Halter", "Alto/Envolvente", "Solapa", 
        "Hawaiano/Bowling", "Asimétrico", "Escotado", "Barca", "Mao", "Polo", 
        "Cruzado", "Babydoll/Peter Pan", "Off Shoulder", "Kimono", "Cisne", 
        "Palabra Honor", "Peak Lapel", "Button Down", "Shawl", "Panadero", 
        "Drapeado", "Smoking", "Cutaway", "Waterfall", "Sin solapa"
    ],
    "silhouette_type": [
        "INVALID", "Slim", "Oversize", "Recto", "Regular", "Evase", "Slouchy", "Cargo", 
        "Parachute", "Jogger", "Tapered", "5 Bolsillos", "Skinny", "Wide leg", 
        "Paperbag", "Relaxed", "Acampanado/Flare", "Ancho", "Lápiz", 
        "Acampanado/Bootcut", "Chino", "Push Up", "Palazzo", "Fino", 
        "Modern slim", "Culotte", "Superslim", "Mom", "Boyfriend", "Halter", 
        "Loose", "Carrot", "Bandeau", "Sarouel"
    ],
    "sleeve_length_type": ["INVALID", "Larga", "Corta", "Sin Manga", "Tirante Ancho", "Tres Cuartos", "Tirante Fino"],
    "toecap_type": ["INVALID", "Redonda", "Abierta", "Con punta", "Cuadrada"],
    "waist_type": ["INVALID", "Ajustable/Goma", "High Waist", "Regular Waist", "Low Waist"],
    "woven_structure": ["INVALID", "Ligero", "Medio", "Pesado", "Elástico"]
}

METADATA_ALLOWED_VALUES = {
    "cod_color": [
        None, "82", "01", "70", "37", "43", "56", "02", "76", "52", "36", "08", "05", "95", "99", "88", "06", 
        "91", "41", "TC", "40", "50", "20", "94", "45", "12", "92", "15", "68", "26", "69", "65", "BB", 
        "TN", "TM", "TG", "35", "85", "78", "79", "07", "81", "TO", "80", "87", "10", "96", "16", "59", 
        "60", "09", "CO", "30", "83", "32", "90", "46", "49", "54", "03", "OR", "14", "93", "84", "PL", 
        "77", "CU", "17", "23", "28", "18", "74", "GM", "44", "27", "97", "11", "57", "55", "BL", "39", 
        "31", "61", "CG", "48", "25", "51", "DC", "TA", "TU", "21", "GC", "53", "72", "38", "47", "DO", 
        "04", "58", "75", "42", "TS", "33", "34", "19", "DI", "98", "24", "73", "62", "66", "67", "71", 
        "TL", "89", "13", "63", "64", "86", "TD", "22", "29"
    ],
    "des_sex": [None, "Female", "Male", "Unisex"],
    "des_age": [None, "Kids", "Teen", "Adult", "Baby", "Newborn"],
    "des_line": [None, "KIDS", "MAN", "WOMAN"],
    "des_fabric": [None, "TRICOT", "WOVEN", "CIRCULAR", "JEANS", "LEATHER", "ACCESSORIES", "SYNTHETIC LEATHER"],
    "des_product_category": [
        None, "Tops", "Dresses, jumpsuits and Complete set", "Bottoms", 
        "Outerwear", "Accesories", "Swim and Intimate"
    ],
    "des_product_aggregated_family": [
        None, "Sweaters and Cardigans", "Dresses and jumpsuits", "T-shirts", "Jeans", 
        "Trousers & leggings", "Shirts", "Jackets and Blazers", "Coats and Parkas", 
        "Tops", "Accessories", "Skirts and shorts"
    ],
    "des_product_family": [
        None, "Sweater", "Dresses", "T-shirt", "Jeans", "Sweatshirts", "Leggings and joggers", 
        "Shirt", "Trousers", "Blazers", "Coats", "Jackets", "Poloshirts", "Jumpsuit", 
        "Cardigans", "Parkas", "Tops", "Footwear", "Skirts", "Shorts", "Puffer coats", 
        "Outer Vest", "Vest", "Hats, scarves and gloves", "Bodysuits", "Leather jackets", 
        "Trenchcoats"
    ],
    "des_product_type": [
        None, "Sweater", "Dress", "T-Shirt", "Jeans", "Sweatshirt", "Joggers", "Blouse", 
        "Trousers", "Blazer", "Shirt", "Coat", "Jacket", "Poloshirt", "Leggings", 
        "Jumpsuit", "Cardigan", "Parka", "Top", "Ankle Boots", "Skirt", "Shoes", 
        "Sandals", "Trainers", "Bermudas", "Puffer coat", "Outer vest", "Shorts", 
        "Overall", "Pichi", "Vest", "Headband", "Slippers", "Bodysuit", "Leather Jacket", 
        "Boots", "Romper", "Overshirt", "Trenchcoat", "Beach Shoes", "Clogs", 
        "Sweater Vest", "Cardigan Vest", "Cape", "Poncho", "Kaftan", "Bolero", 
        "Jacket (Cazadora)"
    ],
    "des_color": [
        None, "ROSA LIGHT", "BLANCO", "ROJO", "KHAKI", "VERDE", "NAVY", "OFFWHITE", "GRANATE", 
        "AZUL", "OLIVA", "BEIGE", "CRUDO", "ANTRACITA", "NEGRO", "FUCSIA", "PIEDRA", 
        "GRIS CLARO VIGORE", "MENTA", "TEJANO CLARO", "VERDE PASTEL", "CELESTE", 
        "NARANJA", "GRIS MEDIO VIGORE", "AGUA", "AMARILLO", "GRIS", "MOSTAZA", 
        "MALVA", "PEACH", "MARINO", "MORADO", "BLUEBLACK", "TEJANO NEGRO", 
        "TEJANO MEDIO", "TEJANO GRIS", "TOPO", "ROSA", "BURDEOS", "CALDERO", 
        "ARENA", "ROSA PASTEL", "TEJANO OSCURO", "NUDE", "CORAL", "VAINILLA", 
        "GRIS OSCURO VIGORE", "OCRE", "PETROLEO", "LAVANDA", "CAMEL", "COBRE", 
        "MARRON", "ROSA PALO", "CHOCOLATE", "PERLA", "TURQUESA", "BOTELLA", 
        "INDIGO", "HIELO", "ORO", "LIMA", "CENIZA", "CHICLE", "PLATA", "VINO", 
        "CUERO", "CARAMELO", "MANDARINA", "TERRACOTA", "CANELA", "FRESA", 
        "GUNMETAL", "ESMERALDA", "PIMENTON", "ASFALTO", "AMARILLO PASTEL", 
        "AZUL NOCHE", "TINTA", "BLEACH", "DARK CAZA", "TABACO", "LILA", "COGNAC", 
        "MUSGO", "SALMON", "PORCELANA", "DIRTY CLARO", "TAUPE", "TEJANO GRIS OSCURO", 
        "NARANJA PASTEL", "GREEN CAST", "ELECTRICO", "BERMELLON", "CAZA", 
        "BILLAR", "DIRTY OSCURO", "MARFIL", "PRUSIA", "CEREZA", "MANZANA", 
        "TEJANO SOFT", "COFFEE", "MISTERIO", "CURRY", "DIRTY", "VISON", "POMELO", 
        "BLOOD", "VIOLETA", "CIRUELA", "GROSELLA", "ROJO VALENTINO", 
        "TEJANO GRIS CLARO", "GERANIO", "AMARILLO FLUOR", "PURPURA", "MORA", 
        "ROSA FLUOR", "TESTA DI MORO", "NARANJA FLUOR", "ALBARICOQUE"
    ]
}

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
MAX_DISPLAY_WIDTH=800
# Functions to initialize tables
def initialize_metadata_table():
    """
    Initialize the metadata table with empty values in the 'Values' column.
    """
    df = pd.read_csv('../data/product_data.csv')
    # Create an empty 'Values' column
    return pd.DataFrame({"Metadata": df.columns, "Values": [None] * len(df.columns)})

def initialize_attributes_table(image, metadata):
    metadf = pd.DataFrame([metadata['Values'].values], columns=metadata['Metadata'].values)
    df = backend.predict_attributes(remove_background(Image.open(image)), metadf)
    return pd.DataFrame({"Attributes": df.columns, "Values": df.iloc[0, :]})

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["uploaded_image"] = None
    st.session_state["metadata"] = None

# Handlers for button actions
def go_to_step(step):
    st.session_state["step"] = step
    
def show_image():
    # Remove the background and display the image
    if "uploaded_image" in st.session_state and st.session_state["uploaded_image"] is not None:
        try:
            # Load the uploaded image
            uploaded_image = Image.open(st.session_state["uploaded_image"])

            # Apply background removal
            processed_image = remove_background(uploaded_image)

            # Dynamically adjust the image size for display
            width, height = processed_image.size
            if width > MAX_DISPLAY_WIDTH:
                # Resize to fit the screen width
                new_height = int(MAX_DISPLAY_WIDTH * height / width)
                processed_image = processed_image.resize((MAX_DISPLAY_WIDTH, new_height), Image.Resampling.LANCZOS)

            # Display the processed image
            st.image(processed_image, use_column_width=False, output_format="auto")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.error("No image uploaded.")

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

    show_image()

    # Initialize the metadata table
    if "metadata" not in st.session_state or st.session_state["metadata"] is None:
        st.session_state["metadata"] = initialize_metadata_table()

    df = st.session_state["metadata"]
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

    # Update session state with the edited metadata table
    st.session_state["metadata"] = pd.DataFrame(response["data"])

    # Validate metadata values
    validation_errors = []
    for _, row in st.session_state["metadata"].iterrows():
        field = row["Metadata"]
        value = row["Values"]

        if field in METADATA_ALLOWED_VALUES and value not in METADATA_ALLOWED_VALUES[field]:
            validation_errors.append(f"Invalid value '{value}' for '{field}'. Allowed: {METADATA_ALLOWED_VALUES[field]}")

    # Display validation errors
    if validation_errors:
        st.error("Some metadata values are invalid. Please correct the following errors:")
        for error in validation_errors:
            st.write(f"- {error}")
    else:
        col1, col2 = st.columns(2)
        col1.button("Previous Step", on_click=go_to_step, args=(1,))
        if col2.button("Next Step"):
            with st.spinner("Processing information..."):
                # Pass null values for empty cells
                metadata = st.session_state["metadata"].copy()
                metadata["Values"] = metadata["Values"].apply(lambda x: None if pd.isna(x) or x == "" else x)

                # Initialize attributes table
                st.session_state["attributes_df"] = initialize_attributes_table(
                    st.session_state["uploaded_image"], metadata
                )

            st.session_state["step"] = 3
            st.experimental_rerun()

# Step 3: Display attributes table
elif st.session_state["step"] == 3:
    import os
    import random
    import string

    st.write("### Step 3: Check the attributes and their values")
    show_image()
    # Load the attributes table from session state
    attributes_df = st.session_state["attributes_df"]
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

    # Update session state with the edited attributes table
    st.session_state["attributes_df"] = pd.DataFrame(response["data"])

    # Add a "Submit" button
    if st.button("Submit"):
        # Create a dictionary from both dataframes
        metadata_dict = dict(zip(st.session_state["metadata"]["Metadata"], st.session_state["metadata"]["Values"]))
        attributes_dict = dict(zip(st.session_state["attributes_df"]["Attributes"], st.session_state["attributes_df"]["Values"]))

        # Combine both dictionaries
        combined_dict = {**metadata_dict, **attributes_dict}

        # Ensure 'cod_modelo_color' is not null and unique in the dataset
        dataset_path = 'imageDB/merged_dataset.csv'
        if os.path.exists(dataset_path):
            existing_df = pd.read_csv(dataset_path)
            existing_cod_modelo_colors = set(existing_df['cod_modelo_color'])
        else:
            existing_cod_modelo_colors = set()

        if not combined_dict['cod_modelo_color']:
            while True:
                new_cod_modelo_color = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                if new_cod_modelo_color not in existing_cod_modelo_colors:
                    combined_dict['cod_modelo_color'] = new_cod_modelo_color
                    break

        # Ensure 'des_filename' has a valid value
        if not combined_dict['des_filename']:
            file_extension = os.path.splitext(st.session_state["uploaded_image"].name)[-1] or '.jpg'
            combined_dict['des_filename'] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=15)) + file_extension

        # Create a single-row dataframe
        result_df = pd.DataFrame([combined_dict])

        # Append the result_df to the CSV file
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        if os.path.exists(dataset_path):
            result_df.to_csv(dataset_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(dataset_path, index=False)

        # Save the image with the name in 'des_filename'
        save_dir = '../../datathon-fme-mango/archive/images/images'
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, combined_dict['des_filename'])
        with open(image_path, "wb") as f:
            f.write(st.session_state["uploaded_image"].getbuffer())

        st.success(f"Data successfully saved. Image saved as '{combined_dict['des_filename']}'.")

    # Navigation buttons
    col1, col2 = st.columns(2)
    col1.button("Previous Step", on_click=go_to_step, args=(2,))
    col2.button("Restart", on_click=go_to_step, args=(1,))