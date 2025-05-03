"""
Tile Matching Application

A Streamlit web application that matches on-the-fly photos of tiles with a catalog
of reference tile images.
"""

import os
import uuid
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import glob
import shutil

from tile_matcher.preprocessing import ImagePreprocessor
from tile_matcher.matcher import TileMatcher

# Try to import the ViT matcher if available
try:
    from tile_matcher.vit_matcher import ViTTileMatcher
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    st.warning("Vision Transformer (ViT) is not available. Using feature-based matching only.")

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, 'static', 'uploads')
CATALOG_FOLDER = os.path.join(APP_DIR, 'catalog')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Tile Matcher",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize preprocessing
@st.cache_resource
def get_preprocessor():
    return ImagePreprocessor()

# Initialize matchers
@st.cache_resource
def get_matcher(matcher_type):
    if matcher_type == "vit" and VIT_AVAILABLE:
        try:
            return ViTTileMatcher(), "vit"
        except Exception as e:
            st.warning(f"Could not initialize Vision Transformer: {e}. Falling back to feature-based matching.")
            matcher_type = "feature"
    
    return TileMatcher(method='feature'), "feature"

# Load catalog images
@st.cache_data
def load_catalog(_matcher, catalog_path):
    try:
        image_ids = _matcher.load_catalog_directory(catalog_path)
        return len(_matcher.catalog)
    except Exception as e:
        st.error(f"Error loading catalog: {e}")
        return 0

# Preprocess and match image
def process_and_match(image_file, _matcher, _preprocessor):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(image_file.getbuffer())
        tmp_path = tmp_file.name
    
    # Read and preprocess the image
    image = cv2.imread(tmp_path)
    if image is None:
        st.error("Could not read the uploaded image.")
        os.unlink(tmp_path)
        return None, None, None
    
    # Preprocess the image
    processed_image = _preprocessor.preprocess(image)
    
    # Save processed image
    processed_path = tmp_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(processed_path, processed_image)
    
    # Match against catalog
    matches = _matcher.match(processed_image, top_n=5)
    
    return image, processed_image, matches

# Convert OpenCV BGR to RGB for display
def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show a catalog item
def show_catalog_item(item_path, item_name, col):
    image = Image.open(item_path)
    col.image(image, caption=item_name, use_container_width=True)

# Main application
def main():
    st.title("Tile Matcher")
    st.write("Match photos of tiles with a catalog of reference images")
    
    # Initialize components
    preprocessor = get_preprocessor()
    
    # Sidebar for settings and actions
    st.sidebar.title("Settings")
    
    # Matcher selection
    matcher_options = ["Vision Transformer (ViT)", "Feature-based (ORB)"]
    matcher_types = ["vit", "feature"]
    
    # Filter unavailable options
    if not VIT_AVAILABLE:
        matcher_options = ["Feature-based (ORB)"]
        matcher_types = ["feature"]
        selected_matcher_type = "feature"
    else:
        # Allow matcher selection when ViT is available
        matcher_index = st.sidebar.selectbox(
            "Select matching method",
            range(len(matcher_options)),
            format_func=lambda i: matcher_options[i]
        )
        selected_matcher_type = matcher_types[matcher_index]
    
    matcher, actual_matcher_type = get_matcher(selected_matcher_type)
    
    if actual_matcher_type != selected_matcher_type:
        st.sidebar.warning(f"Using {actual_matcher_type} matcher instead.")
    
    # Load catalog
    catalog_size = load_catalog(matcher, CATALOG_FOLDER)
    st.sidebar.info(f"Catalog contains {catalog_size} images")
    
    # Tabs for different functions
    tab1, tab2 = st.tabs(["Match Tiles", "Manage Catalog"])
    
    # Tab 1: Match Tiles
    with tab1:
        st.header("Upload a tile image to match")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process and match the image
            original, processed, matches = process_and_match(uploaded_file, matcher, preprocessor)
            
            if original is not None and processed is not None:
                # Display original and processed images
                col1, col2 = st.columns(2)
                col1.image(bgr_to_rgb(original), caption="Original Image", use_container_width=True)
                col2.image(bgr_to_rgb(processed), caption="Processed Image", use_container_width=True)
                
                # Display matches
                if matches and len(matches) > 0:
                    st.subheader("Best Matches")
                    st.write(f"Using matcher: {actual_matcher_type.upper()}")
                    
                    match_cols = st.columns(min(5, len(matches)))
                    
                    for i, (image_id, score) in enumerate(matches):
                        if i < len(match_cols) and image_id in matcher.catalog:
                            catalog_item = matcher.catalog[image_id]
                            col = match_cols[i]
                            col.image(Image.open(catalog_item['path']), 
                                     caption=f"{os.path.basename(catalog_item['path'])}\nScore: {score:.2f}", 
                                     use_container_width=True)
                else:
                    st.warning("No matches found in the catalog.")
    
    # Tab 2: Manage Catalog
    with tab2:
        st.header("Manage Catalog")
        
        # Upload to catalog
        st.subheader("Add images to catalog")
        catalog_files = st.file_uploader("Upload tile images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if catalog_files and st.button("Add to Catalog"):
            added_count = 0
            for file in catalog_files:
                # Save the file to the catalog directory
                filename = file.name
                file_path = os.path.join(CATALOG_FOLDER, filename)
                
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                try:
                    # Add to matcher catalog
                    matcher.add_catalog_image(file_path)
                    added_count += 1
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
            
            if added_count > 0:
                st.success(f"Successfully added {added_count} images to the catalog")
                st.rerun()  # Refresh the app
        
        # Display catalog images
        st.subheader("Current Catalog")
        
        if catalog_size > 0:
            # Get all catalog images
            catalog_items = []
            for image_id, item in matcher.catalog.items():
                catalog_items.append({
                    'id': image_id,
                    'path': item['path'],
                    'name': os.path.basename(item['path'])
                })
            
            # Display in a grid
            cols_per_row = 5
            for i in range(0, len(catalog_items), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(catalog_items):
                        item = catalog_items[idx]
                        cols[j].image(Image.open(item['path']), caption=item['name'], use_container_width=True)
                        if cols[j].button(f"Remove", key=f"remove_{item['id']}"):
                            # Remove from matcher
                            if item['id'] in matcher.catalog:
                                del matcher.catalog[item['id']]
                                
                                # Delete file
                                try:
                                    os.remove(item['path'])
                                    st.success(f"Removed {item['name']} from catalog")
                                    st.rerun()  # Refresh the app
                                except Exception as e:
                                    st.error(f"Error removing file: {e}")
        else:
            st.info("Catalog is empty. Upload some images to get started.")

if __name__ == "__main__":
    main()
