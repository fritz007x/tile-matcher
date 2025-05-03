"""
Tile Matching Application - Gradio Interface

A web application that matches on-the-fly photos of tiles with a catalog
of reference tile images.
"""

import os
import shutil
import uuid
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import gradio as gr

from tile_matcher.preprocessing import ImagePreprocessor
from tile_matcher.matcher import TileMatcher

# Try to import the ViT matcher if available
try:
    from tile_matcher.vit_matcher import ViTTileMatcher
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Vision Transformer (ViT) is not available. Using feature-based matching only.")

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, 'static', 'uploads')
CATALOG_FOLDER = os.path.join(APP_DIR, 'catalog')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)

# Global instances of matchers and preprocessor
preprocessor = ImagePreprocessor()
feature_matcher = TileMatcher(method='feature')
vit_matcher = None

if VIT_AVAILABLE:
    try:
        vit_matcher = ViTTileMatcher()
        print("Vision Transformer matcher initialized successfully")
    except Exception as e:
        print(f"Could not initialize Vision Transformer: {e}")

# Load initial catalog
def load_catalog(matcher, catalog_path):
    try:
        image_ids = matcher.load_catalog_directory(catalog_path)
        print(f"Loaded {len(matcher.catalog)} images into catalog")
        return len(matcher.catalog)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return 0

# Initialize catalogs
load_catalog(feature_matcher, CATALOG_FOLDER)
if vit_matcher:
    load_catalog(vit_matcher, CATALOG_FOLDER)

# Helper function to convert between RGB and BGR
def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Match functionality
def match_tile(image, matcher_type="feature"):
    if image is None:
        return None, None, [], "Please upload an image"
    
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = rgb_to_bgr(img_array)
    
    # Preprocess
    processed_image = preprocessor.preprocess(img_cv)
    
    # Select matcher
    if matcher_type == "vit" and vit_matcher:
        matcher = vit_matcher
        matcher_name = "Vision Transformer"
    else:
        matcher = feature_matcher
        matcher_name = "Feature-based (ORB)"
    
    # Match against catalog
    matches = matcher.match(processed_image, top_n=5)
    
    # Prepare results
    result_images = []
    for image_id, score in matches:
        if image_id in matcher.catalog:
            img_path = matcher.catalog[image_id]['path']
            result_images.append((img_path, score))
    
    status = f"Found {len(result_images)} matches using {matcher_name}"
    
    # Return original, processed, and match results
    return bgr_to_rgb(img_cv), bgr_to_rgb(processed_image), result_images, status

# Add image to catalog
def add_to_catalog(image, filename=""):
    if image is None:
        return [], "No image provided"
    
    # Generate a filename if not provided
    if not filename.strip():
        filename = f"tile_{uuid.uuid4().hex[:8]}.jpg"
    
    # Ensure filename has extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        filename += ".jpg"
    
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = rgb_to_bgr(img_array)
    
    # Save to catalog
    file_path = os.path.join(CATALOG_FOLDER, filename)
    cv2.imwrite(file_path, img_cv)
    
    # Add to matchers
    feature_matcher.add_catalog_image(file_path)
    if vit_matcher:
        vit_matcher.add_catalog_image(file_path)
    
    # Get updated catalog
    catalog_images = get_catalog_images()
    
    return catalog_images, f"Added '{filename}' to catalog"

# Remove image from catalog
def remove_from_catalog(image_path):
    if not image_path or not os.path.exists(image_path):
        return get_catalog_images(), "Image not found"
    
    # Get image ID from path
    image_id = Path(image_path).stem
    
    # Remove from matchers
    if image_id in feature_matcher.catalog:
        del feature_matcher.catalog[image_id]
    
    if vit_matcher and image_id in vit_matcher.catalog:
        del vit_matcher.catalog[image_id]
    
    # Delete file
    try:
        os.remove(image_path)
        status = f"Removed {os.path.basename(image_path)} from catalog"
    except Exception as e:
        status = f"Error removing file: {e}"
    
    # Get updated catalog
    catalog_images = get_catalog_images()
    
    return catalog_images, status

# Get all images in the catalog
def get_catalog_images():
    catalog_images = []
    
    for ext in ALLOWED_EXTENSIONS:
        for file_path in Path(CATALOG_FOLDER).glob(f"*.{ext}"):
            catalog_images.append(str(file_path))
    
    return catalog_images

# Format gallery output for matches
def format_matches(original, processed, matches, status):
    result = [(original, "Original")]
    result.append((processed, "Processed"))
    
    for img_path, score in matches:
        img = Image.open(img_path)
        result.append((np.array(img), f"{os.path.basename(img_path)}\nScore: {score:.2f}"))
    
    return result, status

# Build the Gradio interface
with gr.Blocks(title="Tile Matcher") as demo:
    gr.Markdown("# 🏷️ Tile Matcher")
    gr.Markdown("Match photos of tiles with a catalog of reference images")
    
    with gr.Tabs():
        with gr.TabItem("Match Tiles"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload tile to match")
                    
                    matcher_choice = gr.Radio(
                        ["feature", "vit"] if VIT_AVAILABLE else ["feature"],
                        label="Matching Method",
                        value="feature",
                        info="Select matching algorithm"
                    )
                    
                    match_button = gr.Button("Match", variant="primary")
                
                with gr.Column(scale=2):
                    match_gallery = gr.Gallery(
                        label="Results", 
                        columns=3, 
                        rows=2,
                        object_fit="contain",
                        height="500px"
                    )
                    match_status = gr.Textbox(label="Status")
            
            match_button.click(
                fn=match_tile,
                inputs=[input_image, matcher_choice],
                outputs=[input_image, input_image, match_gallery, match_status],
                postprocess=format_matches
            )
        
        with gr.TabItem("Manage Catalog"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_image = gr.Image(label="Upload image to catalog")
                    filename_input = gr.Textbox(
                        label="Filename (optional)", 
                        placeholder="Leave blank for auto-generated name"
                    )
                    add_button = gr.Button("Add to Catalog", variant="primary")
                    
                    remove_button = gr.Button("Remove Selected", variant="secondary")
                    catalog_status = gr.Textbox(label="Status")
                
                with gr.Column(scale=2):
                    catalog_gallery = gr.Gallery(
                        label="Catalog Images",
                        columns=4,
                        rows=None,
                        object_fit="contain",
                        height="600px",
                        allow_preview=True,
                        selected_index=gr.State(None)
                    ).style(grid=4)
            
            # Load initial catalog
            catalog_gallery.value = get_catalog_images()
            
            # Add to catalog
            add_button.click(
                fn=add_to_catalog,
                inputs=[upload_image, filename_input],
                outputs=[catalog_gallery, catalog_status]
            )
            
            # Handle selection and removal
            selected_image = gr.State(None)
            
            def select_image(evt: gr.SelectData, gallery):
                if evt.index < len(gallery):
                    return gallery[evt.index]
                return None
            
            catalog_gallery.select(
                fn=select_image,
                inputs=[catalog_gallery],
                outputs=[selected_image]
            )
            
            remove_button.click(
                fn=remove_from_catalog,
                inputs=[selected_image],
                outputs=[catalog_gallery, catalog_status]
            )

# Launch the app
if __name__ == "__main__":
    demo.launch()
