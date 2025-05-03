"""
Tile Matching Application

A web application that matches on-the-fly photos of tiles with a catalog
of reference tile images.
"""

import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tile_matcher.preprocessing import ImagePreprocessor
from tile_matcher.matcher import TileMatcher
from tile_matcher.clip_matcher import CLIPTileMatcher
from tile_matcher.vit_matcher import ViTTileMatcher

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
CATALOG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalog')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'tile_model.h5')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CATALOG_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'), exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CATALOG_FOLDER'] = CATALOG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.secret_key = 'tile-matching-secret-key'

# Initialize tile matching components
preprocessor = ImagePreprocessor()

# Select matching method (MATCHER_TYPE can be "vit", "clip", or "feature")
MATCHER_TYPE = "vit"  # Change this to use different matcher types

# Initialize the appropriate matcher
matcher = None
using_clip = False
using_vit = False
using_feature = False

if MATCHER_TYPE == "vit":
    print("Initializing Vision Transformer model for tile matching...")
    try:
        matcher = ViTTileMatcher()
        using_vit = True
        print("Vision Transformer model initialized successfully!")
    except Exception as e:
        print(f"Error initializing Vision Transformer model: {e}")
        print("Falling back to CLIP model...")
        MATCHER_TYPE = "clip"

if MATCHER_TYPE == "clip":
    print("Initializing CLIP model for tile matching...")
    try:
        matcher = CLIPTileMatcher()
        using_clip = True
        print("CLIP model initialized successfully!")
    except Exception as e:
        print(f"Error initializing CLIP model: {e}")
        print("Falling back to feature-based matching...")
        MATCHER_TYPE = "feature"

if MATCHER_TYPE == "feature":
    print("Using feature-based (ORB) matcher...")
    matcher = TileMatcher(method='feature')
    using_feature = True

# Load catalog images
try:
    matcher.load_catalog_directory(app.config['CATALOG_FOLDER'])
    print(f"Loaded {len(matcher.catalog)} images into catalog")
except Exception as e:
    print(f"Note: Catalog could not be loaded: {e}")


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    catalog_images = []
    
    for image_id, item in matcher.catalog.items():
        catalog_images.append({
            'id': image_id,
            'path': os.path.relpath(item['path'], os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'),
            'name': os.path.basename(item['path'])
        })

    # Pass matching method info to the template    
    return render_template('index.html', 
                          catalog_images=catalog_images,
                          using_clip=using_clip,
                          using_vit=using_vit,
                          using_feature=using_feature)


@app.route('/upload_catalog', methods=['POST'])
def upload_catalog():
    """Handle catalog image uploads."""
    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['CATALOG_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Add to matcher catalog
                matcher.add_catalog_image(file_path)
                uploaded_count += 1
            except Exception as e:
                flash(f"Error processing {filename}: {e}")
    
    if uploaded_count > 0:
        flash(f"Successfully uploaded {uploaded_count} catalog images")
    
    return redirect(url_for('index'))


@app.route('/match', methods=['POST'])
def match_tile():
    """Match an uploaded tile image against the catalog."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        file_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({'error': 'Could not read uploaded image'}), 400
            
            # Preprocess the image
            processed_image = preprocessor.preprocess(image)
            
            # Save the processed image
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_processed.{ext}")
            cv2.imwrite(processed_path, processed_image)
            
            # Match against catalog
            matches = matcher.match(processed_image, top_n=5)
            
            # Prepare results
            results = []
            for image_id, score in matches:
                if image_id in matcher.catalog:
                    catalog_item = matcher.catalog[image_id]
                    results.append({
                        'id': image_id,
                        'score': float(score),
                        'path': os.path.relpath(catalog_item['path'], 
                                              os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'),
                        'name': os.path.basename(catalog_item['path'])
                    })
            
            return jsonify({
                'success': True,
                'original': os.path.join('uploads', filename).replace('\\', '/'),
                'processed': os.path.join('uploads', f"{file_id}_processed.{ext}").replace('\\', '/'),
                'matches': results,
                'using_clip': using_clip,
                'using_vit': using_vit,
                'using_feature': using_feature
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/match_text', methods=['POST'])
def match_text():
    """Match a text description against the catalog images."""
    if not using_clip:
        return jsonify({'error': 'Text matching is only available with CLIP model'}), 400
    
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Match text against catalog
        matches = matcher.match_with_text(text, top_n=5)
        
        # Prepare results
        results = []
        for image_id, score in matches:
            if image_id in matcher.catalog:
                catalog_item = matcher.catalog[image_id]
                results.append({
                    'id': image_id,
                    'score': float(score),
                    'path': os.path.relpath(catalog_item['path'], 
                                          os.path.dirname(os.path.abspath(__file__))).replace('\\', '/'),
                    'name': os.path.basename(catalog_item['path'])
                })
        
        return jsonify({
            'success': True,
            'query': text,
            'matches': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/remove_catalog/<image_id>')
def remove_catalog(image_id):
    """Remove an image from the catalog."""
    if image_id in matcher.catalog:
        file_path = matcher.catalog[image_id]['path']
        
        # Remove from matcher
        del matcher.catalog[image_id]
        
        # Delete file (optional)
        try:
            os.remove(file_path)
            flash(f"Removed {os.path.basename(file_path)} from catalog")
        except Exception as e:
            flash(f"Removed from catalog but couldn't delete file: {e}")
    else:
        flash("Image not found in catalog")
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    matcher_description = "Unknown"
    if using_vit:
        matcher_description = "Vision Transformer (ViT)"
    elif using_clip:
        matcher_description = "CLIP"
    elif using_feature:
        matcher_description = "Feature-based (ORB)"
        
    print(f"Starting Tile Matcher application with {matcher_description} matching...")
    print(f"Catalog directory: {app.config['CATALOG_FOLDER']}")
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True)
