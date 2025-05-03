"""
Entry point for Hugging Face Spaces deployment
"""

import os
import sys

# Set environment variables for Streamlit and temporary directories
os.environ['STREAMLIT_SERVER_PORT'] = "7860"  # Default port for HF Spaces
os.environ['STREAMLIT_SERVER_HEADLESS'] = "true"
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = "none"

# Create necessary directories with explicit permissions
import tempfile
TEMP_DIR = tempfile.gettempdir()
CACHE_DIR = os.path.join(TEMP_DIR, "st_cache")
UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")
CATALOG_DIR = os.path.join(os.getcwd(), "catalog")

for directory in [CACHE_DIR, UPLOAD_DIR, CATALOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set application config
os.environ['UPLOAD_FOLDER'] = UPLOAD_DIR
os.environ['CATALOG_FOLDER'] = CATALOG_DIR

# Directly import and run the streamlit app
import streamlit_app
