"""
Entry point for Hugging Face Spaces deployment
"""

import os
import sys
import tempfile

# Set environment variables for Streamlit and temporary directories
os.environ['STREAMLIT_SERVER_PORT'] = "7860"  # Default port for HF Spaces
os.environ['STREAMLIT_SERVER_HEADLESS'] = "true"
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = "none"

# Create necessary directories with explicit permissions
TEMP_DIR = tempfile.gettempdir()
CACHE_DIR = os.path.join(TEMP_DIR, "st_cache")
UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")
CATALOG_DIR = os.path.join(os.getcwd(), "catalog")

for directory in [CACHE_DIR, UPLOAD_DIR, CATALOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set application config
os.environ['UPLOAD_FOLDER'] = UPLOAD_DIR
os.environ['CATALOG_FOLDER'] = CATALOG_DIR

# Import after setting environment variables
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config

_config.set_option("server.port", 7860)
_config.set_option("server.headless", True)
_config.set_option("server.enableCORS", False)
_config.set_option("server.enableXsrfProtection", False)
_config.set_option("browser.gatherUsageStats", False)
_config.set_option("runner.magicEnabled", False)

# Run the actual Streamlit app
if __name__ == "__main__":
    bootstrap.run("streamlit_app.py", "", [], [])
