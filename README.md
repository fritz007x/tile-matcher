---
title: Tile Matcher
emoji: 🏷️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.0
app_file: gradio_app.py
pinned: false
---

# Tile Matching Application

This application matches photos of tiles taken on-the-fly with a catalog of reference tile images.

## Features

- **Catalog Management**: Organize and manage a collection of reference tile images
- **Image Capture**: Take photos using a camera interface
- **Advanced Image Processing**: Handles real-world photography challenges like:
  - Shadows
  - Light reflections
  - Varying backgrounds
  - Lower quality captures
- **Intelligent Matching**: Uses computer vision and deep learning to match tiles accurately

## Interface Options

This application comes with two interface options:

1. **Gradio Interface** (`gradio_app.py`): 
   - Optimized for Hugging Face Spaces deployment
   - Simpler runtime model with better performance
   - Improved compatibility with transformer models

2. **Streamlit Interface** (`streamlit_app.py`): 
   - Comprehensive data app experience
   - Rich interactive components
   - Good for local development

## Installation

1. Clone this repository
2. Install the requirements:
   ```bash
   # For Gradio interface (recommended for Hugging Face deployment)
   pip install -r requirements-gradio.txt
   
   # For Streamlit interface
   pip install -r requirements-streamlit.txt
   ```

## Usage

### Gradio Interface (Default)
```bash
python gradio_app.py
```

### Streamlit Interface
```bash
streamlit run streamlit_app.py
```

## Technical Details

The application uses several computer vision techniques:
- Feature extraction (SIFT, ORB)
- Deep learning-based image similarity
- Image preprocessing to handle real-world conditions
- Hybrid matching algorithm for best results

## Project Structure

- `gradio_app.py`: Gradio web interface (optimized for Hugging Face Spaces)
- `streamlit_app.py`: Streamlit web interface (alternative option)
- `tile_matcher/`: Core functionality modules
  - `matcher.py`: Feature-based tile matching algorithms
  - `preprocessing.py`: Image preprocessing functions
  - `vit_matcher.py`: Vision Transformer-based matching
- `catalog/`: Default location for catalog images
