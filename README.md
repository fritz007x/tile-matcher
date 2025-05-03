

# Tile Matching Application

This is my final project for the course CAI2840C-2253-7384 Introduction to Computer Vision. The application matches photos of tiles taken on-the-fly with a catalog of reference tile images.

## Features

- **Catalog Management**: Organize and manage a collection of reference tile images
- **Advanced Image Processing**: Handles real-world image matching challenges like:
  - Shadows
  - Light reflections
  - Varying backgrounds
  - Lower quality captures
    
- **Intelligent Matching**: Uses computer vision and deep learning to match tiles accurately

## Interface Options

This application comes with a *Streamlit Interface* option (`streamlit_app.py`): 
   - Rich interactive components
   - Good for local development

## Installation

1. Clone this repository
2. Install the requirements:
   ```bash
    # For Streamlit interface
   pip install -r requirements-streamlit.txt
   ```

## Usage
```bash
streamlit run streamlit_app.py
```

## Technical Details

The application uses the following computer vision techniques:
- Feature extraction with ORB 
- Deep learning-based image similarity
- Image preprocessing to handle real-world conditions


## Project Structure

- `streamlit_app.py`: Streamlit web interface 
- `tile_matcher/`: Core functionality modules
  - `matcher.py`: Feature-based tile matching algorithms
  - `preprocessing.py`: Image preprocessing functions
  - `vit_matcher.py`: Vision Transformer-based matching
- `catalog/`: Default location for catalog images
