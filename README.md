---
title: Tile Matcher
emoji: 🏷️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.45.0
app_file: hf_app.py
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

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Upload catalog images through the web interface
3. Use the camera function to match tiles in real-time

## Technical Details

The application uses several computer vision techniques:
- Feature extraction (SIFT, ORB)
- Deep learning-based image similarity
- Image preprocessing to handle real-world conditions
- Hybrid matching algorithm for best results

## Project Structure

- `app.py`: Main application entry point
- `tile_matcher/`: Core matching functionality
  - `preprocessing.py`: Image enhancement and preparation
  - `matcher.py`: Matching algorithms
  - `models.py`: Deep learning models
- `static/`: Web assets (CSS, JS, etc.)
- `templates/`: HTML templates
- `catalog/`: Default location for catalog images