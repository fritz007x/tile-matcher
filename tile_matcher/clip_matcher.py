"""
CLIP-based tile matching module.

This module provides functionality to match tiles using OpenAI's CLIP model,
which has powerful zero-shot visual recognition capabilities.
"""

import torch
import clip
import os
from PIL import Image
import numpy as np
from pathlib import Path


class CLIPTileMatcher:
    """Class for matching tile images using OpenAI's CLIP model."""
    
    def __init__(self):
        """Initialize the CLIP-based tile matcher."""
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.catalog = {}  # Will store catalog images and their features
    
    def add_catalog_image(self, image_path, image_id=None):
        """
        Add an image to the matching catalog.
        
        Args:
            image_path: Path to the catalog image
            image_id: Optional ID for the image, defaults to filename
            
        Returns:
            image_id of the added catalog item
        """
        # Use filename as ID if not provided
        if image_id is None:
            image_id = Path(image_path).stem
        
        # Load and preprocess the image for CLIP
        try:
            image = Image.open(image_path).convert("RGB")
            preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract image features
            with torch.no_grad():
                image_features = self.model.encode_image(preprocessed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Store the image features and path in the catalog
            self.catalog[image_id] = {
                'features': image_features.cpu().numpy(),
                'path': image_path
            }
            
            return image_id
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")
    
    def load_catalog_directory(self, directory_path):
        """
        Load all images from a directory into the catalog.
        
        Args:
            directory_path: Path to the directory containing catalog images
            
        Returns:
            List of image IDs added to the catalog
        """
        image_ids = []
        
        # Check if directory exists
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Load supported image files
        for ext in ['jpg', 'jpeg', 'png']:
            for file_path in Path(directory_path).glob(f"*.{ext}"):
                try:
                    image_id = self.add_catalog_image(str(file_path))
                    image_ids.append(image_id)
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")
        
        return image_ids
    
    def match(self, query_image, top_n=1):
        """
        Match a query image against the catalog.
        
        Args:
            query_image: Query image (PIL Image, numpy array, or path to image file)
            top_n: Number of top matches to return
            
        Returns:
            List of (image_id, score) tuples for the top matches
        """
        # Load image if path is provided
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert("RGB")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(query_image, np.ndarray):
            query_image = Image.fromarray(query_image.astype('uint8')).convert("RGB")
        
        # Preprocess the query image for CLIP
        preprocessed_image = self.preprocess(query_image).unsqueeze(0).to(self.device)
        
        # Extract query image features
        with torch.no_grad():
            query_features = self.model.encode_image(preprocessed_image)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        
        # Match against catalog
        matches = []
        query_features_np = query_features.cpu().numpy()
        
        for image_id, catalog_item in self.catalog.items():
            # Calculate cosine similarity between features
            similarity = np.dot(query_features_np, catalog_item['features'].T)[0][0]
            matches.append((image_id, float(similarity)))
        
        # Sort by score (descending)
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        
        # Return top N matches
        return matches[:top_n]
    
    def match_with_text(self, query_text, top_n=1):
        """
        Match a text description against the catalog images.
        
        Args:
            query_text: Text description of the tile
            top_n: Number of top matches to return
            
        Returns:
            List of (image_id, score) tuples for the top matches
        """
        # Tokenize and encode text
        text = clip.tokenize([query_text]).to(self.device)
        
        # Extract text features
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Match against catalog
        matches = []
        text_features_np = text_features.cpu().numpy()
        
        for image_id, catalog_item in self.catalog.items():
            # Calculate cosine similarity between features
            similarity = np.dot(text_features_np, catalog_item['features'].T)[0][0]
            matches.append((image_id, float(similarity)))
        
        # Sort by score (descending)
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        
        # Return top N matches
        return matches[:top_n]
