"""
Vision Transformer-based tile matching using Hugging Face models.

This module provides functionality to match tiles using Vision Transformer 
models from Hugging Face, offering state-of-the-art image understanding.
"""

import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import os
import numpy as np
from pathlib import Path
import cv2


class ViTTileMatcher:
    """Class for matching tile images using Vision Transformer models."""
    
    def __init__(self, model_name="google/vit-base-patch16-224"):
        """
        Initialize the Vision Transformer tile matcher.
        
        Args:
            model_name: Name of the Vision Transformer model to use
        """
        # Load Vision Transformer model from Hugging Face
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
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
        
        # Load and process the image
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Extract image features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the CLS token as the image representation
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
                # Normalize features
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            # Store the image features and path in the catalog
            self.catalog[image_id] = {
                'features': features,
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
            # Convert BGR to RGB if from OpenCV
            if query_image.shape[2] == 3:
                query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            query_image = Image.fromarray(query_image.astype('uint8'))
        
        # Process the query image
        inputs = self.feature_extractor(images=query_image, return_tensors="pt").to(self.device)
        
        # Extract query image features
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_features = outputs.last_hidden_state[:, 0].cpu().numpy()
            query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        
        # Match against catalog
        matches = []
        
        for image_id, catalog_item in self.catalog.items():
            # Calculate cosine similarity between features
            similarity = np.dot(query_features, catalog_item['features'].T)[0][0]
            matches.append((image_id, float(similarity)))
        
        # Sort by score (descending)
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        
        # Return top N matches
        return matches[:top_n]
