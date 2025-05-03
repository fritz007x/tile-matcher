"""
Tile matching module.

This module implements algorithms for matching tile images:
- Feature-based matching using ORB
- Color histogram matching
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path


class TileMatcher:
    """Class for matching tile images with a catalog."""
    
    def __init__(self, method='feature', model_path=None):
        """
        Initialize the tile matcher.
        
        Args:
            method: Matching method ('feature', 'color', or 'hybrid')
            model_path: Not used, kept for backward compatibility
        """
        valid_methods = ['feature', 'color', 'hybrid']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        self.method = method
        self.catalog = {}  # Will store catalog images and their features
        
        # Initialize feature extractor - using ORB instead of SIFT
        self.feature_extractor = cv2.ORB_create(nfeatures=1000)
        
        # For feature matching with ORB
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Deep learning is not supported in this version
        self.model = None

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
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Extract and store features based on the matching method
        features = {}
        
        if self.method in ['feature', 'hybrid']:
            # Extract ORB keypoints and descriptors
            keypoints, descriptors = self.feature_extractor.detectAndCompute(image, None)
            features['keypoints'] = keypoints
            features['descriptors'] = descriptors
        
        if self.method in ['color', 'hybrid']:
            # Calculate color histograms for each channel
            features['color_hist'] = self.calculate_color_histogram(image)
        
        # Store the image and its features in the catalog
        self.catalog[image_id] = {
            'image': image,
            'features': features,
            'path': image_path
        }
        
        return image_id
    
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
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
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
            query_image: Query image (numpy array or path to image file)
            top_n: Number of top matches to return
            
        Returns:
            List of (image_id, score) tuples for the top matches
        """
        # Load image if path is provided
        if isinstance(query_image, str):
            query_image = cv2.imread(query_image)
        
        if query_image is None:
            raise ValueError("Invalid query image")
        
        # Get results based on the chosen method
        if self.method == 'feature':
            matches = self.feature_based_matching(query_image)
        elif self.method == 'color':
            matches = self.color_based_matching(query_image)
        elif self.method == 'hybrid':
            matches = self.hybrid_matching(query_image)
        else:
            # Fallback to feature matching
            matches = self.feature_based_matching(query_image)
        
        # Return top N matches
        return matches[:top_n]
    
    def feature_based_matching(self, query_image):
        """
        Perform feature-based matching using ORB features.
        
        Args:
            query_image: Query image
            
        Returns:
            List of (image_id, score) tuples sorted by score (descending)
        """
        # Extract features from the query image
        keypoints, descriptors = self.feature_extractor.detectAndCompute(query_image, None)
        
        if descriptors is None or len(descriptors) == 0:
            return []
        
        matches = []
        
        for image_id, catalog_item in self.catalog.items():
            if 'descriptors' not in catalog_item['features'] or catalog_item['features']['descriptors'] is None:
                continue
                
            catalog_descriptors = catalog_item['features']['descriptors']
            
            # Skip if no descriptors
            if len(catalog_descriptors) == 0:
                continue
                
            # Match descriptors using BFMatcher with k=2 for ratio test
            try:
                matches_knn = self.bf_matcher.knnMatch(descriptors, catalog_descriptors, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches_knn:
                    if len(match_pair) == 2:  # Ensure we have 2 matches for the ratio test
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1:  # Some matches might only have one result
                        good_matches.append(match_pair[0])
                
                # Score is the number of good matches relative to the total possible matches
                score = len(good_matches) / max(len(keypoints), 
                                              len(catalog_item['features']['keypoints']))
                
                matches.append((image_id, score))
            except Exception as e:
                print(f"Error matching {image_id}: {e}")
        
        # Sort by score (descending)
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def calculate_color_histogram(self, image):
        """
        Calculate color histograms for each channel.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of histograms for each channel
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        return {
            'h': h_hist,
            's': s_hist,
            'v': v_hist
        }
    
    def color_based_matching(self, query_image):
        """
        Perform color-based matching using color histograms.
        
        Args:
            query_image: Query image
            
        Returns:
            List of (image_id, score) tuples sorted by score (descending)
        """
        # Calculate histogram for query image
        query_hist = self.calculate_color_histogram(query_image)
        
        matches = []
        
        for image_id, catalog_item in self.catalog.items():
            if 'color_hist' not in catalog_item['features']:
                continue
                
            # Compare histograms
            h_corr = cv2.compareHist(query_hist['h'], 
                                     catalog_item['features']['color_hist']['h'], 
                                     cv2.HISTCMP_CORREL)
            s_corr = cv2.compareHist(query_hist['s'], 
                                     catalog_item['features']['color_hist']['s'], 
                                     cv2.HISTCMP_CORREL)
            v_corr = cv2.compareHist(query_hist['v'], 
                                     catalog_item['features']['color_hist']['v'], 
                                     cv2.HISTCMP_CORREL)
            
            # Weight and combine the correlations
            score = 0.5 * h_corr + 0.25 * s_corr + 0.25 * v_corr
            
            matches.append((image_id, score))
        
        # Sort by score (descending)
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def extract_deep_features(self, image):
        """
        Extract deep learning features from an image.
        
        This method is kept for backward compatibility but always returns None
        since deep learning is not supported in this version.
        
        Args:
            image: Input BGR image
            
        Returns:
            None
        """
        return None
    
    def deep_learning_matching(self, query_image):
        """
        Perform deep learning-based matching.
        
        This method is kept for backward compatibility but always returns an empty list
        since deep learning is not supported in this version.
        
        Args:
            query_image: Query image
            
        Returns:
            Empty list
        """
        return []
    
    def hybrid_matching(self, query_image):
        """
        Perform hybrid matching using a combination of methods.
        
        Args:
            query_image: Query image
            
        Returns:
            List of (image_id, score) tuples sorted by score (descending)
        """
        # Get results from all methods
        feature_matches = dict(self.feature_based_matching(query_image))
        color_matches = dict(self.color_based_matching(query_image))
        
        # Combine the results
        all_image_ids = set(feature_matches.keys()) | set(color_matches.keys())
        
        combined_matches = []
        
        for image_id in all_image_ids:
            # Get scores from each method, default to 0 if not present
            feature_score = feature_matches.get(image_id, 0)
            color_score = color_matches.get(image_id, 0)
            
            # Weight and combine the scores - reducing color weight
            score = 0.8 * feature_score + 0.2 * color_score  # Reduced color weight
            
            combined_matches.append((image_id, score))
        
        # Sort by score (descending)
        return sorted(combined_matches, key=lambda x: x[1], reverse=True)
