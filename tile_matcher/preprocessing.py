"""
Image preprocessing module for tile matching.

This module handles the preprocessing of images to improve matching accuracy:
- Normalization and enhancement
- Shadow and reflection removal
- Background removal
- Image quality improvement
"""

import cv2
import numpy as np
from skimage import exposure, restoration, color


class ImagePreprocessor:
    """Class for preprocessing images to improve matching accuracy."""
    
    def __init__(self):
        """Initialize the image preprocessor."""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess(self, image):
        """
        Apply all preprocessing steps to an image.
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            Preprocessed image
        """
        # Convert to proper format if needed
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Invalid image provided")
            
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Apply individual preprocessing steps
        processed = self.normalize_lighting(processed)
        processed = self.enhance_contrast(processed)
        processed = self.remove_noise(processed)
        
        return processed
    
    def normalize_lighting(self, image):
        """
        Normalize lighting conditions and reduce shadows.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with normalized lighting
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split into channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def enhance_contrast(self, image):
        """
        Enhance image contrast to make features more distinguishable.
        
        Args:
            image: Input BGR image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Perform contrast stretching on V channel
        v_eq = exposure.equalize_hist(v)
        v_eq = (v_eq * 255).astype(np.uint8)
        
        # Merge channels and convert back to BGR
        hsv_eq = cv2.merge([h, s, v_eq])
        enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def remove_noise(self, image):
        """
        Remove noise from the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised
    
    def extract_tile(self, image):
        """
        Attempt to extract the tile from the background.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with isolated tile (if possible)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, threshold = cv2.threshold(blurred, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assumed to be the tile)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # Set background to white
        
        return result
