"""
Deep learning models for tile matching.

This module provides functionality to create and train models for feature extraction
and similarity matching of tile images.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import os
import cv2
from pathlib import Path


class TileMatchingModel:
    """Class for creating and training deep learning models for tile matching."""
    
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=128):
        """
        Initialize the tile matching model.
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            embedding_dim: Dimension of the embedding vector
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_model(self, base_model='mobilenet'):
        """
        Build a deep learning model for feature extraction.
        
        Args:
            base_model: Base model architecture to use ('mobilenet', 'resnet50', 'efficientnet')
            
        Returns:
            Compiled model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Base model (feature extractor)
        if base_model == 'mobilenet':
            base = applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_model == 'resnet50':
            base = applications.ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_model == 'efficientnet':
            base = applications.EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Freeze the base model layers
        base.trainable = False
        
        # Connect the base model to the input
        x = base(inputs, training=False)
        
        # Add pooling and projection layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.embedding_dim)(x)
        
        # Normalize embeddings to have unit length
        outputs = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='embeddings'
        )(x)
        
        # Create the model
        model = models.Model(inputs, outputs)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=self.triplet_loss
        )
        
        self.model = model
        return model
    
    def triplet_loss(self, y_true, y_pred, margin=0.2):
        """
        Triplet loss for training the model.
        
        Args:
            y_true: True labels (not used in triplet loss)
            y_pred: Predicted embeddings
            margin: Margin for triplet loss
            
        Returns:
            Loss value
        """
        # Extract anchor, positive, and negative embeddings
        embeddings = y_pred
        anchor, positive, negative = tf.split(embeddings, 3, axis=0)
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate triplet loss
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)
    
    def prepare_triplets(self, data_dir):
        """
        Prepare triplets for training from a directory of class-organized images.
        
        Args:
            data_dir: Directory containing subdirectories of images, where each subdirectory
                    represents a different tile class
                    
        Returns:
            List of triplets (anchor, positive, negative) for training
        """
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes to form triplets")
        
        # Dictionary to store images by class
        class_images = {}
        
        # Load images for each class
        for cls in classes:
            class_dir = os.path.join(data_dir, cls)
            images = []
            
            for ext in ['jpg', 'jpeg', 'png']:
                for file_path in Path(class_dir).glob(f"*.{ext}"):
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        img = cv2.resize(img, self.input_shape[:2])
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
            
            if len(images) >= 2:  # Need at least 2 images per class for anchor and positive
                class_images[cls] = images
        
        # Form triplets
        triplets = []
        viable_classes = list(class_images.keys())
        
        if len(viable_classes) < 2:
            raise ValueError("Not enough viable classes with multiple images")
        
        for anchor_class in viable_classes:
            # Get anchor and positive from the same class
            anchor_images = class_images[anchor_class]
            
            for i, anchor in enumerate(anchor_images):
                # Positive: different image from the same class
                for j, positive in enumerate(anchor_images):
                    if i != j:  # Skip the same image
                        # Negative: image from a different class
                        for negative_class in viable_classes:
                            if negative_class != anchor_class:
                                negative_images = class_images[negative_class]
                                for negative in negative_images:
                                    triplets.append((anchor, positive, negative))
        
        return triplets
    
    def train(self, data_dir, epochs=10, batch_size=32):
        """
        Train the model using triplets.
        
        Args:
            data_dir: Directory containing subdirectories of images, where each subdirectory
                    represents a different tile class
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Prepare triplets
        triplets = self.prepare_triplets(data_dir)
        
        if not triplets:
            raise ValueError("No triplets generated for training")
        
        # Convert triplets to NumPy arrays
        anchors = np.array([t[0] for t in triplets])
        positives = np.array([t[1] for t in triplets])
        negatives = np.array([t[2] for t in triplets])
        
        # Concatenate for batch processing
        x_train = np.concatenate([anchors, positives, negatives], axis=0)
        
        # Dummy labels (not used in triplet loss)
        y_train = np.zeros((len(x_train), self.embedding_dim))
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(model_path)
        
    def load_model(self, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'triplet_loss': self.triplet_loss}
        )
        
    def extract_features(self, image):
        """
        Extract features from an image using the model.
        
        Args:
            image: Input image (NumPy array)
            
        Returns:
            Feature vector
        """
        if self.model is None:
            raise ValueError("No model available for feature extraction")
            
        # Preprocess the image
        img = cv2.resize(image, self.input_shape[:2])
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = self.model.predict(img)
        
        return features[0]
