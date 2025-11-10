# models/face_model.py
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from config import MODEL_PATH
import numpy as np


def load_face_model():
    face_model = load_model(MODEL_PATH, compile=False)
    face_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return face_model

def get_feature_extractor(face_model):
    return Model(inputs=face_model.input, outputs=face_model.get_layer("embedding_layer").output)

face_model = load_face_model()
feature_extractor = get_feature_extractor(face_model)

def extract_features(processed_face):
    """
    Extract feature embeddings from a face image using a pre-trained feature extractor.
    
    Args:
        face_image: Input face image (numpy array).
        feature_extractor: Pre-trained model for feature extraction.
        
    Returns:
        embedding: Normalized 1D feature embedding (numpy array), or None if extraction fails.
    """
    try:
        # Step 1: Add batch dimension
        processed_face = np.expand_dims(processed_face, axis=0)  # Shape: (1, 64, 64, 3)

        # Step 2: Extract features using the feature extractor model
        embedding = feature_extractor.predict(processed_face)
        print(f"Embedding shape: {embedding.shape}")  # Debug print

        # Step 3: Flatten the embedding to a 1D array
        embedding = embedding.flatten()

        # Step 4: Normalize the embedding to unit length
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None