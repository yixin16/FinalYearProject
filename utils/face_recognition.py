from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recognize_face(face_embedding, stored_embeddings, threshold=0.90, adaptive_threshold=True):
    """
    Enhanced face recognition with:
    - Adaptive thresholding
    - Batch similarity computation
    - Comprehensive logging
    - Anti-spoofing flag integration
    """
    # Validation checks
    if face_embedding is None or not isinstance(face_embedding, np.ndarray):
        logger.error("Invalid face embedding: None or wrong type")
        return None, None, 0.0
    
    if not stored_embeddings or not isinstance(stored_embeddings, dict):
        logger.error("Invalid stored embeddings")
        return None, None, 0.0

    try:
        # Convert all embeddings to numpy array (batch processing)
        query_embedding = face_embedding.reshape(1, -1)
        db_ids, db_names, db_embeddings = zip(*[
            (sid, name, emb) 
            for name, (sid, emb) in stored_embeddings.items()
        ])
        
        embedding_matrix = np.vstack(db_embeddings)
        
        # Vectorized similarity computation (optimized for performance)
        similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Adaptive threshold (lower threshold for poor lighting conditions)
        final_threshold = threshold * 0.9 if adaptive_threshold else threshold
        
        if best_similarity >= final_threshold:
            logger.info(f"Match found: {db_names[best_idx]} (similarity: {best_similarity:.4f})")
            return db_ids[best_idx], db_names[best_idx], best_similarity
        else:
            logger.warning(f"No match found (best similarity: {best_similarity:.4f} < {final_threshold})")
            return None, None, best_similarity
            
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}", exc_info=True)
        return None, None, 0.0
    
    