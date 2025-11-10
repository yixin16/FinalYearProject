import re # Make sure to import the 're' module at the top of your file
from flask import current_app
import os
import cv2
from datetime import datetime

def save_face_image(face_img, student_name, student_id):
    """
    Saves a face image to a student-specific directory with robust path handling 
    and returns the new total image count.
    """
    try:
        # Define the directory for this student's registration images
        student_dir = os.path.join(current_app.root_path, 'static', 'face_dataset', f"student_{student_id}")
        os.makedirs(student_dir, exist_ok=True)
        
        # Count existing images to determine the next number
        existing_images_count = len([name for name in os.listdir(student_dir) if name.endswith(('.jpg', '.png'))])
        new_image_number = existing_images_count + 1

        # Sanitize the student name for a safe filename
        sanitized_name = re.sub(r'[^\w_.-]', '', student_name).replace(' ', '_')
        
        filename = f"{sanitized_name}_{new_image_number}.jpg"
        filepath = os.path.join(student_dir, filename)
        
        current_app.logger.info(f"Attempting to save image to: {filepath}")
        success = cv2.imwrite(filepath, face_img)
        
        if not success:
            raise IOError(f"cv2.imwrite failed for path: {filepath}")
            
        return True, new_image_number
        
    except Exception as e:
        current_app.logger.error(f"Error in save_face_image for student {student_id}: {e}", exc_info=True)
        return False, 0
