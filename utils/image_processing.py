import cv2
import numpy as np


# Resize images to match model input size
def resize_images(images, size=(64, 64), interpolation=cv2.INTER_AREA):
    """
    Resize images to the specified size (default: 64x64).
    Ensures the output is in BGR format with 3 channels.
    """
    resized_images = []
    for img in images:
        if img is None or img.size == 0:
            raise ValueError("Invalid or empty image provided.")
        
        if img.shape[:2] != size:
            resized = cv2.resize(img, size, interpolation=interpolation)
            if resized.shape != (size[0], size[1], 3):  # Ensure 3-channel output
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            resized_images.append(resized)
        else:
            resized_images.append(img)  # Keep the original image if it already matches the size
    return np.array(resized_images)

# Apply gradient transformation (edge enhancement)
def apply_gradient(image, edge_threshold=50):
    """
    Apply gradient transformation to enhance edges in the image.
    Returns the image in BGR format.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image  # Image is already grayscale
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)

    if grad.max() > edge_threshold:
        print("Significant edges detected. Applying gradient transformation.")
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        return grad
    else:
        print("No significant edges detected. Skipping gradient transformation.")
        return image
    
    
# Check if an image has low contrast
def is_low_contrast(image, fraction_threshold=0.04):
    """
    Check if the image has low contrast based on the standard deviation of pixel intensities.
    Handles both grayscale (1 channel) and BGR (3 channels) images.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image  # Image is already grayscale
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    std_dev = np.std(gray)
    return std_dev < fraction_threshold * 255

def contrast_enhance(img, fraction_threshold=0.05, clip_limit=2.0):
    """
    Enhance the contrast of the image using histogram equalization in L*a*b color space.
    Returns the image in BGR format.
    """
    if is_low_contrast(img, fraction_threshold=fraction_threshold):
        print("Low contrast detected. Applying contrast enhancement.")
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_lab)

        if L.dtype != np.uint8:
            L = np.uint8(L)

        L = cv2.equalizeHist(L)
        img_lab_merge = cv2.merge((L, a, b))
        enhanced_img = cv2.cvtColor(img_lab_merge, cv2.COLOR_Lab2BGR)
        return enhanced_img
    else:
        print("Image has sufficient contrast. No enhancement applied.")
        return img

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image, contrast_threshold=0.05, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to enhance local contrast in the image.
    Returns the image in BGR format.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if is_low_contrast(l, fraction_threshold=contrast_threshold):
        print("Low local contrast detected. Applying CLAHE.")
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:
        print("Sufficient local contrast. Skipping CLAHE.")
        return image


# Apply noise reduction using Gaussian blur
def apply_noise_reduction(image, noise_threshold=10, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to reduce noise in the image.
    Returns the image in BGR format.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)

    if std_dev > noise_threshold:
        print("Noisy image detected. Applying noise reduction.")
        return cv2.GaussianBlur(image, kernel_size, sigma)
    else:
        print("Image is not noisy. Skipping noise reduction.")
        return image

# Main preprocessing function
def preprocess_image(image):
    """
    Apply all preprocessing steps to the image.
    Returns the preprocessed image in BGR format with shape (64, 64, 3).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided.")

    print(f"Input image shape: {image.shape}")  # Debug

    # Resize image to (64, 64, 3)
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 2:  # If grayscale, convert to 3 channels
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    print(f"Resized image shape: {resized.shape}")  # Debug

    # Apply noise reduction
    denoised = apply_noise_reduction(resized)
    print(f"Denoised image shape: {denoised.shape}")  # Debug

    # Apply contrast enhancement
    enhanced = contrast_enhance(denoised)
    print(f"Enhanced image shape: {enhanced.shape}")  # Debug

    # Apply CLAHE for local contrast enhancement
    clahe_applied = apply_clahe(enhanced)
    print(f"CLAHE applied image shape: {clahe_applied.shape}")  # Debug

    # Apply gradient transformation for edge enhancement
    final_image = apply_gradient(clahe_applied)
    print(f"Final image shape: {final_image.shape}")  # Debug

    # Ensure the final image has shape (64, 64, 3)
    if final_image.shape != (64, 64, 3):
        final_image = cv2.resize(final_image, (64, 64))
        if len(final_image.shape) == 2:
            final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

    return final_image


def align_face(image, landmarks):
    """
    Align the face using facial landmarks.
    """
    if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
        return image
    # Example: Align based on eye positionsq
    left_eye = landmarks['left_eye'] 
    right_eye = landmarks['right_eye']
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the image to align the eyes horizontally
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_face
