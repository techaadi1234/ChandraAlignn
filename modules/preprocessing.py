import numpy as np
import cv2

def normalize_lunar_image(img):
    """
    Converts input image to single-channel 8-bit grayscale with standard intensity normalization.
    """
    if img is None:
        raise ValueError("Input image is None")
        
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Contrast stretching to [0, 255]
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)
        
    return gray

def apply_clahe(img, clip_limit=2.5, grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to handle severe solar illumination shadow variations across lunar terrain.
    """
    gray = normalize_lunar_image(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def apply_bandpass_filter(img, sigma_low=1.0, sigma_high=5.0):
    """
    Difference of Gaussians (DoG) bandpass filter to attenuate low-frequency solar shadows
    and enhance crater rim structural boundaries.
    """
    gray = normalize_lunar_image(img).astype(np.float32)
    blur_low = cv2.GaussianBlur(gray, (0, 0), sigma_low)
    blur_high = cv2.GaussianBlur(gray, (0, 0), sigma_high)
    dog = blur_low - blur_high
    norm_dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_dog
