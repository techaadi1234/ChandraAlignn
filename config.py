import os
import torch

class Config:
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image Resolution Settings
    IMAGE_SIZE = (512, 512)
    INPUT_CHANNELS = 1
    NUM_CLASSES = 1
    
    # Pre-trained Hugging Face Detection Model Configuration
    HF_CRATER_MODEL = "Gurveer05/moon-crater-boulder-detection-yolov5"
    DETECTION_CONF_THRESH = 0.48
    DETECTION_IOU_THRESH = 0.35
    
    # Preprocessing Settings
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_GRID_SIZE = (8, 8)
    
    # Feature Matching & ANMS Settings
    FEATURE_METHOD = "SIFT"  # SIFT or ORB
    MAX_FEATURES = 3000
    TARGET_KEYPOINTS = 800
    GRID_SHAPE = (10, 10)  # 10x10 spatial density grid for ANMS & Uniformity
    LOWE_RATIO_THRESH = 0.75
    
    # Registration Engine Settings
    MAGSAC_INLIER_THRESH = 2.0  # Reprojection threshold in pixels
    CORNER_SUBPIX_WIN_SIZE = (5, 5)
    CORNER_SUBPIX_ZERO_ZONE = (-1, -1)
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TEST_SAMPLES_DIR = os.path.join(DATA_DIR, "test_samples")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
