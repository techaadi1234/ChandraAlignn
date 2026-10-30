"""
Lunar Image Registration & Deep Crater Segmentation Modules
"""

from .preprocessing import normalize_lunar_image, apply_clahe, apply_bandpass_filter
from .ml_crater_detector import load_crater_detector, extract_crater_landmarks, predict_craters
from .feature_matching import apply_anms, extract_features_and_fuse_landmarks, match_descriptors
from .registration_engine import register_images
from .lroc_fetcher import fetch_lroc_tile, parse_quickmap_url, normalize_to_uint8, pad_to_square
from .metrics import (
    calculate_subpixel_rmse,
    compute_inlier_ratio,
    compute_spatial_uniformity_score,
    compute_ssim
)

__all__ = [
    "normalize_lunar_image",
    "apply_clahe",
    "apply_bandpass_filter",
    "load_crater_detector",
    "extract_crater_landmarks",
    "predict_craters",
    "apply_anms",
    "extract_features_and_fuse_landmarks",
    "match_descriptors",
    "register_images",
    "calculate_subpixel_rmse",
    "compute_inlier_ratio",
    "compute_spatial_uniformity_score",
    "compute_ssim",
    "fetch_lroc_tile",
    "parse_quickmap_url",
    "normalize_to_uint8",
    "pad_to_square",
]
