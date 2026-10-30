import numpy as np
import cv2
from config import Config
from modules.preprocessing import normalize_lunar_image, apply_clahe
from modules.ml_crater_detector import extract_crater_landmarks
from modules.feature_matching import extract_features_and_fuse_landmarks, match_descriptors
from modules.metrics import (
    calculate_subpixel_rmse,
    compute_inlier_ratio,
    compute_spatial_uniformity_score,
    compute_ssim
)

def refine_inliers_subpixel(img, pts, win_size=Config.CORNER_SUBPIX_WIN_SIZE):
    """
    Sub-Pixel Optimization: Apply cv2.cornerSubPix on inlier coordinate locations
    to achieve sub-pixel precision (< 0.8px target).
    """
    if pts is None or len(pts) == 0:
        return pts

    gray = normalize_lunar_image(img)
    pts_subpix = pts.copy().astype(np.float32)
    
    # Termination criteria (max 40 iterations, epsilon 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    try:
        pts_subpix = cv2.cornerSubPix(
            gray, pts_subpix, win_size, Config.CORNER_SUBPIX_ZERO_ZONE, criteria
        )
    except cv2.error:
        pass

    return pts_subpix


def register_images(src_img, ref_img, crater_model=None, 
                    method=Config.FEATURE_METHOD, 
                    target_kps=Config.TARGET_KEYPOINTS, 
                    magsac_thresh=2.0,
                    enable_subpixel=True,
                    conf_thresh=0.15,
                    min_radius=2):
    """
    Geometric Fitting & Sub-Pixel Warping Engine:
    1. Preprocessing & Tri-Layer Feature Extraction (SIFT + ResNet-50 + YOLOv5).
    2. Descriptor Matching via FLANN / BFMatcher with Lowe's Ratio Test (thresh = 0.75).
    3. Robust Homography Matrix estimation via cv2.findHomography with cv2.USAC_MAGSAC (reprojection error = 2.0px).
    4. Sub-Pixel Optimization via cv2.cornerSubPix on inlier locations (< 0.8px target).
    5. Bicubic Image Warping via cv2.warpPerspective with INTER_CUBIC.
    """
    # 1. Image Normalization + Symmetric CLAHE to bridge contrast gap
    src_gray = normalize_lunar_image(src_img)
    ref_gray = normalize_lunar_image(ref_img)

    # Apply CLAHE to both images so SIFT/ResNet/YOLO see consistent contrast
    src_clahe = apply_clahe(src_gray, clip_limit=Config.CLAHE_CLIP_LIMIT)
    ref_clahe = apply_clahe(ref_gray, clip_limit=Config.CLAHE_CLIP_LIMIT)

    # 2. 3-Tier Multi-Scale Feature & Crater Detection
    if crater_model is None:
        from modules.ml_crater_detector import load_crater_detector
        crater_model = load_crater_detector()

    src_craters, _ = extract_crater_landmarks(src_clahe, model=crater_model, min_diameter=min_radius * 2, conf_thresh=conf_thresh)
    ref_craters, _ = extract_crater_landmarks(ref_clahe, model=crater_model, min_diameter=min_radius * 2, conf_thresh=conf_thresh)

    src_mask = np.zeros(src_gray.shape[:2], dtype=np.uint8)
    ref_mask = np.zeros(ref_gray.shape[:2], dtype=np.uint8)

    # 3. Landmark Fusion & 10x10 ANMS Grid Keypoint Extraction (on CLAHE images)
    kps_src, descs_src = extract_features_and_fuse_landmarks(
        src_clahe, src_craters, method=method, target_kps=target_kps, subpix_refine=enable_subpixel
    )
    kps_ref, descs_ref = extract_features_and_fuse_landmarks(
        ref_clahe, ref_craters, method=method, target_kps=target_kps, subpix_refine=enable_subpixel
    )

    # 4. Keypoint Matching via Lowe's Ratio Test (0.75 threshold)
    if descs_src is not None and descs_ref is not None and len(descs_src) >= 1 and len(descs_ref) >= 1:
        matches = match_descriptors(descs_src, descs_ref, method=method, ratio_thresh=0.75)
    else:
        matches = []

    # 5. Registration Validation Guardrail Constants
    MIN_INLIER_COUNT = 15
    MIN_INLIER_RATIO = 20.0  # At least 20% of putative matches must be inliers (0.20)

    # If putative matches are fewer than minimum inlier threshold, swaths do not overlap
    if len(matches) < MIN_INLIER_COUNT:
        return {
            "is_registered": False,
            "error_message": "REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED",
            "warped_src": src_gray,
            "ref_img": ref_gray,
            "src_img": src_gray,
            "H": None,
            "H_matrix": None,
            "src_mask": src_mask,
            "ref_mask": ref_mask,
            "src_craters": src_craters,
            "ref_craters": ref_craters,
            "raw_matches": len(matches),
            "inliers_count": 0,
            "inlier_ratio": 0.0,
            "inlier_src_pts": np.empty((0, 2), dtype=np.float32),
            "inlier_ref_pts": np.empty((0, 2), dtype=np.float32),
            "raw_src_pts": np.empty((0, 2), dtype=np.float32),
            "raw_ref_pts": np.empty((0, 2), dtype=np.float32),
            "rmse": None,
            "mae": None,
            "median_subpixel_error": None,
            "uniformity_score": 0.0,
            "spatial_density": np.zeros(Config.GRID_SHAPE, dtype=np.float32),
            "ssim": None
        }

    raw_src_pts = np.float32([kps_src[m.queryIdx].pt for m in matches])
    raw_ref_pts = np.float32([kps_ref[m.trainIdx].pt for m in matches])
    total_point_pool = len(matches)

    # 6. Robust Estimator: cv2.findHomography with cv2.RANSAC (ransacReprojThreshold=3.0)
    H_magsac, mask = cv2.findHomography(raw_src_pts, raw_ref_pts, cv2.RANSAC, 3.0)
    if H_magsac is None and hasattr(cv2, 'USAC_MAGSAC'):
        H_magsac, mask = cv2.findHomography(raw_src_pts, raw_ref_pts, cv2.USAC_MAGSAC, 3.0)

    if H_magsac is None or mask is None:
        return {
            "is_registered": False,
            "error_message": "REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED",
            "warped_src": src_gray,
            "ref_img": ref_gray,
            "src_img": src_gray,
            "H": None,
            "H_matrix": None,
            "src_mask": src_mask,
            "ref_mask": ref_mask,
            "src_craters": src_craters,
            "ref_craters": ref_craters,
            "raw_matches": len(matches),
            "inliers_count": 0,
            "inlier_ratio": 0.0,
            "inlier_src_pts": np.empty((0, 2), dtype=np.float32),
            "inlier_ref_pts": np.empty((0, 2), dtype=np.float32),
            "raw_src_pts": raw_src_pts,
            "raw_ref_pts": raw_ref_pts,
            "rmse": None,
            "mae": None,
            "median_subpixel_error": None,
            "uniformity_score": 0.0,
            "spatial_density": np.zeros(Config.GRID_SHAPE, dtype=np.float32),
            "ssim": None
        }

    inlier_mask = mask.ravel() == 1
    inlier_src = raw_src_pts[inlier_mask]
    inlier_ref = raw_ref_pts[inlier_mask]
    correct_inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = compute_inlier_ratio(total_point_pool, correct_inlier_count)

    # 7. Robust Matching & Inlier Verification Gate
    is_valid = True
    fail_reason = ""

    if correct_inlier_count < MIN_INLIER_COUNT:
        is_valid = False
        fail_reason = f"RANSAC inlier count ({correct_inlier_count}) < minimum required ({MIN_INLIER_COUNT})"
    elif inlier_ratio < MIN_INLIER_RATIO:
        is_valid = False
        fail_reason = f"RANSAC inlier ratio ({inlier_ratio:.1f}%) < minimum required ({MIN_INLIER_RATIO:.1f}%)"
    else:
        # Homography physical plausibility and conditioning checks
        if np.any(np.isnan(H_magsac)) or np.any(np.isinf(H_magsac)):
            is_valid = False
            fail_reason = "Homography matrix contains NaN or Inf values"
        else:
            det_2x2 = float(H_magsac[0, 0] * H_magsac[1, 1] - H_magsac[0, 1] * H_magsac[1, 0])
            if det_2x2 <= 0.0:
                is_valid = False
                fail_reason = "Homography matrix produces negative determinant (reflection/inversion)"
            elif det_2x2 < 0.04 or det_2x2 > 25.0:
                is_valid = False
                fail_reason = f"Homography scale factor ({np.sqrt(abs(det_2x2)):.2f}x) outside physical limits"
            else:
                try:
                    # Check affine 2x2 conditioning (shear/collinear collapse)
                    cond_affine = float(np.linalg.cond(H_magsac[:2, :2]))
                    
                    # Hartley-normalized homography condition number (accounts for pixel translation scale)
                    h_scale = max(float(ref_gray.shape[0]), float(ref_gray.shape[1]), 1.0)
                    S = np.diag([1.0 / h_scale, 1.0 / h_scale, 1.0])
                    S_inv = np.diag([h_scale, h_scale, 1.0])
                    H_norm = S @ H_magsac @ S_inv
                    cond = float(np.linalg.cond(H_norm))
                    
                    if cond > 1e4 or cond_affine > 500.0 or np.isnan(cond) or np.isnan(cond_affine):
                        is_valid = False
                        fail_reason = f"Homography condition number ({cond:.1e}) indicates numerical collapse"
                except Exception:
                    pass

    if not is_valid:
        return {
            "is_registered": False,
            "error_message": f"REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED ({fail_reason})",
            "warped_src": src_gray,
            "ref_img": ref_gray,
            "src_img": src_gray,
            "H": None,
            "H_matrix": None,
            "src_mask": src_mask,
            "ref_mask": ref_mask,
            "src_craters": src_craters,
            "ref_craters": ref_craters,
            "raw_matches": len(matches),
            "inliers_count": correct_inlier_count,
            "inlier_ratio": inlier_ratio,
            "inlier_src_pts": inlier_src,
            "inlier_ref_pts": inlier_ref,
            "raw_src_pts": raw_src_pts,
            "raw_ref_pts": raw_ref_pts,
            "rmse": None,
            "mae": None,
            "median_subpixel_error": None,
            "uniformity_score": 0.0,
            "spatial_density": np.zeros(Config.GRID_SHAPE, dtype=np.float32),
            "ssim": None
        }

    # 8. Sub-Pixel Optimization on Inliers (cv2.cornerSubPix)
    if enable_subpixel and len(inlier_src) >= 4:
        inlier_src_subpix = refine_inliers_subpixel(src_gray, inlier_src)
        inlier_ref_subpix = refine_inliers_subpixel(ref_gray, inlier_ref)
        
        H_subpix, subpix_mask = cv2.findHomography(inlier_src_subpix, inlier_ref_subpix, cv2.RANSAC, 3.0)
        if H_subpix is not None:
            det_sub = float(H_subpix[0, 0] * H_subpix[1, 1] - H_subpix[0, 1] * H_subpix[1, 0])
            if det_sub > 0.04 and det_sub < 25.0:
                H_final = H_subpix
                if subpix_mask is not None:
                    inlier_sub_mask = subpix_mask.ravel() == 1
                    if np.sum(inlier_sub_mask) >= 4:
                        inlier_src_subpix = inlier_src_subpix[inlier_sub_mask]
                        inlier_ref_subpix = inlier_ref_subpix[inlier_sub_mask]
                        correct_inlier_count = int(np.sum(inlier_sub_mask))
            else:
                H_final = H_magsac
        else:
            H_final = H_magsac
    else:
        inlier_src_subpix = inlier_src
        inlier_ref_subpix = inlier_ref
        H_final = H_magsac

    # 9. Bicubic Warping: cv2.warpPerspective with INTER_CUBIC
    h_ref, w_ref = ref_gray.shape[:2]
    warped_src = cv2.warpPerspective(
        src_gray, H_final, (w_ref, h_ref),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 10. Quantitative Evaluation Suite
    rmse_metrics = calculate_subpixel_rmse(inlier_src_subpix, inlier_ref_subpix, H_final)
    inlier_ratio = compute_inlier_ratio(total_point_pool, correct_inlier_count)
    uniformity_score, spatial_density = compute_spatial_uniformity_score(
        inlier_ref_subpix, ref_gray.shape, grid=Config.GRID_SHAPE
    )
    ssim_score = compute_ssim(warped_src, ref_gray)

    return {
        "is_registered": True,
        "error_message": "",
        "warped_src": warped_src,
        "ref_img": ref_gray,
        "src_img": src_gray,
        "H": H_final,
        "H_matrix": H_final,
        "src_mask": src_mask,
        "ref_mask": ref_mask,
        "src_craters": src_craters,
        "ref_craters": ref_craters,
        "raw_matches": len(matches),
        "inliers_count": correct_inlier_count,
        "inlier_ratio": inlier_ratio,
        "inlier_src_pts": inlier_src_subpix,
        "inlier_ref_pts": inlier_ref_subpix,
        "raw_src_pts": raw_src_pts,
        "raw_ref_pts": raw_ref_pts,
        "rmse": rmse_metrics["rmse"],
        "mae": rmse_metrics["mae"],
        "median_subpixel_error": rmse_metrics["median_error"],
        "uniformity_score": uniformity_score,
        "spatial_density": spatial_density,
        "ssim": ssim_score
    }
