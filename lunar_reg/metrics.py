import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_registration_metrics(src_pts, ref_pts, H_matrix, warped_src, ref_img, grid_size=4):
    """
    Computes comprehensive quantitative metrics for lunar image registration evaluation.
    
    Parameters:
    - src_pts: (N, 2) array of moving source keypoint coordinates
    - ref_pts: (N, 2) array of target reference keypoint coordinates
    - H_matrix: (3, 3) estimated transformation matrix (Homography)
    - warped_src: Warped source image
    - ref_img: Reference image
    - grid_size: Grid size for calculating keypoint spatial uniformity
    
    Returns:
    - Dict of quantitative metrics
    """
    if len(src_pts) == 0 or H_matrix is None:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "subpixel_accuracy": float("inf"),
            "ssim": 0.0,
            "psnr": 0.0,
            "uniformity_std": float("inf")
        }

    # Transform source points using H matrix
    src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    pred_ref_pts_homo = (H_matrix @ src_pts_homo.T).T
    pred_ref_pts = pred_ref_pts_homo[:, :2] / pred_ref_pts_homo[:, 2:]
    
    # Calculate Euclidean reprojection errors
    errors = np.linalg.norm(pred_ref_pts - ref_pts, axis=1)
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    subpixel_accuracy = np.median(errors)
    
    inlier_count = len(src_pts)
    
    # Compute SSIM & PSNR on aligned non-black pixel regions
    mask = (warped_src > 0) & (ref_img > 0)
    if np.sum(mask) > 100:
        ssim_val = ssim(warped_src, ref_img, data_range=255)
        
        mse_img = np.mean((warped_src[mask].astype(np.float32) - ref_img[mask].astype(np.float32))**2)
        psnr_val = 10 * np.log10((255.0**2) / (mse_img + 1e-10)) if mse_img > 0 else 100.0
    else:
        ssim_val = 0.0
        psnr_val = 0.0
        
    # Spatial Uniformity Metric across NxN grid cells
    h, w = ref_img.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    counts = np.zeros((grid_size, grid_size))
    
    for (x, y) in ref_pts:
        gx = min(int(x // cell_w), grid_size - 1)
        gy = min(int(y // cell_h), grid_size - 1)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            counts[gy, gx] += 1
            
    uniformity_std = float(np.std(counts))
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "inlier_count": int(inlier_count),
        "subpixel_accuracy": float(subpixel_accuracy),
        "ssim": float(ssim_val),
        "psnr": float(psnr_val),
        "uniformity_std": uniformity_std,
        "grid_counts": counts.tolist()
    }
