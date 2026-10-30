import numpy as np
import cv2
from config import Config
from modules.preprocessing import apply_clahe, normalize_lunar_image

def apply_anms(keypoints, descriptors, num_to_keep=800, grid_shape=(10, 10), c_max=0.9,
               img_shape=None, max_per_cell=None):
    """
    2-Phase Adaptive Non-Maximal Suppression with Grid-Based Cell Capping.

    Phase 1 — Spatial Grid Pre-Filter:
        Divide the image into a grid_shape (default 10×10) spatial grid.
        Keep only the top `max_per_cell` strongest keypoints per cell.
        This prevents high-density keypoint clusters (e.g. a texture-rich crater
        interior) from flooding the match pool and starving peripheral regions.

    Phase 2 — Classic ANMS Radius Suppression:
        From the grid-filtered set, rank by suppression radius to nearest stronger
        neighbour and keep the top `num_to_keep` most spatially spread keypoints.
    """
    if len(keypoints) == 0:
        return keypoints, descriptors

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    # ── Phase 1: Per-Cell Grid Capping ─────────────────────────────────────────
    if img_shape is not None:
        h, w = img_shape[:2]
    else:
        # Infer from point cloud bounding box (safe fallback)
        h = float(pts[:, 1].max() + 1) if len(pts) > 0 else 1024.0
        w = float(pts[:, 0].max() + 1) if len(pts) > 0 else 1024.0

    grid_r, grid_c = grid_shape
    cell_h = max(1.0, h / grid_r)
    cell_w = max(1.0, w / grid_c)

    # Default: allow at most ceil(num_to_keep / (grid_r*grid_c) * 3) per cell
    # This gives headroom for the ANMS phase to work with while still capping clusters
    if max_per_cell is None:
        max_per_cell = max(3, int(np.ceil(num_to_keep / (grid_r * grid_c) * 3)))

    # Assign each keypoint to a cell, keep top-max_per_cell by response per cell
    cell_buckets = {}  # (row_idx, col_idx) -> list of original indices
    for idx in range(len(keypoints)):
        ri = min(int(pts[idx, 1] / cell_h), grid_r - 1)
        ci = min(int(pts[idx, 0] / cell_w), grid_c - 1)
        cell_buckets.setdefault((ri, ci), []).append(idx)

    kept_indices = []
    for (ri, ci), idxs in cell_buckets.items():
        if len(idxs) <= max_per_cell:
            kept_indices.extend(idxs)
        else:
            # Sort by descending response, keep top max_per_cell
            idxs_sorted = sorted(idxs, key=lambda i: -responses[i])
            kept_indices.extend(idxs_sorted[:max_per_cell])

    # Rebuild filtered arrays
    kept_indices = np.array(kept_indices)
    pts = pts[kept_indices]
    responses = responses[kept_indices]
    keypoints = [keypoints[i] for i in kept_indices]
    descriptors = descriptors[kept_indices] if descriptors is not None else None

    # ── Phase 2: Classic ANMS Radius Suppression ────────────────────────────────
    if len(keypoints) <= num_to_keep:
        return keypoints, descriptors

    # Sort descending by response
    order = np.argsort(-responses)
    pts = pts[order]
    responses = responses[order]
    keypoints = [keypoints[i] for i in order]
    descriptors = descriptors[order] if descriptors is not None else None

    num_pts = len(pts)
    radii = np.full(num_pts, np.inf)

    # Vectorised batch ANMS: for each point i, find minimum distance to any
    # stronger point j<i that satisfies c_max * response[j] > response[i]
    for i in range(1, num_pts):
        stronger_mask = responses[:i] * c_max > responses[i]
        if np.any(stronger_mask):
            stronger_pts = pts[:i][stronger_mask]
            dists = np.linalg.norm(stronger_pts - pts[i], axis=1)
            radii[i] = np.min(dists)

    # Select top num_to_keep keypoints with largest suppression radii
    selected_idx = np.argsort(-radii)[:num_to_keep]

    selected_kps = [keypoints[i] for i in selected_idx]
    selected_descs = descriptors[selected_idx] if descriptors is not None else None

    return selected_kps, selected_descs


def extract_features_and_fuse_landmarks(img, crater_landmarks=[], method="SIFT", nfeatures=3000, target_kps=800, subpix_refine=True):
    """
    3-Tier Multi-Scale Fusion Pipeline:
    - Tier 1 (Micro Extrema - SIFT): cv2.SIFT_create() for local micro-textures & crater rim points (<12px).
    - Tier 2 (Deep Invariant Features - ResNet-50): Large crater basins (>50px) invariant under shadow flips.
    - Tier 3 (Macro Landmark Anchors - YOLOv5): Geometrical crater centroids.
    - Fusion & Deduplication: Merge candidates with an 8px spatial suppression window.
    - Descriptors: Compute 128-dimensional SIFT descriptors for all unified landmarks.
    - Spatial Uniformity: 10x10 grid ANMS targeting 600-800 keypoints.
    """
    enhanced = apply_clahe(img, clip_limit=2.5)
    gray = normalize_lunar_image(img)
    
    if method.upper() == "SIFT":
        sift_detector = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.008, edgeThreshold=15)
    else:
        sift_detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)

    # Tier 1: Micro Extrema SIFT keypoints
    sift_kps, _ = sift_detector.detectAndCompute(enhanced, None)
    if sift_kps is None:
        sift_kps = []

    fused_kps_raw = []
    existing_coords = []

    yolo_centroids = []
    resnet_centroids = []

    if isinstance(crater_landmarks, list):
        for item in crater_landmarks:
            if isinstance(item, dict):
                src = item.get("layer_source", "YOLOv5")
                cx, cy = item["centroid"]
                if src == "ResNet-50":
                    resnet_centroids.append((cx, cy))
                else:
                    yolo_centroids.append((cx, cy))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                yolo_centroids.append((float(item[0]), float(item[1])))

    # 1. Tier 3: Macro Landmark Anchors (YOLOv5 Centroids) - Priority 1 (Response = 9999.0)
    for (cx, cy) in yolo_centroids:
        kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=16.0, response=9999.0)
        fused_kps_raw.append(kp)
        existing_coords.append((float(cx), float(cy)))

    # 2. Tier 2: Deep Invariant Features (ResNet-50 Basin Peaks) - Priority 2 (Response = 5000.0) with 8px Suppression
    for (cx, cy) in resnet_centroids:
        too_close = False
        for (ex, ey) in existing_coords:
            if np.hypot(cx - ex, cy - ey) < 8.0:
                too_close = True
                break
        if not too_close:
            kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=32.0, response=5000.0)
            fused_kps_raw.append(kp)
            existing_coords.append((float(cx), float(cy)))

    # 3. Tier 1: Micro Extrema (SIFT Points <12px & rim extrema) with 8px Suppression Window
    for kp in sift_kps:
        cx, cy = kp.pt
        too_close = False
        for (ex, ey) in existing_coords:
            if np.hypot(cx - ex, cy - ey) < 8.0:
                too_close = True
                break
        if not too_close:
            fused_kps_raw.append(kp)
            existing_coords.append((float(cx), float(cy)))

    if len(fused_kps_raw) == 0:
        return [], np.array([])

    # Sub-pixel corner optimization via cv2.cornerSubPix
    if subpix_refine and len(fused_kps_raw) > 0:
        pts = np.array([kp.pt for kp in fused_kps_raw], dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        try:
            refined_pts = cv2.cornerSubPix(gray, pts, Config.CORNER_SUBPIX_WIN_SIZE, Config.CORNER_SUBPIX_ZERO_ZONE, criteria)
            refined_pts = refined_pts.reshape(-1, 2)
            for i, kp in enumerate(fused_kps_raw):
                kp.pt = (float(refined_pts[i][0]), float(refined_pts[i][1]))
        except cv2.error:
            pass

    # Compute SIFT descriptors for all unified landmarks
    fused_kps_final, descriptors = sift_detector.compute(enhanced, fused_kps_raw)
    if fused_kps_final is None or descriptors is None:
        return [], np.array([])

    # Apply 10x10 Grid-Based ANMS with per-cell capping then radius suppression
    kps_uniform, descs_uniform = apply_anms(
        fused_kps_final, descriptors,
        num_to_keep=target_kps,
        grid_shape=Config.GRID_SHAPE,
        img_shape=enhanced.shape,     # pass image shape for accurate cell sizing
    )

    return kps_uniform, descs_uniform


def match_descriptors(descs_src, descs_ref, method="SIFT", ratio_thresh=0.75):
    """
    Keypoint Matching: FLANN / BFMatcher with Lowe's Ratio Test (threshold = 0.75).
    """
    if descs_src is None or descs_ref is None or len(descs_src) < 4 or len(descs_ref) < 4:
        return []

    if method.upper() == "SIFT":
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        knn_matches = matcher.knnMatch(descs_src, descs_ref, k=2)
    except cv2.error:
        return []

    good_matches = []
    for m_n in knn_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches
