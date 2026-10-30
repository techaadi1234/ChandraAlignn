import numpy as np
import cv2
import math
from .metrics import compute_registration_metrics

class LunarImageRegistrar:
    """
    State-of-the-Art Lunar Image Registration Engine.
    Handles Illumination variations, Viewpoint distortion, and Scale Disparities
    with sub-pixel accuracy and spatial uniformity.
    """
    def __init__(self, method="SIFT", nfeatures=3000, target_uniform_points=800):
        self.method = method
        self.nfeatures = nfeatures
        self.target_uniform_points = target_uniform_points

    def preprocess_image(self, img):
        """
        Applies local contrast enhancement (CLAHE) to handle severe lunar shadow variations.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def apply_anms(self, keypoints, descriptors, num_to_keep, c_max=0.9):
        """
        Adaptive Non-Maximal Suppression (ANMS) for Uniform Feature Distribution.
        Prevents keypoints from clustering on a single high-contrast crater rim.
        """
        if len(keypoints) <= num_to_keep:
            return keypoints, descriptors

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        response = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # Sort by response descending
        order = np.argsort(-response)
        pts = pts[order]
        descriptors = descriptors[order]
        keypoints = [keypoints[i] for i in order]

        num_pts = len(pts)
        radii = np.full(num_pts, np.inf)

        # Calculate minimum distance to a stronger keypoint
        for i in range(1, num_pts):
            stronger_pts = pts[:i]
            # Find points where response[j] * c_max > response[i]
            distances = np.linalg.norm(stronger_pts - pts[i], axis=1)
            radii[i] = np.min(distances)

        # Select top num_to_keep points with largest radii
        selected_idx = np.argsort(-radii)[:num_to_keep]
        
        selected_kps = [keypoints[i] for i in selected_idx]
        selected_descs = descriptors[selected_idx]

        return selected_kps, selected_descs

    def extract_features(self, img):
        """
        Detects multiscale features and applies ANMS for spatial uniformity.
        """
        prep_img = self.preprocess_image(img)

        if self.method == "SIFT":
            detector = cv2.SIFT_create(nfeatures=self.nfeatures, contrastThreshold=0.02, edgeThreshold=10)
        else:
            detector = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=1.2, nlevels=8)

        keypoints, descriptors = detector.detectAndCompute(prep_img, None)

        if keypoints is None or len(keypoints) == 0:
            return [], None

        # Apply ANMS for uniform spatial coverage
        kps_uniform, descs_uniform = self.apply_anms(keypoints, descriptors, self.target_uniform_points)
        return kps_uniform, descs_uniform

    def match_descriptors(self, descs1, descs2, ratio_thresh=0.75):
        """
        Descriptor matching using FLANN and Lowe's Ratio Test.
        """
        if self.method == "SIFT":
            # FLANN parameters for SIFT (KDTree)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # FLANN parameters for ORB (LSH)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        matches = matcher.knnMatch(descs1, descs2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_transformation(self, kps_src, kps_ref, matches):
        """
        Robust homography estimation using MAGSAC++ or USAC_MAGSAC (OpenCV 4.5+).
        """
        if len(matches) < 4:
            return None, None, None, None

        src_pts = np.float32([kps_src[m.queryIdx].pt for m in matches])
        ref_pts = np.float32([kps_ref[m.trainIdx].pt for m in matches])

        # Use MAGSAC++ for superior robustness under high outlier ratios
        if hasattr(cv2, 'USAC_MAGSAC'):
            H, mask = cv2.findHomography(src_pts, ref_pts, cv2.USAC_MAGSAC, 2.5)
        else:
            H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 3.0)

        if mask is None:
            return None, None, None, None

        inlier_mask = mask.ravel() == 1
        inlier_src = src_pts[inlier_mask]
        inlier_ref = ref_pts[inlier_mask]

        return H, inlier_mask, inlier_src, inlier_ref

    def refine_subpixel_ecc(self, ref_img, src_img, H_init):
        """
        Enhanced Correlation Coefficient (ECC) algorithm for sub-pixel accuracy optimization.
        Refines initial Homography matrix H using direct gradient descent on intensity normalized image maps.
        Note: cv2.findTransformECC maps template (ref) coordinates to input (src) coordinates.
        """
        if H_init is None:
            return H_init

        ref_prep = self.preprocess_image(ref_img)
        src_prep = self.preprocess_image(src_img)

        # H_init maps src -> ref. ECC warpMatrix maps ref -> src, so we invert H_init.
        try:
            H_inv_init = np.linalg.inv(H_init).astype(np.float32)
        except np.linalg.LinAlgError:
            return H_init

        # Define termination criteria (max iterations, eps)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

        try:
            _, H_inv_refined = cv2.findTransformECC(
                ref_prep, src_prep, H_inv_init, 
                cv2.MOTION_HOMOGRAPHY, criteria, None, 5
            )
            H_refined = np.linalg.inv(H_inv_refined)
            return H_refined
        except cv2.error:
            # If ECC fails to converge, return initial H
            return H_init

    def register(self, src_img, ref_img, enable_ecc_subpixel=True):
        """
        Executes full registration pipeline:
        Extract -> ANMS -> FLANN -> MAGSAC++ -> ECC Sub-Pixel Refinement -> Warping & Metrics.
        """
        # 1. Feature extraction with ANMS
        kps_src, descs_src = self.extract_features(src_img)
        kps_ref, descs_ref = self.extract_features(ref_img)

        if descs_src is None or descs_ref is None:
            raise ValueError("Feature extraction failed on one or both images.")

        # 2. Descriptor matching
        matches = self.match_descriptors(descs_src, descs_ref)

        # 3. Geometric transform (MAGSAC++)
        H, mask, inlier_src, inlier_ref = self.estimate_transformation(kps_src, kps_ref, matches)

        if H is None:
            raise ValueError("Insufficient inlier matches to calculate Homography.")

        # 4. Sub-pixel ECC refinement
        if enable_ecc_subpixel:
            H_final = self.refine_subpixel_ecc(ref_img, src_img, H)
        else:
            H_final = H

        # 5. Warp moving source image into reference coordinate frame
        h_ref, w_ref = ref_img.shape[:2]
        warped_src = cv2.warpPerspective(src_img, H_final, (w_ref, h_ref), 
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 6. Evaluation metrics
        raw_match_count = len(matches)
        inlier_count = len(inlier_src)
        inlier_ratio = (inlier_count / raw_match_count * 100.0) if raw_match_count > 0 else 0.0

        metrics = compute_registration_metrics(inlier_src, inlier_ref, H_final, warped_src, ref_img)
        metrics["raw_match_count"] = raw_match_count
        metrics["inlier_ratio"] = inlier_ratio

        return {
            "warped_src": warped_src,
            "H_matrix": H_final,
            "inlier_src_pts": inlier_src,
            "inlier_ref_pts": inlier_ref,
            "raw_match_count": raw_match_count,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "metrics": metrics,
            "kps_src": kps_src,
            "kps_ref": kps_ref
        }
