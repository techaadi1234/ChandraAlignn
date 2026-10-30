import os
import sys
import argparse
import cv2
import json
import numpy as np

from config import Config
from modules.preprocessing import normalize_lunar_image
from modules.ml_crater_detector import load_crater_detector
from modules.registration_engine import register_images

def test_isro_raw_images(src_path, ref_path, output_dir="data/output"):
    """
    Command-line testing script for running sub-pixel registration and pre-trained
    YOLOv5 crater detection directly on raw ISRO Chandrayaan-2 / LROC satellite images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("==========================================================================")
    print("  [+] ChandraAlign: ISRO Chandrayaan-2 Raw Imagery Testing Pipeline        ")
    print("==========================================================================")
    print(f" Source Image (Chandrayaan-2) : {src_path}")
    print(f" Reference Image (LROC Base)  : {ref_path}")
    print(f" Output Directory             : {output_dir}")
    print("==========================================================================")

    # 1. Load Images
    if not os.path.exists(src_path):
        print(f"[!] Error: Source image file '{src_path}' not found!")
        return
    if not os.path.exists(ref_path):
        print(f"[!] Error: Reference image file '{ref_path}' not found!")
        return

    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    if src_img is None or ref_img is None:
        print("[!] Error: Failed to decode image file(s). Ensure PNG, JPG, or TIFF format.")
        return

    print(f"[+] Loaded Source Image    : {src_img.shape[1]}x{src_img.shape[0]} px")
    print(f"[+] Loaded Reference Image : {ref_img.shape[1]}x{ref_img.shape[0]} px")

    # 2. Load Pre-trained Hugging Face Detection Model
    crater_model = load_crater_detector()

    # 3. Execute Sub-Pixel Registration Engine
    print("\n[+] Running ChandraAlign Sub-Pixel Registration Engine...")
    res = register_images(
        src_img, ref_img,
        crater_model=crater_model,
        method=Config.FEATURE_METHOD,
        target_kps=Config.TARGET_KEYPOINTS,
        magsac_thresh=Config.MAGSAC_INLIER_THRESH,
        enable_subpixel=True
    )

    # 4. Save Output Products
    warped_path = os.path.join(output_dir, "registered_chandrayaan2.png")
    cv2.imwrite(warped_path, res["warped_src"])

    diff = cv2.absdiff(ref_img, res["warped_src"])
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff_colored[res["warped_src"] == 0] = 0
    diff_path = os.path.join(output_dir, "residual_heatmap.png")
    cv2.imwrite(diff_path, diff_colored)

    report_path = os.path.join(output_dir, "alignment_report.json")
    report = {
        "source_image": src_path,
        "reference_image": ref_path,
        "metrics": {
            "subpixel_rmse_px": res["rmse"],
            "mae_px": res["mae"],
            "median_subpixel_error_px": res["median_subpixel_error"],
            "inliers_count": res["inliers_count"],
            "inlier_ratio_percent": res["inlier_ratio"],
            "spatial_uniformity_score": res["uniformity_score"],
            "ssim": res["ssim"]
        },
        "extracted_crater_landmarks_count": len(res["src_craters"])
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # 5. Print Results Summary
    print("\n==========================================================================")
    print("                    EVALUATION METRIC RESULTS SUMMARY                     ")
    print("==========================================================================")
    print(f" Sub-Pixel Reprojection RMSE : {res['rmse']:.4f} pixels  (Target < 0.80 px)")
    print(f" Median Sub-Pixel Error     : {res['median_subpixel_error']:.4f} pixels  (Target < 0.50 px)")
    print(f" Inliers Count               : {res['inliers_count']} points")
    print(f" Inlier Match Ratio          : {res['inlier_ratio']:.2f}%")
    print(f" 10x10 Spatial Uniformity    : {res['uniformity_score']:.4f}  (Target > 0.75)")
    print(f" Structural SSIM             : {res['ssim']:.4f}")
    print("==========================================================================")
    print(f"[+] Saved Registered Output  : {warped_path}")
    print(f"[+] Saved Residual Heatmap   : {diff_path}")
    print(f"[+] Saved Metrics Report JSON : {report_path}")
    print("==========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ChandraAlign on Raw ISRO Satellite Images")
    parser.add_argument("--src", type=str, required=True, help="Path to raw Chandrayaan-2 moving image")
    parser.add_argument("--ref", type=str, required=True, help="Path to raw LROC reference base image")
    parser.add_argument("--out", type=str, default="data/output", help="Output directory for registered products")
    args = parser.parse_args()

    test_isro_raw_images(args.src, args.ref, args.out)
