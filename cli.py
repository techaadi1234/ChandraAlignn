import argparse
import cv2
import json
import os
from lunar_reg.core import LunarImageRegistrar
from lunar_reg.generator import LunarImageGenerator
from lunar_reg.visualizer import create_checkerboard, create_residual_error_map, draw_matches_side_by_side

def main():
    parser = argparse.ArgumentParser(description="Chandrayaan-2 Lunar Image Registration Tool")
    parser.add_argument("--source", type=str, help="Path to moving source lunar image")
    parser.add_argument("--reference", type=str, help="Path to fixed reference lunar image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save registered results")
    parser.add_argument("--synthetic", action="store_true", help="Generate and run on synthetic lunar crater dataset")
    parser.add_argument("--method", type=str, default="SIFT", choices=["SIFT", "ORB"], help="Feature detector")
    parser.add_argument("--no_ecc", action="store_true", help="Disable sub-pixel ECC optimization")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic or (not args.source and not args.reference):
        print("[+] Generating synthetic lunar crater image pair (Illumination delta = 20 deg, Tilt = 12 deg)...")
        gen = LunarImageGenerator(width=800, height=800, seed=42)
        ref_img, src_img, H_gt = gen.generate_pair(scale=1.08, angle_deg=10.0, tx=20.0, ty=-10.0, sun_angle1=40, sun_angle2=60)
        
        cv2.imwrite(os.path.join(args.output_dir, "synthetic_ref.png"), ref_img)
        cv2.imwrite(os.path.join(args.output_dir, "synthetic_src.png"), src_img)
    else:
        ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        src_img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or src_img is None:
            print("Error: Could not load input image files.")
            return

    print(f"[+] Initializing Lunar Registration Engine ({args.method})...")
    registrar = LunarImageRegistrar(method=args.method, nfeatures=3000, target_uniform_points=800)

    print("[+] Processing: ANMS Keypoint Selection -> FLANN Matching -> MAGSAC++ -> ECC Sub-Pixel Optimization...")
    res = registrar.register(src_img, ref_img, enable_ecc_subpixel=not args.no_ecc)

    # Save outputs
    warped_path = os.path.join(args.output_dir, "registered_source.png")
    checkerboard_path = os.path.join(args.output_dir, "checkerboard_overlay.png")
    matches_path = os.path.join(args.output_dir, "inlier_matches.png")
    residual_path = os.path.join(args.output_dir, "residual_error_map.png")
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")

    cv2.imwrite(warped_path, res["warped_src"])
    
    checkerboard = create_checkerboard(ref_img, res["warped_src"])
    cv2.imwrite(checkerboard_path, checkerboard)

    matches_img = draw_matches_side_by_side(ref_img, src_img, res["inlier_ref_pts"], res["inlier_src_pts"])
    cv2.imwrite(matches_path, matches_img)

    residual_map = create_residual_error_map(ref_img, res["warped_src"])
    cv2.imwrite(residual_path, residual_map)

    with open(metrics_path, "w") as f:
        json.dump(res["metrics"], f, indent=4)

    print("\n==================================================")
    print("           LUNAR REGISTRATION RESULTS             ")
    print("==================================================")
    print(f" Raw Matches Found : {res['raw_match_count']}")
    print(f" Inlier Match Count: {res['inlier_count']}")
    print(f" Inlier Ratio      : {res['inlier_ratio']:.2f}%")
    print(f" Reprojection RMSE : {res['metrics']['rmse']:.4f} pixels")
    print(f" Sub-Pixel Accuracy: {res['metrics']['subpixel_accuracy']:.4f} pixels (Median Error)")
    print(f" Structural SSIM   : {res['metrics']['ssim']:.4f}")
    print(f" Image PSNR        : {res['metrics']['psnr']:.2f} dB")
    print(f" Uniformity Std    : {res['metrics']['uniformity_std']:.2f}")
    print("==================================================")
    print(f"[+] All output files successfully saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()
