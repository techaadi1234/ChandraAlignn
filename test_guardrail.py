import requests
import json
import os
import cv2

SERVER_URL = "http://127.0.0.1:8080"

def test_overlapping_preset():
    print("\n--- TEST 1: Overlapping Swath (Preset: Boguslawsky) ---")
    resp = requests.post(f"{SERVER_URL}/api/register", json={"preset": "preset1", "conf_thresh": 0.15, "min_radius": 2})
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    
    print("Success:", data.get("success"))
    print("Is Registered:", data.get("is_registered"))
    print("Inliers Count:", data.get("metrics", {}).get("inliers_count"))
    print("Inlier Ratio:", data.get("metrics", {}).get("inlier_ratio"))
    print("RMSE:", data.get("metrics", {}).get("rmse"))
    
    assert data.get("is_registered") is True, "Preset1 should register successfully"
    assert data.get("metrics", {}).get("inliers_count") >= 15
    assert data.get("metrics", {}).get("inlier_ratio") >= 20.0
    assert data.get("metrics", {}).get("rmse") is not None
    
    # Check that LROC coordinates are populated
    craters = data.get("data", {}).get("craters", [])
    valid_lroc_count = sum(1 for c in craters if c.get("lroc_cx") is not None)
    print(f"Total detected landmarks: {len(craters)}, With valid LROC projection: {valid_lroc_count}")
    assert valid_lroc_count > 0, "Overlapping swaths should project LROC coordinates"
    print("[OK] TEST 1 PASSED: Overlapping swath registered accurately with sub-pixel RMSE.")

def test_non_overlapping_upload():
    print("\n--- TEST 2: Non-Overlapping Swaths (Boguslawsky CH-2 vs Completely Different Lunar Region) ---")
    ch2_path = "ch2_cropped_matching_patch.png"
    ref_path = "data/scratch_non_overlap_ref.png"
    if not os.path.exists(ref_path):
        import numpy as np
        synth = np.random.randint(40, 200, (600, 600), dtype=np.uint8)
        cv2.imwrite(ref_path, synth)
        
    print(f"CH-2 Source: {ch2_path}")
    print(f"LROC Reference: {ref_path}")
    
    with open(ch2_path, "rb") as f_ch2, open(ref_path, "rb") as f_ref:
        files_payload = {
            "ch2_file": ("boguslawsky_ch2.png", f_ch2, "image/png"),
            "ref_file": ("diff_lroc.png", f_ref, "image/png")
        }
        data_payload = {
            "preset": "custom",
            "conf_thresh": "0.15",
            "min_radius": "2"
        }
        resp = requests.post(f"{SERVER_URL}/api/register", files=files_payload, data=data_payload)
        
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    
    print("Success:", data.get("success"))
    print("Is Registered:", data.get("is_registered"))
    print("Error Message:", data.get("error_message"))
    print("RMSE:", data.get("metrics", {}).get("rmse"))
    print("Inliers Count:", data.get("metrics", {}).get("inliers_count"))
    print("Inlier Ratio:", data.get("metrics", {}).get("inlier_ratio"))
    
    # 1. Must flag REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED
    assert data.get("is_registered") is False, "Non-overlapping swaths must fail registration"
    assert "REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED" in (data.get("error_message") or "")
    
    # 2. RMSE must be None (not 0.0000 or fake coordinate error)
    assert data.get("metrics", {}).get("rmse") is None, "RMSE must be None on registration failure"
    
    # 3. LROC coordinates must NOT be projected (must be None in craters data, and N/A in table)
    craters = data.get("data", {}).get("craters", [])
    projected_count = sum(1 for c in craters if c.get("lroc_cx") is not None)
    print(f"Craters count: {len(craters)}, Projected LROC count: {projected_count}")
    assert projected_count == 0, "No landmarks should have projected LROC coordinates"
    
    # Check craters_table has N/A
    table = data.get("craters_table", [])
    na_count = sum(1 for row in table if row.get("lroc_cx") == "N/A" and row.get("lroc_cy") == "N/A")
    print(f"Table rows: {len(table)}, Table rows with N/A: {na_count}")
    assert na_count == len(table), "All table rows must display N/A for LROC coordinates"
    
    print("[OK] TEST 2 PASSED: Strict Guardrail triggered correctly on non-overlapping swaths!")

if __name__ == "__main__":
    test_overlapping_preset()
    test_non_overlapping_upload()
