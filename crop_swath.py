"""
ChandraAlign — Automated Chandrayaan-2 Swath Coordinate Cropper & LROC Matcher
=============================================================================
Automated Python module to calculate orbital-track pixel offsets via linear
interpolation, extract the exact target sub-array from long vertical Chandrayaan-2
TMC-2 swaths, fetch matching NASA LROC reference tiles over WMS, and optionally
load them into the live dual-viewport comparison dashboard.

Usage:
    python crop_swath.py --lat-min -7.094 --lat-max -5.418 --center-lon 142.260
    python crop_swath.py --load-to-dashboard
"""

import os
import sys
import argparse
import json
import base64
import requests
import numpy as np
import cv2

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Default ChandraAlign mission parameters
DEFAULT_SWATH_PATH = os.path.join(
    "data", "unpacked_swath", "browse", "calibrated", "20260813",
    "ch2_tmc_nca_20260813T0627378526_b_brw_d18.png"
)
DEFAULT_LAT_NORTH = -5.44     # Northern boundary (Y = 0)
DEFAULT_LAT_SOUTH = -29.48    # Southern boundary (Y = 14811)
DEFAULT_LAT_RANGE = (-7.094, -5.418)
DEFAULT_CENTER_LAT = -6.256
DEFAULT_CENTER_LON = 142.260
DEFAULT_OUTPUT_PATH = "ch2_cropped_matching_patch.png"
DEFAULT_DASHBOARD_URL = "http://localhost:8080"


def fetch_lroc_wms_tile(bounds, size=1024, timeout=12):
    """
    Queries NASA Moon Trek or USGS Astrogeology Planetary WMS for the matching reference tile.
    Falls back to the local dashboard proxy endpoint if direct planetary WMS is unreachable.
    """
    west = float(bounds["west"])
    east = float(bounds["east"])
    south = float(bounds["south"])
    north = float(bounds["north"])

    headers = {
        "User-Agent": "ChandraAlign-ISRO-TMC2-Cropper/2.0 (Chandrayaan-2 Research)"
    }

    # 1. USGS Astrogeology LROC WAC Global Mosaic
    usgs_url = (
        f"https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map"
        f"&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS=LROC_WAC&STYLES=&SRS=EPSG:4326"
        f"&BBOX={west},{south},{east},{north}"
        f"&WIDTH={size}&HEIGHT={size}"
        f"&FORMAT=image/png"
    )
    try:
        resp = requests.get(usgs_url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1024 and b"PNG" in resp.content[:16]:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                label = f"NASA Moon Trek / USGS LROC WAC (BBOX: [{south:.3f}°, {north:.3f}°] Lat, [{west:.3f}°, {east:.3f}°] Lon)"
                return img, label
    except Exception:
        pass

    # 2. Local ChandraAlign Server WMS Proxy
    try:
        proxy_url = "http://localhost:8080/api/fetch_wms_tile"
        resp = requests.post(proxy_url, json={"tile_bounds": bounds, "size": size}, timeout=timeout)
        if resp.status_code == 200:
            json_data = resp.json()
            if json_data.get("success") and json_data.get("image_b64"):
                raw_b64 = json_data["image_b64"].split(",")[-1]
                arr = np.frombuffer(base64.b64decode(raw_b64), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                return img, json_data.get("label", "NASA LROC WAC Reference Tile")
    except Exception:
        pass

    return None, "Reference tile unavailable"


def create_side_by_side_comparison(cropped_patch, ref_img, metadata, output_path="ch2_lroc_side_by_side_comparison.png"):
    """
    Renders an annotated mission-control comparison canvas of the cropped Chandrayaan-2 patch
    side-by-side with the matching NASA LROC reference tile.
    """
    canvas_h = 720
    panel_w = 480
    header_h = 70
    footer_h = 50
    total_w = panel_w * 2 + 60
    total_h = canvas_h + header_h + footer_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (11, 15, 25)  # Dark mission control background (#0B0F19)

    # Header
    cv2.putText(canvas, "CHANDRAALIGN // DUAL-VIEWPORT GEOSPATIAL VERIFICATION", (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 2, cv2.LINE_AA)
    coords_text = (f"Target Footprint: Lat [{metadata['target_lat_range'][0]:.3f}deg, {metadata['target_lat_range'][1]:.3f}deg] "
                   f"| Center Lat: {metadata['center_lat']:.3f}deg | Lon: {metadata.get('center_lon', 142.26):.3f}deg")
    cv2.putText(canvas, coords_text, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    # Prepare CH-2 crop for display (preserve aspect ratio)
    c_h, c_w = cropped_patch.shape[:2]
    c_bgr = cv2.cvtColor(cropped_patch, cv2.COLOR_GRAY2BGR) if len(cropped_patch.shape) == 2 else cropped_patch.copy()
    c_scale = min(panel_w / c_w, canvas_h / c_h)
    new_cw, new_ch = int(c_w * c_scale), int(c_h * c_scale)
    c_resized = cv2.resize(c_bgr, (new_cw, new_ch), interpolation=cv2.INTER_AREA)

    x1_offset = 20 + (panel_w - new_cw) // 2
    y1_offset = header_h + (canvas_h - new_ch) // 2
    canvas[y1_offset:y1_offset + new_ch, x1_offset:x1_offset + new_cw] = c_resized
    cv2.rectangle(canvas, (x1_offset - 2, y1_offset - 2), (x1_offset + new_cw + 2, y1_offset + new_ch + 2), (0, 242, 254), 1)

    # Label for CH-2
    tag1 = f"CHANDRAYAAN-2 TMC-2 (Lines [{metadata['y_start']}, {metadata['y_end']}])"
    cv2.putText(canvas, tag1, (20, header_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 254), 1, cv2.LINE_AA)
    dim1 = f"Extracted Shape: {c_w}x{c_h} px"
    cv2.putText(canvas, dim1, (20, header_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)

    # Prepare LROC tile for display
    if ref_img is not None:
        r_h, r_w = ref_img.shape[:2]
        r_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img.copy()
        r_scale = min(panel_w / r_w, canvas_h / r_h)
        new_rw, new_rh = int(r_w * r_scale), int(r_h * r_scale)
        r_resized = cv2.resize(r_bgr, (new_rw, new_rh), interpolation=cv2.INTER_AREA)

        x2_offset = panel_w + 40 + (panel_w - new_rw) // 2
        y2_offset = header_h + (canvas_h - new_rh) // 2
        canvas[y2_offset:y2_offset + new_rh, x2_offset:x2_offset + new_rw] = r_resized
        cv2.rectangle(canvas, (x2_offset - 2, y2_offset - 2), (x2_offset + new_rw + 2, y2_offset + new_rh + 2), (255, 184, 0), 1)

        tag2 = "NASA LROC WAC/NAC REFERENCE BASEMAP"
        cv2.putText(canvas, tag2, (panel_w + 40, header_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 184, 0), 1, cv2.LINE_AA)
        dim2 = f"WMS Tile Resolution: {r_w}x{r_h} px"
        cv2.putText(canvas, dim2, (panel_w + 40, header_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)

    # Footer
    footer_text = f"Linear Track Formula: Y_pixel = ((Lat_north - Lat_target) / (Lat_north - Lat_south)) * Height  |  Output: {os.path.basename(output_path)}"
    cv2.putText(canvas, footer_text, (24, total_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    return output_path


def submit_to_dashboard(cropped_patch, ref_img, metadata, dashboard_url=DEFAULT_DASHBOARD_URL):
    """
    Submits the cropped patch and matching reference tile directly to the running ChandraAlign dashboard.
    """
    print(f"\n[+] Submitting cropped patch & reference to Dashboard at {dashboard_url}...")

    # Base64 encode images
    _, crop_buf = cv2.imencode(".png", cropped_patch)
    src_b64 = "data:image/png;base64," + base64.b64encode(crop_buf).decode("utf-8")

    ref_b64 = None
    if ref_img is not None:
        _, ref_buf = cv2.imencode(".png", ref_img)
        ref_b64 = "data:image/png;base64," + base64.b64encode(ref_buf).decode("utf-8")

    payload = {
        "src_b64": src_b64,
        "ref_b64": ref_b64,
        "ref_label": f"NASA Moon Trek LROC (Lat [{metadata['target_lat_range'][0]}°, {metadata['target_lat_range'][1]}°], Lon {metadata.get('center_lon', 142.26)}°)"
    }

    reg_url = f"{dashboard_url.rstrip('/')}/api/register"
    try:
        res = requests.post(reg_url, json=payload, timeout=30)
        if res.status_code == 200:
            res_json = res.json()
            print("    [OK] Successfully loaded into Mission Control Dashboard!")
            print(f"    [OK] Registration Status : {res_json.get('success')}")
            tel = res_json.get("telemetry", {})
            if tel:
                print(f"    [OK] Sub-Pixel RMSE      : {tel.get('rmse_px')} px")
                print(f"    [OK] Inlier Vector Count : {tel.get('inlier_matches')} / {tel.get('total_matches')}")
                print(f"    [OK] Inlier Ratio        : {tel.get('inlier_ratio')}%")
                print(f"    [OK] Processing Latency  : {tel.get('latency_breakdown_ms', {}).get('total_pipeline')} ms")
            return res_json
        else:
            print(f"    [!] Dashboard returned HTTP {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"    [!] Could not connect to dashboard at {reg_url}: {e}")
    return None


def crop_swath_by_coordinates(
    swath_path=DEFAULT_SWATH_PATH,
    lat_range=DEFAULT_LAT_RANGE,
    center_lat=None,
    center_lon=DEFAULT_CENTER_LON,
    window_height_px=None,
    lat_north=DEFAULT_LAT_NORTH,
    lat_south=DEFAULT_LAT_SOUTH,
    output_path=DEFAULT_OUTPUT_PATH,
    fetch_reference_lroc=True,
    load_to_dashboard=False,
    dashboard_url=DEFAULT_DASHBOARD_URL,
    visualize_side_by_side=True,
    comparison_output="ch2_lroc_side_by_side_comparison.png"
):
    """
    Performs linear interpolation along the orbital flight track to convert lunar latitude
    coordinates into vertical pixel offsets, clips safely, extracts the swath sub-array,
    saves the patch, and optionally fetches matching NASA LROC reference imagery.

    Formula:
        Y_pixel = ((Lat_north - Lat_target) / (Lat_north - Lat_south)) * Height

    Returns:
        cropped_patch (np.ndarray): The extracted image sub-array.
        metadata (dict): {y_start, y_end, height_px, width_px, center_lat, center_lon, ...}
    """
    # 1. Validate and load swath raster
    if not os.path.exists(swath_path):
        alt_paths = [
            swath_path,
            os.path.join("..", swath_path),
            os.path.join(os.getcwd(), swath_path),
            os.path.join("data", "unpacked_swath", "browse", "calibrated", "20260813", "ch2_tmc_nca_20260813T0627378526_b_brw_d18.png")
        ]
        found = None
        for p in alt_paths:
            if os.path.exists(p):
                found = p
                break
        if not found:
            raise FileNotFoundError(f"Chandrayaan-2 swath image not found at: {swath_path}")
        swath_path = found

    img = cv2.imread(swath_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV failed to decode swath image at: {swath_path}")

    height, width = img.shape[:2]
    print("=" * 72)
    print(" CHANDRAALIGN // CHANDRAYAAN-2 ORBITAL SWATH COORDINATE CROPPER")
    print("=" * 72)
    print(f"[+] Swath File       : {os.path.basename(swath_path)}")
    print(f"[+] Swath Dimensions : Width = {width} px, Height = {height} px")
    print(f"[+] Swath Lat Range  : North (Y=0) = {lat_north:.4f}°, South (Y={height}) = {lat_south:.4f}°")

    lat_span = lat_north - lat_south
    if abs(lat_span) < 1e-6:
        raise ValueError("lat_north and lat_south cannot be identical (zero latitude span).")

    # 2. Linear Interpolation along Orbital Track (Y-axis)
    #    Y_pixel = ((Lat_north - Lat_target) / (Lat_north - Lat_south)) * Height
    if lat_range is not None:
        lat_min = min(lat_range[0], lat_range[1])
        lat_max = max(lat_range[0], lat_range[1])
        eff_center_lat = center_lat if center_lat is not None else ((lat_min + lat_max) / 2.0)

        y_max_lat = ((lat_north - lat_max) / lat_span) * height
        y_min_lat = ((lat_north - lat_min) / lat_span) * height

        y_start_raw = min(y_max_lat, y_min_lat)
        y_end_raw = max(y_max_lat, y_min_lat)
    elif center_lat is not None:
        eff_center_lat = center_lat
        y_center_raw = ((lat_north - center_lat) / lat_span) * height
        half_h = (window_height_px or 1024) / 2.0
        y_start_raw = y_center_raw - half_h
        y_end_raw = y_center_raw + half_h

        lat_max = lat_north - (y_start_raw / height) * lat_span
        lat_min = lat_north - (y_end_raw / height) * lat_span
    else:
        raise ValueError("Either lat_range or center_lat must be provided.")

    # 3. Safety Boundary Clipping within [0, Height]
    y_start = max(0, min(height, int(round(y_start_raw))))
    y_end = max(0, min(height, int(round(y_end_raw))))

    # Guarantee non-empty slice
    if y_end <= y_start:
        if y_start >= height:
            y_start = max(0, height - 1)
            y_end = height
        else:
            y_end = min(height, y_start + 1)

    print(f"\n[+] Target Latitude  : [{lat_min:.4f}°, {lat_max:.4f}°] (Center: {eff_center_lat:.4f}°)")
    print(f"[+] Interpolated Y   : Y_start = {y_start_raw:.2f} -> {y_start} px | Y_end = {y_end_raw:.2f} -> {y_end} px")
    print(f"[+] Cropped Span     : {y_end - y_start} vertical lines across full width (X = [0, {width}])")

    # 4. Extract Sub-Array Across Full Swath Width
    cropped_patch = img[y_start:y_end, 0:width].copy()
    assert cropped_patch.size > 0, "Extracted sub-array is empty."

    # 5. Save Cropped Patch
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, cropped_patch)
    print(f"[+] Saved Patch      : {os.path.abspath(output_path)}")

    # 6. Metadata Payload
    metadata = {
        "swath_file": os.path.basename(swath_path),
        "swath_dimensions": {"width": width, "height": height},
        "swath_lat_bounds": {"north": lat_north, "south": lat_south},
        "y_start": int(y_start),
        "y_end": int(y_end),
        "height_px": int(y_end - y_start),
        "width_px": int(cropped_patch.shape[1]),
        "center_lat": round(float(eff_center_lat), 4),
        "center_lon": round(float(center_lon), 4) if center_lon is not None else None,
        "target_lat_range": [round(float(lat_min), 4), round(float(lat_max), 4)],
        "output_path": os.path.abspath(output_path)
    }

    # 7. Optional Reference Tile Retrieval (NASA Moon Trek WMS)
    ref_img = None
    if fetch_reference_lroc and center_lon is not None:
        lon_half_span = 0.75
        tile_bounds = {
            "west": center_lon - lon_half_span,
            "east": center_lon + lon_half_span,
            "south": lat_min,
            "north": lat_max
        }
        print(f"\n[+] Fetching NASA LROC Reference Tile over WMS (Lon: [{tile_bounds['west']:.2f}°, {tile_bounds['east']:.2f}°])...")
        ref_img, ref_label = fetch_lroc_wms_tile(tile_bounds, size=1024)
        if ref_img is not None:
            ref_path = "lroc_reference_matching_tile.png"
            cv2.imwrite(ref_path, ref_img)
            metadata["reference_tile"] = {
                "source": ref_label,
                "shape": list(ref_img.shape),
                "saved_path": os.path.abspath(ref_path),
                "tile_bounds": tile_bounds
            }
            print(f"    [OK] Saved LROC Reference Tile : {os.path.abspath(ref_path)}")
            print(f"    [OK] Reference Source          : {ref_label}")

    # 8. Side-by-Side Visualization
    if visualize_side_by_side:
        vis_path = create_side_by_side_comparison(cropped_patch, ref_img, metadata, comparison_output)
        metadata["comparison_image"] = os.path.abspath(vis_path)
        print(f"[+] Side-by-Side View: {os.path.abspath(vis_path)}")

    # 9. Optional Dashboard Submission
    if load_to_dashboard:
        dashboard_result = submit_to_dashboard(cropped_patch, ref_img, metadata, dashboard_url)
        metadata["dashboard_response"] = dashboard_result

    print("=" * 72)
    return cropped_patch, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Automated Chandrayaan-2 Swath Latitude-to-Pixel Cropper & LROC Reference Matcher"
    )
    parser.add_argument("--swath-path", default=DEFAULT_SWATH_PATH, help="Path to Chandrayaan-2 swath image")
    parser.add_argument("--lat-min", type=float, default=-7.094, help="Lower latitude boundary (e.g. -7.094)")
    parser.add_argument("--lat-max", type=float, default=-5.418, help="Upper latitude boundary (e.g. -5.418)")
    parser.add_argument("--center-lat", type=float, default=-6.256, help="Target center latitude (e.g. -6.256)")
    parser.add_argument("--center-lon", type=float, default=142.260, help="Target center longitude (e.g. 142.260)")
    parser.add_argument("--lat-north", type=float, default=-5.44, help="Swath north latitude at Y=0 (e.g. -5.44)")
    parser.add_argument("--lat-south", type=float, default=-29.48, help="Swath south latitude at Y=Height (e.g. -29.48)")
    parser.add_argument("--output", default="ch2_cropped_matching_patch.png", help="Output filename for cropped patch")
    parser.add_argument("--no-wms", action="store_true", help="Disable automated NASA LROC WMS reference retrieval")
    parser.add_argument("--load-to-dashboard", action="store_true", help="Submit crop directly to active ChandraAlign dashboard")
    parser.add_argument("--dashboard-url", default="http://localhost:8080", help="ChandraAlign dashboard URL")
    parser.add_argument("--no-vis", action="store_true", help="Disable side-by-side comparison image creation")

    args = parser.parse_args()

    crop_swath_by_coordinates(
        swath_path=args.swath_path,
        lat_range=(args.lat_min, args.lat_max),
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        lat_north=args.lat_north,
        lat_south=args.lat_south,
        output_path=args.output,
        fetch_reference_lroc=not args.no_wms,
        load_to_dashboard=args.load_to_dashboard,
        dashboard_url=args.dashboard_url,
        visualize_side_by_side=not args.no_vis
    )


if __name__ == "__main__":
    main()
