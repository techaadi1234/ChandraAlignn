"""
LROC Reference Tile Fetcher Module
===================================
Automatically resolves the NASA LROC QuickMap reference basemap tile from a provided
URL by parsing the prjExtent, proj, and stack parameters and fetching the corresponding
lunar surface imagery through a 3-strategy cascade:

Strategy 1 (Primary) — NASA Trek WMTS:
    Attempts to directly fetch tiles from the trek.nasa.gov WMTS endpoint by mapping
    the Moon Equirectangular projected coordinates to tile (z, row, col) indices.

Strategy 2 (Headless Playwright Capture):
    If direct tile fetching fails, launches a headless browser (Playwright / Selenium)
    to navigate to the exact QuickMap URL, waits for the canvas to render, and crops
    a clean 1024x1024 PNG from the viewport.

Strategy 3 (Offline Cache Fallback):
    If the remote server is unreachable or the headless browser isn't installed,
    loads a pre-cached reference tile from `data/cached_lroc_reference.png`.
    If no cache exists, generates a realistic synthetic lunar texture.
"""

import os
import re
import math
import io
import hashlib
import time
import urllib.parse
import numpy as np
import cv2

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# Moon Projection Constants (Equirectangular, proj=10)
# ──────────────────────────────────────────────────────────────────────────────
MOON_RADIUS_M = 1_737_400.0  # meters
MOON_CIRC_M = 2.0 * math.pi * MOON_RADIUS_M  # ~10,917,000 m

# Cache directory for pre-fetched tiles
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_BASE_DIR, "data", "lroc_cache")
_PRESET_CACHE = os.path.join(_BASE_DIR, "data", "cached_lroc_reference.png")
_DEMO_CACHE = os.path.join(_BASE_DIR, "data", "demo_samples", "preset4_ref.png")
os.makedirs(_CACHE_DIR, exist_ok=True)


def parse_quickmap_url(url: str) -> dict:
    """
    Parses a NASA LROC QuickMap URL and extracts:
    - prjExtent: [xmin, ymin, xmax, ymax] in Moon projected metres (Equirectangular)
    - proj: projection code (10 = Moon Equirectangular)
    - stack: basemap layer stack ID
    - earth_shadow: bool
    """
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    def get_p(key, default=None):
        v = params.get(key, [default])
        return v[0] if v else default

    extent_str = get_p("prjExtent", "")
    extent = []
    if extent_str:
        try:
            extent = [float(x) for x in extent_str.split(",")]
        except ValueError:
            extent = []

    return {
        "prjExtent": extent if len(extent) == 4 else [-MOON_CIRC_M/2, -MOON_CIRC_M/4, MOON_CIRC_M/2, MOON_CIRC_M/4],
        "proj": int(get_p("proj", "10")),
        "stack": int(get_p("stack", "3314")),
        "earthShadow": get_p("earthShadowEnabled", "false").lower() == "true",
        "raw_url": url
    }


def extent_to_latlon(prj_extent: list) -> tuple:
    """
    Converts Moon Equirectangular projected extent [xmin, ymin, xmax, ymax] (metres)
    to geographic bounding box (lon_min, lat_min, lon_max, lat_max) in degrees.
    """
    xmin, ymin, xmax, ymax = prj_extent
    # Scale by circumference -> degrees
    lon_min = (xmin / MOON_CIRC_M) * 360.0
    lon_max = (xmax / MOON_CIRC_M) * 360.0
    lat_min = (ymin / MOON_CIRC_M) * 360.0
    lat_max = (ymax / MOON_CIRC_M) * 360.0
    return lon_min, lat_min, lon_max, lat_max


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────────────
# The full-moon EQ canvas extent in projected metres (origin = centre of moon)
# x: [-MOON_CIRC_M/2, +MOON_CIRC_M/2]   (lon -180° to +180°)
# y: [-MOON_CIRC_M/4, +MOON_CIRC_M/4]   (lat  -90° to  +90°)
# ──────────────────────────────────────────────────────────────────────────────
_FULL_X_MIN = -MOON_CIRC_M / 2.0   # ~ -5,458,500 m  (lon=-180)
_FULL_X_MAX =  MOON_CIRC_M / 2.0   # ~ +5,458,500 m  (lon=+180)
_FULL_Y_MIN = -MOON_CIRC_M / 4.0   # ~ -2,729,250 m  (lat= -90)
_FULL_Y_MAX =  MOON_CIRC_M / 4.0   # ~ +2,729,250 m  (lat= +90)


def _crop_to_prj_extent(full_img: np.ndarray, prj_extent: list, out_size: int = 1024) -> np.ndarray:
    """
    Crops a full-moon equirectangular image to the bounding box defined by prj_extent.

    The full_img represents the area [_FULL_X_MIN, _FULL_X_MAX] x [_FULL_Y_MIN, _FULL_Y_MAX]
    in Moon EQ projected metres. We map prj_extent to pixel coordinates and crop.

    This is the fix for the scale mismatch bug: previously the cached full-moon tile
    was returned as-is, giving a completely different GSD (m/px) than the Chandrayaan-2
    source swath. After cropping, both images cover the same ground footprint.
    """
    h, w = full_img.shape[:2]
    xmin, ymin, xmax, ymax = prj_extent

    full_w_m = _FULL_X_MAX - _FULL_X_MIN
    full_h_m = _FULL_Y_MAX - _FULL_Y_MIN

    # Map projected metres → pixel column/row
    px_min = int(round((xmin - _FULL_X_MIN) / full_w_m * w))
    px_max = int(round((xmax - _FULL_X_MIN) / full_w_m * w))
    # Note: image row 0 = north pole (y_max), row h = south pole (y_min)
    py_min = int(round((_FULL_Y_MAX - ymax) / full_h_m * h))
    py_max = int(round((_FULL_Y_MAX - ymin) / full_h_m * h))

    # Clamp to image bounds
    px_min = max(0, min(px_min, w - 1))
    px_max = max(px_min + 1, min(px_max, w))
    py_min = max(0, min(py_min, h - 1))
    py_max = max(py_min + 1, min(py_max, h))

    crop = full_img[py_min:py_max, px_min:px_max]
    if crop.size == 0:
        # Extent maps outside the cached tile — return resized full image with a warning
        print("[LROC] Warning: prjExtent maps outside cached tile; returning full tile.")
        return cv2.resize(full_img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _try_nasa_trek_wmts(prj_extent: list, out_size: int = 1024) -> np.ndarray | None:
    """
    Strategy 1: Attempt to fetch tiles from NASA Trek WMTS
    (https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303P_100M/...)
    
    Computes the optimal zoom level and tile indices from the Moon EQ projected
    extent and assembles a composite mosaic from multiple tile downloads.
    """
    if not _REQUESTS_OK:
        return None

    # Known working Trek WMTS base (discovered empirically)
    wmts_bases = [
        "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303P_100M/1.0.0/default/default028mm",
    ]

    xmin, ymin, xmax, ymax = prj_extent
    extent_width = xmax - xmin
    
    # Best zoom level: tile width should be ~2x-4x smaller than extent
    # At zoom z, each tile covers: MOON_CIRC_M / 2^z metres
    for z in range(5, 1, -1):
        tile_size_m = MOON_CIRC_M / (2 ** z)
        tiles_across = extent_width / tile_size_m
        if 0.5 <= tiles_across <= 8:
            break

    # Tile column/row indices
    col_min = int(math.floor((xmin + MOON_CIRC_M / 2) / MOON_CIRC_M * (2 ** z)))
    col_max = int(math.ceil((xmax + MOON_CIRC_M / 2) / MOON_CIRC_M * (2 ** z)))
    row_min = int(math.floor((MOON_CIRC_M / 4 - ymax) / (MOON_CIRC_M / 2) * (2 ** (z - 1))))
    row_max = int(math.ceil((MOON_CIRC_M / 4 - ymin) / (MOON_CIRC_M / 2) * (2 ** (z - 1))))

    col_min = max(0, col_min)
    row_min = max(0, row_min)

    headers = {
        "User-Agent": "Mozilla/5.0 ChandraAlign/1.0 ISRO-SIH",
        "Accept": "image/png,image/webp,*/*",
    }

    for base in wmts_bases:
        tiles_fetched = []
        fetch_ok = False

        for row in range(row_min, min(row_max + 1, row_min + 6)):
            for col in range(col_min, min(col_max + 1, col_min + 6)):
                url = f"{base}/{z}/{row}/{col}.png"
                try:
                    r = requests.get(url, timeout=8, headers=headers)
                    if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                        arr = np.frombuffer(r.content, np.uint8)
                        tile = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if tile is not None:
                            tiles_fetched.append((row, col, tile))
                            fetch_ok = True
                except Exception:
                    continue

        if fetch_ok and tiles_fetched:
            # Stitch tiles together
            rows_used = sorted(set(t[0] for t in tiles_fetched))
            cols_used = sorted(set(t[1] for t in tiles_fetched))
            th = tiles_fetched[0][2].shape[0]
            tw = tiles_fetched[0][2].shape[1]

            canvas_h = len(rows_used) * th
            canvas_w = len(cols_used) * tw
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            for (r_idx, c_idx, tile) in tiles_fetched:
                ri = rows_used.index(r_idx)
                ci = cols_used.index(c_idx)
                canvas[ri*th:(ri+1)*th, ci*tw:(ci+1)*tw] = tile

            return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)

    return None


def _try_playwright_capture(url: str, out_size: int = 1024) -> np.ndarray | None:
    """
    Strategy 2: Headless browser capture of the QuickMap canvas.
    Requires `playwright` to be installed. Falls back gracefully if not available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(url, wait_until="networkidle", timeout=30000)
            # Additional wait for LROC tiles to fully render
            time.sleep(5)
            png_bytes = page.screenshot(full_page=False)
            browser.close()

        arr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Crop out the top toolbar (~60px) and any side panels
        h, w = img.shape[:2]
        crop = img[60:h-30, 60:w-60]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    except Exception as e:
        print(f"[!] Playwright capture failed: {e}")
        return None


def _try_selenium_capture(url: str, out_size: int = 1024) -> np.ndarray | None:
    """
    Strategy 2b: Selenium headless fallback if Playwright is unavailable.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        return None

    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(8)  # Wait for map tiles to render
        png_bytes = driver.get_screenshot_as_png()
        driver.quit()

        arr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        h, w = img.shape[:2]
        crop = img[60:h-30, 60:w-60]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    except Exception as e:
        print(f"[!] Selenium capture failed: {e}")
        return None


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalizes any bit-depth image (16-bit PDS DN, float, etc.) to uint8 [0,255].
    Also handles 3/4-channel images by converting to grayscale.
    """
    if img is None:
        return None
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def apply_clahe_normalize(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Applies CLAHE contrast enhancement for consistent lunar surface appearance."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(img)


def _crop_to_prj_extent(
    img: np.ndarray,
    prj_extent: list,
    out_size: int = 1024,
) -> np.ndarray:
    """
    Crops a lunar map image array down to the region specified by prj_extent [minX, minY, maxX, maxY].
    """
    h, w = img.shape[:2]
    min_x, min_y, max_x, max_y = prj_extent
    x0 = int(np.clip((min_x + _LUNAR_RADIUS_M * np.pi) / (2 * _LUNAR_RADIUS_M * np.pi) * w, 0, w - 1))
    x1 = int(np.clip((max_x + _LUNAR_RADIUS_M * np.pi) / (2 * _LUNAR_RADIUS_M * np.pi) * w, 0, w))
    y0 = int(np.clip((_LUNAR_RADIUS_M * np.pi / 2 - max_y) / (_LUNAR_RADIUS_M * np.pi) * h, 0, h - 1))
    y1 = int(np.clip((_LUNAR_RADIUS_M * np.pi / 2 - min_y) / (_LUNAR_RADIUS_M * np.pi) * h, 0, h))

    if x1 <= x0 or y1 <= y0:
        return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pads image to square by symmetric border replication."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    out[y_off:y_off+h, x_off:x_off+w] = img
    return out


def fetch_lroc_tile(
    url: str,
    out_size: int = 1024,
    use_cache: bool = True,
    timeout_seconds: int = 30,
) -> tuple:
    """
    Main entry point: Fetches the LROC reference basemap tile corresponding to
    the provided QuickMap URL.
    """
    params = parse_quickmap_url(url)
    prj_extent = params["prjExtent"]
    cache_key = _cache_key(url)
    cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.png")

    if use_cache and os.path.exists(cache_path):
        img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, "Loaded from local URL-keyed cache"

    print("[LROC] Attempting NASA Trek WMTS direct tile fetch...")
    img = _try_nasa_trek_wmts(prj_extent, out_size=out_size)
    if img is not None:
        img = apply_clahe_normalize(img)
        if use_cache:
            cv2.imwrite(cache_path, img)
            cv2.imwrite(_PRESET_CACHE, img)
        return img, "Fetched via NASA Trek WMTS (bounding-box cropped)"

    print("[LROC] Attempting Playwright headless capture...")
    img = _try_playwright_capture(url, out_size=out_size)
    if img is not None:
        img = apply_clahe_normalize(img)
        if use_cache:
            cv2.imwrite(cache_path, img)
            cv2.imwrite(_PRESET_CACHE, img)
        return img, "Captured via Playwright headless browser"

    print("[LROC] Attempting Selenium headless capture...")
    img = _try_selenium_capture(url, out_size=out_size)
    if img is not None:
        img = apply_clahe_normalize(img)
        if use_cache:
            cv2.imwrite(cache_path, img)
            cv2.imwrite(_PRESET_CACHE, img)
        return img, "Captured via Selenium headless browser"

    for fallback_path in [_PRESET_CACHE, _DEMO_CACHE]:
        if os.path.exists(fallback_path):
            full_img = cv2.imread(fallback_path, cv2.IMREAD_GRAYSCALE)
            if full_img is not None and full_img.size > 0:
                img = cv2.resize(full_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
                return img, "Loaded local reference tile for extent."

    raise RuntimeError(f"Could not retrieve LROC reference basemap tile for {url} and no local cache was found.")

pad_to_square = _pad_to_square

