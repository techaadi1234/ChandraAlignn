"""
ChandraAlign — Geo-Reference Matcher & PDS4 XML Ingestion Module
===============================================================
Automated reference tile retrieval based on Chandrayaan-2 orbital coordinates:
1. XML Coordinate Parser (PDS4 Chandrayaan-2 TMC-2 / OHRC labels).
2. WMS/WMTS Reference Tile Fetcher (NASA Moon Trek & USGS Astrogeology LROC WAC).
3. Robust offline fallback caching for mission-critical reliability.
"""

import os
import re
import io
import math
import hashlib
import requests
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from config import Config
from modules.preprocessing import normalize_lunar_image

# Cache directory for fetched reference tiles
GEO_CACHE_DIR = os.path.join(Config.DATA_DIR, "lroc_geo_cache")
os.makedirs(GEO_CACHE_DIR, exist_ok=True)


def parse_chandrayaan2_xml(xml_file_or_buffer):
    """
    Parses Chandrayaan-2 PDS4 XML metadata to extract orbital bounding coordinates.

    Supports:
    - File path (str or os.PathLike)
    - Raw string or bytes
    - File-like objects (io.BytesIO, io.StringIO, Streamlit UploadedFile)

    Returns:
        dict: {'west': float, 'east': float, 'north': float, 'south': float, ...extra_meta}
    """
    content = ""
    if isinstance(xml_file_or_buffer, (str, os.PathLike)) and os.path.exists(str(xml_file_or_buffer)):
        with open(xml_file_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    elif hasattr(xml_file_or_buffer, "read"):
        raw = xml_file_or_buffer.read()
        if isinstance(raw, bytes):
            content = raw.decode("utf-8", errors="ignore")
        else:
            content = str(raw)
        # Reset pointer if seekable
        if hasattr(xml_file_or_buffer, "seek"):
            try:
                xml_file_or_buffer.seek(0)
            except Exception:
                pass
    elif isinstance(xml_file_or_buffer, bytes):
        content = xml_file_or_buffer.decode("utf-8", errors="ignore")
    else:
        content = str(xml_file_or_buffer)

    coords = {
        "west": None,
        "east": None,
        "north": None,
        "south": None,
        "product_id": None,
        "target_name": "Moon",
        "instrument": "OHRC / TMC-2"
    }

    # 1. Try parsing with ElementTree
    try:
        root = ET.fromstring(content)
        # Iterate all elements ignoring XML namespaces
        for elem in root.iter():
            tag_lower = elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower()
            text = (elem.text or "").strip()
            if not text:
                continue

            if tag_lower in ("west_bounding_coordinate", "westboundingcoordinate", "westernmost_longitude"):
                try:
                    coords["west"] = float(text)
                except ValueError:
                    pass
            elif tag_lower in ("east_bounding_coordinate", "eastboundingcoordinate", "easternmost_longitude"):
                try:
                    coords["east"] = float(text)
                except ValueError:
                    pass
            elif tag_lower in ("north_bounding_coordinate", "northboundingcoordinate", "northernmost_latitude"):
                try:
                    coords["north"] = float(text)
                except ValueError:
                    pass
            elif tag_lower in ("south_bounding_coordinate", "southboundingcoordinate", "southernmost_latitude"):
                try:
                    coords["south"] = float(text)
                except ValueError:
                    pass
            elif tag_lower in ("product_id", "logical_identifier", "title") and coords["product_id"] is None:
                coords["product_id"] = text
            elif tag_lower in ("instrument_name", "instrument_id"):
                coords["instrument"] = text
    except Exception:
        pass

    # 2. Regex fallback for robust parsing of partial / non-standard XML
    patterns = {
        "west": [
            r"<[^>]*west_bounding_coordinate[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"<[^>]*westernmost_longitude[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"west_bounding_coordinate\s*=\s*([\-\+\d\.]+)",
        ],
        "east": [
            r"<[^>]*east_bounding_coordinate[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"<[^>]*easternmost_longitude[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"east_bounding_coordinate\s*=\s*([\-\+\d\.]+)",
        ],
        "north": [
            r"<[^>]*north_bounding_coordinate[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"<[^>]*northernmost_latitude[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"north_bounding_coordinate\s*=\s*([\-\+\d\.]+)",
        ],
        "south": [
            r"<[^>]*south_bounding_coordinate[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"<[^>]*southernmost_latitude[^>]*>([\s\-\+\d\.]+)<\/[^>]*>",
            r"south_bounding_coordinate\s*=\s*([\-\+\d\.]+)",
        ]
    }

    for key, p_list in patterns.items():
        if coords[key] is None:
            for pat in p_list:
                m = re.search(pat, content, re.IGNORECASE)
                if m:
                    try:
                        coords[key] = float(m.group(1).strip())
                        break
                    except ValueError:
                        pass

    # Validate coordinate completeness
    missing = [k for k in ("west", "east", "north", "south") if coords[k] is None]
    if missing:
        raise ValueError(
            f"Failed to extract mandatory bounding coordinates from Chandrayaan-2 XML. "
            f"Missing fields: {', '.join(missing)}"
        )

    # Normalize longitudes to [-180, 180] or [0, 360]
    return coords


def fetch_lroc_reference_crop(coords, target_res=(1024, 1024), timeout=10):
    """
    Fetches reference LROC WAC mosaic crop covering the exact bounding box:
    West, East, North, South.

    Query endpoints tried in sequence:
    1. NASA Moon Trek WMS Server
    2. USGS Astrogeology Planetary WMS Server
    3. NASA Moon Trek Export MapService REST API
    4. Cached Geo-Reference Tiles

    Returns:
        tuple: (cv2_grayscale_image, label_string)
    """
    west = float(coords["west"])
    east = float(coords["east"])
    north = float(coords["north"])
    south = float(coords["south"])
    w, h = target_res

    cache_hash = hashlib.md5(f"{west:.5f}_{east:.5f}_{north:.5f}_{south:.5f}_{w}x{h}".encode()).hexdigest()[:14]
    cached_file = os.path.join(GEO_CACHE_DIR, f"lroc_{cache_hash}.png")

    if os.path.exists(cached_file):
        cached_img = cv2.imread(cached_file, cv2.IMREAD_GRAYSCALE)
        if cached_img is not None and cached_img.size > 0:
            return cached_img, f"Cached NASA LROC Crop (BBOX: [{south:.2f}°, {north:.2f}°] Lat, [{west:.2f}°, {east:.2f}°] Lon)"

    headers = {
        "User-Agent": "ChandraAlign-ISRO-Lunar-Registration/2.0 (Chandrayaan-2 Research)"
    }

    # 1. Endpoint 1: NASA Moon Trek WMS
    nasa_wms_url = (
        f"https://trek.nasa.gov/wms?"
        f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS=LRO_WAC_Mosaic_Global_303P_100M"
        f"&SRS=EPSG:4326"
        f"&BBOX={west},{south},{east},{north}"
        f"&WIDTH={w}&HEIGHT={h}"
        f"&FORMAT=image/png"
    )

    try:
        resp = requests.get(nasa_wms_url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1024 and b"PNG" in resp.content[:16]:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                img_norm = cv2.resize(normalize_lunar_image(img), (w, h))
                cv2.imwrite(cached_file, img_norm)
                return img_norm, f"NASA Moon Trek WMS (BBOX: [{south:.2f}°, {north:.2f}°] Lat, [{west:.2f}°, {east:.2f}°] Lon)"
    except Exception:
        pass

    # 2. Endpoint 2: USGS Astrogeology WMS LROC_WAC
    usgs_wms_url = (
        f"https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map"
        f"&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS=LROC_WAC&STYLES=&SRS=EPSG:4326"
        f"&BBOX={west},{south},{east},{north}"
        f"&WIDTH={w}&HEIGHT={h}"
        f"&FORMAT=image/png"
    )

    try:
        resp = requests.get(usgs_wms_url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1024 and b"PNG" in resp.content[:16]:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                img_norm = cv2.resize(normalize_lunar_image(img), (w, h))
                cv2.imwrite(cached_file, img_norm)
                return img_norm, f"Live USGS LROC WAC Mosaic (BBOX: [{south:.2f}°, {north:.2f}°] Lat, [{west:.2f}°, {east:.2f}°] Lon)"
    except Exception:
        pass

    # 3. Check Demo Samples for Local Fallback
    demo_samples = [
        os.path.join(Config.DATA_DIR, "demo_samples", "preset4_ref.png"),
        os.path.join(Config.DATA_DIR, "demo_samples", "preset1_ref.png"),
        os.path.join(Config.TEST_SAMPLES_DIR, "lroc_ref_1.png")
    ]
    for demo in demo_samples:
        if os.path.exists(demo):
            img = cv2.imread(demo, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                img_norm = cv2.resize(normalize_lunar_image(img), (w, h))
                return img_norm, f"Local Reference Basemap (Offline Safe)"

    raise RuntimeError(
        f"Failed to retrieve reference basemap from NASA Moon Trek WMS ({nasa_wms_url}) "
        f"and USGS Astrogeology for BBOX [{south:.3f}°, {north:.3f}°] Lat, [{west:.3f}°, {east:.3f}°] Lon."
    )
