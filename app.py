import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
import cv2
import io
import json
import glob
import pandas as pd
from PIL import Image

from config import Config
from modules.preprocessing import normalize_lunar_image, apply_clahe
from modules.ml_crater_detector import load_crater_detector, extract_crater_landmarks
from modules.registration_engine import register_images
from modules.swath_processor import ChandrayaanSwathTiler
from modules.geo_reference_matcher import parse_chandrayaan2_xml, fetch_lroc_reference_crop

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChandraAlign | ISRO Sub-Pixel Lunar Registration",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Deep Space Mission Control Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070B14 !important;
    color: #E2E8F0;
}
#MainMenu, footer, header { visibility: hidden; }

/* Stblock containers */
[data-testid="stAppViewContainer"] { background: #070B14; }
[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #1F2937; }

/* Header */
.mission-header {
    background: linear-gradient(135deg, #0F172A 0%, #111827 60%, #1A2744 100%);
    border: 1px solid #1F2937;
    border-bottom: 3px solid #06B6D4;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(6,182,212,0.12), 0 2px 8px rgba(0,0,0,0.6);
}

/* Panel card */
.panel-card {
    background: #0D1117;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* Source pill status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-left: 0.5rem;
}
.badge-ready { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid #10B981; }
.badge-live  { background: rgba(6,182,212,0.15); color: #06B6D4; border: 1px solid #06B6D4; }
.badge-waiting { background: rgba(100,116,139,0.15); color: #64748B; border: 1px solid #64748B; }
.badge-loading { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid #F59E0B; }

/* KPI metric cards */
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    height: 100%;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 800;
    color: #06B6D4;
}
.kpi-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94A3B8;
    margin-top: 0.2rem;
}
.kpi-target { font-size: 0.75rem; color: #10B981; margin-top: 0.25rem; }

/* Legend row */
.legend-row {
    display: flex; gap: 2rem; align-items: center;
    background: #111827; border: 1px solid #1F2937;
    padding: 0.75rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;
}
.leg-green  { color: #00FF7F; font-weight: 700; }
.leg-yellow { color: #FFD700; font-weight: 700; }
.leg-red    { color: #FF4444; font-weight: 700; }

/* Custom button accent */
div.stButton > button:first-child {
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI crater detection model...")
def get_detector():
    return load_crater_detector()

crater_model = get_detector()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mission-header">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <h1 style="margin:0;font-size:1.9rem;font-weight:800;color:#F8FAFC;">
        🌕 ChandraAlign
        <span style="font-size:1rem;font-weight:400;color:#94A3B8;">
          &nbsp;·&nbsp; Sub-Pixel Lunar Registration Mission Control
        </span>
      </h1>
      <p style="margin:0.4rem 0 0;color:#64748B;font-size:0.9rem;">
        Method A: Automated Geographic Basemap Ingestion · Live NASA Moon Trek WMS · Sub-Pixel MAGSAC++ · Tri-Tier Fusion (SIFT + ResNet-50 + YOLOv5)
      </p>
    </div>
    <div style="text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#64748B;">
      ISRO SIH 2026 &nbsp;·&nbsp; Target: &lt;0.8 px RMSE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Init — STRICTLY LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
st.session_state.mode = "LIVE_ISRO"

for key, default in [
    ("src_img", None), ("source_img", None), ("ref_img", None),
    ("ref_source_label", ""), ("reg_result", None),
    ("swath_tiles", []), ("swath_bounds", None),
    ("selected_tile_idx", 0), ("active_slice_info", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Mission Parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ Mission Parameters")
    st.markdown("---")

    st.markdown("""
    <div style="background: rgba(6,182,212,0.1); border: 1px solid #06B6D4; border-radius: 8px; padding: 10px; margin-bottom: 12px;">
        <span style="color:#06B6D4; font-weight: 700; font-size: 0.85rem;">🛰️ PIPELINE STATUS</span><br/>
        <span style="color:#E2E8F0; font-size: 0.8rem;">Live Chandrayaan-2 Swath Slicing & Automated NASA Moon Trek WMS Retrieval</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Registration Hyperparameters ─────────────────────────────────────────
    st.subheader("⚙️ Registration Hyperparameters")
    magsac_thresh = st.slider("MAGSAC Reprojection Threshold (px)", 1.0, 5.0, 2.0, 0.5)
    target_kps = st.slider("ANMS Target Keypoint Budget", 200, 1200, 800, 50)
    clahe_clip = st.slider("CLAHE Contrast Clip Limit", 1.0, 4.0, 2.5, 0.5)
    conf_thresh = st.slider("YOLOv5 Confidence Threshold", 0.05, 0.50, 0.15, 0.05)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — Live Chandrayaan-2 Ingestion Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown("#### 🛰️ Live Chandrayaan-2 Swath Ingestion & WMS Basemap Retrieval")
st.markdown("<p style='color:#94A3B8;font-size:0.88rem;'>Provide raw Chandrayaan-2 swath products (.zip, .img, .tif, .png, .jpg) and ISRO coordinate grid (.csv / .xml). The pipeline slices standardized 1024×1024 patches and retrieves matching NASA LROC reference basemaps over live WMS.</p>", unsafe_allow_html=True)

up_col1, up_col2 = st.columns([1.5, 1])

with up_col1:
    uploaded_raster = st.file_uploader(
        "Upload Chandrayaan-2 Image / Swath (.zip, .img, .tif, .png, .jpg)",
        type=["zip", "img", "tif", "tiff", "png", "jpg", "jpeg"],
        key="live_raster_upload"
    )

with up_col2:
    uploaded_meta = st.file_uploader(
        "ISRO Coordinate Grid (.csv) or PDS4 Label (.xml, .lbl)",
        type=["csv", "xml", "lbl"],
        key="live_meta_upload"
    )

btn_load = st.button("⚡ Ingest & Wire Swath to Canvas", type="primary", use_container_width=True)

if btn_load or (st.session_state.src_img is None and st.session_state.ref_img is None):
    with st.spinner("📦 Processing Chandrayaan-2 swath and querying NASA Moon Trek WMS..."):
        try:
            raster_path = None
            csv_path = None
            xml_path = None

            # Handle user uploaded files
            if uploaded_raster is not None:
                if uploaded_raster.name.lower().endswith(".zip"):
                    parsed = ChandrayaanSwathTiler.parse_folder_or_zip(uploaded_raster)
                    raster_path = parsed["raster_path"]
                    xml_path = parsed["xml_path"]
                    csv_path = parsed["csv_path"]
                else:
                    save_path = os.path.join(Config.DATA_DIR, f"live_upload_{uploaded_raster.name}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_raster.getbuffer())
                    raster_path = save_path

            if uploaded_meta is not None:
                ext = os.path.splitext(uploaded_meta.name)[1].lower()
                save_path = os.path.join(Config.DATA_DIR, f"live_meta_{uploaded_meta.name}")
                with open(save_path, "wb") as f:
                    f.write(uploaded_meta.getbuffer())
                if ext == ".csv":
                    csv_path = save_path
                elif ext in (".xml", ".lbl"):
                    xml_path = save_path

            # Auto-locate workspace files if none explicitly uploaded
            if raster_path is None:
                browse_cands = glob.glob(os.path.join(Config.DATA_DIR, "**", "ch2_tmc*b_brw*.png"), recursive=True) + \
                               glob.glob(os.path.join(Config.DATA_DIR, "**", "ch2_tmc*b_brw*.jpg"), recursive=True) + \
                               glob.glob(os.path.join(Config.DATA_DIR, "**", "ch2_tmc*d_img*.img"), recursive=True)
                if browse_cands:
                    raster_path = browse_cands[0]

            if csv_path is None:
                csv_cands = glob.glob(os.path.join(Config.DATA_DIR, "**", "ch2_tmc*g_grd*.csv"), recursive=True)
                if csv_cands:
                    csv_path = csv_cands[0]

            if raster_path and os.path.exists(raster_path):
                ext = os.path.splitext(raster_path)[1].lower()
                is_browse = "_b_brw_" in os.path.basename(raster_path).lower() or ext in (".jpg", ".jpeg", ".png")

                if is_browse and ext != ".img":
                    # Directly load browse JPG/PNG image
                    raw_img = cv2.imread(raster_path, cv2.IMREAD_UNCHANGED)
                    if raw_img is not None:
                        h_full, w_full = raw_img.shape[:2]
                        # For Chandrayaan-2 browse products: slice square patch to preserve 1:1 circular geometry
                        crop_w = min(w_full, 1024)
                        crop_h = crop_w
                        if h_full >= 6800 + crop_h:
                            y_start = 6800
                        elif h_full >= crop_h:
                            y_start = 0
                        else:
                            y_start = 0
                        slice_lines = raw_img[y_start : y_start + crop_h, 0 : crop_w]

                        # Convert to grayscale 8-bit uint8 and resize uniformly to 1024x1024
                        if len(slice_lines.shape) == 3:
                            slice_gray = cv2.cvtColor(slice_lines, cv2.COLOR_BGR2GRAY)
                        else:
                            slice_gray = slice_lines

                        slice_1024 = cv2.resize(slice_gray, (1024, 1024), interpolation=cv2.INTER_AREA)
                        slice_norm = apply_clahe(normalize_lunar_image(slice_1024), clip_limit=clahe_clip)

                        st.session_state.src_img = slice_norm
                        st.session_state.source_img = slice_norm
                        st.session_state.active_slice_info = f"Chandrayaan-2 Swath (Square Lines: [{y_start}, {y_start + crop_h}] → 1024×1024)"

                        # Query CSV grid for Scan between 68000 and 78000
                        if csv_path and os.path.exists(csv_path):
                            df_grid = pd.read_csv(csv_path)
                            col_scan = [c for c in df_grid.columns if "scan" in c.lower()]
                            col_lat = [c for c in df_grid.columns if "lat" in c.lower()]
                            col_lon = [c for c in df_grid.columns if "lon" in c.lower() or "long" in c.lower()]

                            if col_scan and col_lat and col_lon:
                                s_col, lat_col, lon_col = col_scan[0], col_lat[0], col_lon[0]
                                sub_df = df_grid[(df_grid[s_col] >= 68000) & (df_grid[s_col] <= 78000)]
                                if not sub_df.empty:
                                    s_lat = float(sub_df[lat_col].min())
                                    n_lat = float(sub_df[lat_col].max())
                                    w_lon = float(sub_df[lon_col].min())
                                    e_lon = float(sub_df[lon_col].max())
                                    bounds = {"south": s_lat, "north": n_lat, "west": w_lon, "east": e_lon}
                                else:
                                    bounds = {"south": -18.1102, "north": -16.4562, "west": 141.3730, "east": 142.2707}
                            else:
                                bounds = {"south": -18.1102, "north": -16.4562, "west": 141.3730, "east": 142.2707}
                        else:
                            bounds = {"south": -18.1102, "north": -16.4562, "west": 141.3730, "east": 142.2707}

                        ref_arr, ref_lbl = ChandrayaanSwathTiler.fetch_lroc_reference(bounds, size=1024)
                        st.session_state.ref_img = ref_arr
                        st.session_state.ref_source_label = ref_lbl
                        st.session_state.swath_bounds = bounds

                else:
                    # Full PDS4 binary swath slicing
                    metadata = ChandrayaanSwathTiler.parse_pds_metadata(xml_path if xml_path else "")
                    csv_tiepoints = ChandrayaanSwathTiler.load_csv_tiepoints(csv_path) if csv_path else None
                    tiles = ChandrayaanSwathTiler.slice_swath_into_tiles(
                        raster_path, metadata=metadata, csv_tiepoints=csv_tiepoints,
                        tile_size=1024, overlap_ratio=0.15, max_tiles=10
                    )
                    st.session_state.swath_tiles = tiles
                    st.session_state.swath_bounds = metadata
                    st.session_state.selected_tile_idx = 0

                    if tiles:
                        cur_t = tiles[0]
                        st.session_state.src_img = cur_t["image"]
                        st.session_state.source_img = cur_t["image"]
                        st.session_state.active_slice_info = cur_t["label"]
                        ref_arr, ref_lbl = ChandrayaanSwathTiler.fetch_lroc_reference(cur_t["tile_bounds"], size=1024)
                        st.session_state.ref_img = ref_arr
                        st.session_state.ref_source_label = ref_lbl

            st.success("✓ Live Chandrayaan-2 swath ingested and NASA WMS reference tile loaded!")
        except Exception as e:
            st.error(f"Ingestion error: {e}")

# Multi-patch selector if full swath sliced
if st.session_state.swath_tiles:
    tiles = st.session_state.swath_tiles
    patch_labels = [f"{t['label']} — BBOX: Lat [{t['tile_bounds']['south']:.3f}°, {t['tile_bounds']['north']:.3f}°]" for t in tiles]
    sel_idx = st.selectbox("Select Swath Patch to Register:", range(len(tiles)), format_func=lambda i: patch_labels[i], index=st.session_state.selected_tile_idx)
    if sel_idx != st.session_state.selected_tile_idx:
        st.session_state.selected_tile_idx = sel_idx
        cur_tile = tiles[sel_idx]
        st.session_state.src_img = cur_tile["image"]
        st.session_state.source_img = cur_tile["image"]
        st.session_state.active_slice_info = cur_tile["label"]
        with st.spinner("🛰️ Querying NASA Moon Trek WMS for exact matching reference tile..."):
            ref_tile, ref_lbl = ChandrayaanSwathTiler.fetch_lroc_reference(cur_tile["tile_bounds"], size=1024)
            st.session_state.ref_img = ref_tile
            st.session_state.ref_source_label = ref_lbl

# ─────────────────────────────────────────────────────────────────────────────
# VIEWPORT WINDOWS: Direct Real Image Rendering (Zero Mock Drawing)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.src_img is not None and st.session_state.ref_img is not None:
    gb = st.session_state.swath_bounds or {}
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #06B6D4;border-radius:8px;padding:12px 16px;margin:12px 0;font-size:0.85rem;">
        <span style="color:#06B6D4;font-weight:700;">📡 Live Geographic Footprint:</span><br/>
        <span style="font-family:'JetBrains Mono',monospace;color:#F8FAFC;">
            • Active Patch: {st.session_state.active_slice_info or 'CH-2 Swath (Lines: [6800, 7824])'}<br/>
            • Bounding Box: Lat [{gb.get('south', -18.110):.4f}°, {gb.get('north', -16.456):.4f}°] | Lon: [{gb.get('west', 141.373):.4f}°, {gb.get('east', 142.271):.4f}°]
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_pv1, col_pv2 = st.columns(2)
    with col_pv1:
        st.markdown("##### CH-2 PATCH 1 <span class='badge badge-ready'>✓ LIVE ISRO READY</span>", unsafe_allow_html=True)
        st.image(st.session_state.src_img, caption="Chandrayaan-2 Optical Swath (1024×1024)", use_container_width=True)

    with col_pv2:
        st.markdown("##### LROC WMS TILE <span class='badge badge-live'>✓ NASA WMS BASEMAP</span>", unsafe_allow_html=True)
        st.image(st.session_state.ref_img, caption=st.session_state.ref_source_label or "NASA Moon Trek / USGS LROC WAC", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Registration Execution Button
# ─────────────────────────────────────────────────────────────────────────────
both_ready = st.session_state.src_img is not None and st.session_state.ref_img is not None

st.markdown("---")

if not both_ready:
    st.info("👆 Click **Ingest & Wire Swath to Canvas** above to load the live Chandrayaan-2 swath and NASA WMS reference.")

run_btn = st.button(
    "🚀 Execute Sub-Pixel Tri-Tier Registration",
    use_container_width=True,
    type="primary",
    disabled=not both_ready,
    key="run_registration"
)

if run_btn and both_ready:
    src_for_reg = st.session_state.src_img
    ref_for_reg = st.session_state.ref_img
    if ref_for_reg.shape != src_for_reg.shape:
        ref_for_reg = cv2.resize(ref_for_reg, (src_for_reg.shape[1], src_for_reg.shape[0]))

    with st.spinner("⚙️ Running 3-Tier Landmark Fusion · MAGSAC++ Homography · Sub-Pixel Optimization..."):
        result = register_images(
            src_for_reg, ref_for_reg,
            crater_model=crater_model,
            method="SIFT",
            target_kps=target_kps,
            magsac_thresh=magsac_thresh,
            enable_subpixel=True,
            conf_thresh=conf_thresh,
            min_radius=2,
        )
        st.session_state.reg_result = result
        if result.get("is_registered", True):
            st.success("Registration complete!")
        else:
            st.error(result.get("error_message", "REGISTRATION FAILED: NO OVERLAPPING TERRAIN DETECTED"))

# ─────────────────────────────────────────────────────────────────────────────
# Results — 4-Tab Mission Control Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.reg_result is not None:
    res = st.session_state.reg_result

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Tab 1 · AI Crater Landmark Extraction",
        "🎯 Tab 2 · Correspondence & Uniformity",
        "🔍 Tab 3 · Registration Inspection",
        "📊 Tab 4 · Scientific Metrics & Export",
    ])

    # ─── Helper: draw ellipse overlays with 3-color legend ──────────────────
    def draw_ellipses(base_gray, craters):
        vis = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR) if len(base_gray.shape) == 2 else base_gray.copy()
        for c in craters:
            x_c, y_c = c["centroid"]
            sm, sn = c["semi_major"], c["semi_minor"]
            ang = c["angle"]
            diam = c["diameter"]
            src_type = c.get("layer_source", "YOLOv5")
            if diam > 25.0 or src_type == "ResNet-50":
                color, th = (0, 0, 255), 2        # 🔴 Red  > 25px basins
            elif diam >= 12.0:
                color, th = (0, 215, 255), 2      # 🟡 Yellow  12-25px
            else:
                color, th = (0, 255, 127), 1      # 🟢 Green  < 12px micro
            ctr = (int(round(x_c)), int(round(y_c)))
            axes = (int(max(2, round(sm))), int(max(2, round(sn))))
            cv2.ellipse(vis, ctr, axes, int(round(ang)), 0, 360, color, th, cv2.LINE_AA)
            cv2.circle(vis, ctr, 2, color, -1)
        return vis

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — AI Crater Landmark Extraction
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("3-Tier AI Crater Landmark Extraction & Feature Categorization")
        st.markdown("""
        <div class="legend-row">
            <b>Color Legend:</b>
            <span class="leg-green">🟢 &lt;12px Micro-craters (SIFT Tier 1)</span>
            <span class="leg-yellow">🟡 12–25px Craters (YOLOv5 Tier 3)</span>
            <span class="leg-red">🔴 &gt;25px Basins (ResNet-50 Tier 2)</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Source — Chandrayaan-2 Detected Landmarks")
            vis_src = draw_ellipses(res["src_img"], res["src_craters"])
            st.image(vis_src, channels="BGR", use_container_width=True)
        with c2:
            st.markdown("##### Reference — NASA LROC Detected Landmarks")
            vis_ref = draw_ellipses(res["ref_img"], res["ref_craters"])
            st.image(vis_ref, channels="BGR", use_container_width=True)

        st.markdown("---")
        st.markdown("##### Detection Summary Table")

        def crater_summary(craters, label):
            micro = [c for c in craters if c["diameter"] < 12]
            med   = [c for c in craters if 12 <= c["diameter"] <= 25]
            large = [c for c in craters if c["diameter"] > 25 or c.get("layer_source") == "ResNet-50"]
            return {
                "Image": label,
                "🟢 Micro (<12px)": len(micro),
                "🟡 Medium (12-25px)": len(med),
                "🔴 Large/Basin (>25px)": len(large),
                "Total": len(craters),
                "Avg Confidence": f"{np.mean([c['confidence'] for c in craters]):.3f}" if craters else "—"
            }

        df = pd.DataFrame([
            crater_summary(res["src_craters"], "Source (Chandrayaan-2)"),
            crater_summary(res["ref_craters"], "Reference (NASA LROC)"),
        ])
        st.dataframe(df.set_index("Image"), use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Correspondence & Uniformity
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Correspondence & 10×10 ANMS Spatial Uniformity")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Candidate Matches (grey) vs. MAGSAC Inliers (cyan)")
            h1, w1 = res["src_img"].shape[:2]
            h2, w2 = res["ref_img"].shape[:2]
            canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = cv2.cvtColor(res["src_img"], cv2.COLOR_GRAY2BGR)
            canvas[:h2, w1:] = cv2.cvtColor(res["ref_img"], cv2.COLOR_GRAY2BGR)

            for pt1, pt2 in zip(res["raw_src_pts"], res["raw_ref_pts"]):
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]) + w1, int(pt2[1]))
                cv2.line(canvas, p1, p2, (60, 70, 90), 1, cv2.LINE_AA)
            for pt1, pt2 in zip(res["inlier_src_pts"], res["inlier_ref_pts"]):
                p1 = (int(pt1[0]), int(pt1[1]))
                p2 = (int(pt2[0]) + w1, int(pt2[1]))
                cv2.line(canvas, p1, p2, (0, 229, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, (0, 200, 255), -1)
                cv2.circle(canvas, p2, 3, (0, 200, 255), -1)
            st.image(canvas, channels="BGR", use_container_width=True,
                     caption=f"Total: {res['raw_matches']} → Inliers: {res['inliers_count']} ({res['inlier_ratio']:.1f}%)")

        with c2:
            import plotly.express as px
            st.markdown(f"##### 10×10 ANMS Spatial Density Heatmap  (Uniformity: {res['uniformity_score']:.4f})")
            density = res["spatial_density"]
            fig = px.imshow(
                density,
                labels=dict(x="Grid Col", y="Grid Row", color="Inlier Count"),
                color_continuous_scale="Viridis",
                title=f"Spatial Dispersion (Uniformity Index: {res['uniformity_score']:.4f})",
            )
            fig.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font=dict(color="#E2E8F0"), margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — Registration Inspection
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Sub-Pixel Alignment & Continuity Inspection")

        inspect_mode = st.radio(
            "Select Visualizer:",
            ["✂️ Split-Screen Curtain Wipe", "⬛ Checkerboard Grid"],
            horizontal=True
        )

        warped = res["warped_src"]
        ref_g = res["ref_img"]

        if inspect_mode == "✂️ Split-Screen Curtain Wipe":
            wipe_pct = st.slider("Wipe Position (%)", 0, 100, 50)
            split_x = int(wipe_pct / 100.0 * ref_g.shape[1])
            wipe_vis = cv2.cvtColor(ref_g, cv2.COLOR_GRAY2BGR)
            warp_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            wipe_vis[:, :split_x] = warp_bgr[:, :split_x]
            cv2.line(wipe_vis, (split_x, 0), (split_x, ref_g.shape[0]), (0, 229, 255), 2)
            st.image(wipe_vis, channels="BGR", use_container_width=True,
                     caption=f"Left: Warped Source ({wipe_pct}%) | Right: Reference ({100-wipe_pct}%)")
        else:
            n_sq = st.slider("Grid Squares", 2, 12, 6, step=2)
            h, w = ref_g.shape[:2]
            sq_h, sq_w = h // n_sq, w // n_sq
            chk = np.zeros((h, w), dtype=np.uint8)
            for r in range(n_sq):
                for c in range(n_sq):
                    y1 = r * sq_h; y2 = (r+1)*sq_h if r < n_sq-1 else h
                    x1 = c * sq_w; x2 = (c+1)*sq_w if c < n_sq-1 else w
                    chk[y1:y2, x1:x2] = warped[y1:y2, x1:x2] if (r+c)%2==0 else ref_g[y1:y2, x1:x2]
            st.image(chk, use_container_width=True, caption=f"{n_sq}×{n_sq} Checkerboard Inspection")

        st.markdown("---")
        st.markdown("##### Pixel Residual Difference Heatmap")
        diff = cv2.absdiff(warped, ref_g)
        diff_heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        st.image(diff_heat, channels="BGR", use_container_width=True,
                 caption="Thermal Error Map — Blue: 0 px offset | Red: High residual")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Scientific Metrics & Export
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Quantitative Metric Evaluation Suite")

        k1, k2, k3, k4 = st.columns(4)
        rmse_color = "#10B981" if res["rmse"] < 0.8 else "#F59E0B"
        inlier_color = "#10B981" if res["inlier_ratio"] > 80 else "#F59E0B"
        unif_color = "#10B981" if res["uniformity_score"] > 0.85 else "#F59E0B"
        ssim_color = "#10B981" if res["ssim"] > 0.85 else "#F59E0B"

        for col, val, label, target, color in [
            (k1, f"{res['rmse']:.3f} px", "Sub-Pixel RMSE", "Target: < 0.80 px", rmse_color),
            (k2, f"{res['inlier_ratio']:.1f}%", "MAGSAC Inlier Ratio",
             f"{res['inliers_count']} / {res['raw_matches']}", inlier_color),
            (k3, f"{res['uniformity_score']:.4f}", "10×10 Uniformity", "Chi-Square Index", unif_color),
            (k4, f"{res['ssim']:.4f}", "SSIM Correlation", "Structural Similarity", ssim_color),
        ]:
            col.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-val" style="color:{color};">{val}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-target">{target}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Reference Basemap Source")
        st.code(st.session_state.ref_source_label or "NASA Moon Trek / USGS LROC WAC", language=None)

        st.markdown("---")
        st.markdown("##### Download Alignment Products")

        ex1, ex2 = st.columns(2)

        with ex1:
            report = {
                "mission": "Chandrayaan-2 Lunar Co-Registration",
                "mode": st.session_state.mode,
                "slice_info": st.session_state.active_slice_info,
                "reference_source": st.session_state.ref_source_label,
                "geographic_bounds": st.session_state.swath_bounds,
                "subpixel_rmse_px": round(res["rmse"], 6),
                "mae_px": round(res.get("mae", 0.0), 6),
                "median_error_px": round(res.get("median_subpixel_error", 0.0), 6),
                "inlier_ratio_pct": round(res["inlier_ratio"], 4),
                "inliers_count": res["inliers_count"],
                "total_matches": res["raw_matches"],
                "spatial_uniformity_index": round(res["uniformity_score"], 6),
                "ssim": round(res["ssim"], 6),
                "homography_matrix": res["H_matrix"].tolist(),
            }
            st.download_button(
                "📄 Export JSON Alignment Report",
                json.dumps(report, indent=2).encode(),
                "chandrayaan2_alignment_report.json",
                "application/json",
                use_container_width=True
            )

        with ex2:
            ok, buf = cv2.imencode(".png", res["warped_src"])
            if ok:
                st.download_button(
                    "💾 Export Registered Image (PNG/GeoTIFF)",
                    buf.tobytes(),
                    "chandrayaan2_registered.png",
                    "image/png",
                    use_container_width=True
                )
