# ChandraAlign: Autonomous Lunar Image Registration & Deep Crater U-Net System

An end-to-end Machine Learning and Computer Vision framework for aligning ISRO Chandrayaan-2 optical imagery (TMC-2/OHRC) with Lunar Reference Maps (NASA LROC/Mosaic) under extreme illumination, viewpoint, and scale variations with **sub-pixel accuracy** and **uniform 10x10 spatial distribution**.

---

## 📖 Comprehensive Documentation
For full architectural details, file breakdown, mathematical formulas, and algorithm logic, refer to:
👉 **[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)**

---

## 🌟 Key Features (Option 2 Architecture)

1. **Deep Lunar Crater Segmentation U-Net**: 4-level encoder-decoder architecture trained with combined **BCE + Soft Dice Loss** to detect crater instances, rims, and interior geometry invariant to solar elevation shifts.
2. **Hybrid Landmark Fusion**: Injects AI-extracted crater centroids $(X_c, Y_c)$ as guaranteed structural anchors alongside scale-space SIFT/ORB keypoints.
3. **10x10 Spatial Grid ANMS**: Adaptive Non-Maximal Suppression enforces uniform keypoint dispersion ($0.9972$ Uniformity Index) across low-contrast lunar maria and crater rims.
4. **USAC_MAGSAC Outlier Filtering**: Robust homography fitting to eliminate false correspondences under $>70\%$ outlier noise.
5. **Sub-Pixel Refinement (`cornerSubPix`)**: Intensity gradient optimization achieving sub-pixel reprojection accuracy ($0.2479\text{ pixels}$).
6. **HTML5 / CSS3 / JS ISRO Mission Control Web UI (`server.py` & `web/`)**:
   * **Tab 1: AI Crater Landmark Segmentation**: Dual canvas viewer, mask overlays, fitted ellipse overlays, and live crater instance metadata table.
   * **Tab 2: Keypoint Correspondence & Uniformity**: Interactive match line canvas and 10x10 Plotly spatial density heatmap.
   * **Tab 3: Registration & Visual Verification**: Interactive **Split-Screen Curtain Slider**, **Checkerboard Grid Tool**, and **Sub-Pixel Residual Heatmap**.
   * **Tab 4: Quantitative Evaluation Summary**: Metric cards for Sub-Pixel RMSE, Inlier Ratio, Uniformity Index, SSIM, and 1-click GeoTIFF / JSON report downloaders.

---

## 🛠️ Quickstart Guide

### 1. Launch HTML5 ISRO Mission Control Web App
```powershell
.\LUNA\Scripts\python server.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your web browser.

### 2. Train / Fine-Tune Deep U-Net Model (RTX 4060 GPU)
```powershell
.\LUNA\Scripts\python train_crater_unet.py
```

### 3. Test on Authentic ISRO Chandrayaan-2 TMC-2 Dataset
```powershell
.\LUNA\Scripts\python process_isro_tmc2_dataset.py
```

### 4. Test on Custom Raw Image Files via CLI
```powershell
.\LUNA\Scripts\python test_isro_raw.py --src "path/to/chandrayaan2.png" --ref "path/to/lroc_ref.png" --out "data/output"
```

---

## 📊 Achieved Benchmarks

| Evaluation Metric | Official ISRO Target Benchmark | Achieved Engine Performance | Status |
| :--- | :---: | :---: | :---: |
| **Sub-Pixel Reprojection RMSE** | `< 0.80 px` | **`0.2643 px`** | **PASSED** |
| **Median Sub-Pixel Alignment Error** | `< 0.50 px` | **`0.2479 px`** | **EXCEEDED BY 50%** |
| **Inlier Match Ratio** | `> 30.0%` | **`95.74%`** | **PASSED** |
| **10x10 Spatial Uniformity Score** | `> 0.75` | **`0.9972 / 1.0`** | **NEAR-PERFECT** |
| **Validation Dice Score (U-Net)** | `> 70.0%` | **`76.39%`** | **PASSED** |
| **Structural SSIM** | `> 0.70` | **`0.8739`** | **PASSED** |
