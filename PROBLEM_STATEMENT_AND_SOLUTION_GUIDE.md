# Chandrayaan-2 Lunar Image Registration Solution & Hackathon Guide

## 🛰️ Recommended Datasets for Hackathon & Production

For Chandrayaan-2 lunar image registration, you should use official planetary datasets from ISRO and NASA PDS, alongside our synthetic benchmark generator.

### 1. Official Chandrayaan-2 Datasets (Moving / Source Images)
* **ISRO PRADAN Portal**: [https://pradan.issdc.gov.in/pradan/](https://pradan.issdc.gov.in/pradan/)
  * **OHRC (Orbiter High Resolution Camera)**: Ultra-high spatial resolution (~0.25 m/pixel). Outstanding for crater rim matching and landing site verification.
  * **TMC-2 (Terrain Mapping Camera-2)**: 5 m/pixel spatial resolution stereo optical imagery covering large swaths of the lunar surface.
  * **Format**: Standard PDS3/PDS4 formats (.IMG, .TIF, GeoTIFF).

### 2. Reference Lunar Maps (Fixed / Target Images)
* **NASA LRO (Lunar Reconnaissance Orbiter) NAC/WAC**:
  * **LRO WAC Global Basemap**: 100 m/pixel global mosaic (USGS / NASA PDS).
  * **LRO NAC Tiles**: 0.5–2 m/pixel sub-meter resolution tiles for specific polar or equatorial targets.
  * **Download Hub**: [USGS Astrogeology PDS Node](https://astrogeology.usgs.gov/search) & [NASA PDS Imaging](https://pds-imaging.jpl.nasa.gov/).
* **SLDEM2015 / LOLA Elevation Maps**: Topographic height maps derived from LRO Lunar Orbiter Laser Altimeter.

### 3. Built-In Synthetic Benchmark Generator (Zero-Setup Demo)
* **Location**: `lunar_reg/generator.py` / `SyntheticLunarGenerator.java`
* **Features**: Procedurally generates lunar cratered terrain with variable solar illumination (shadow angles), spatial resolution scale shifts, rotation, and sensor noise.
* **Why use it**: Guarantees a 100% working demo during hackathon live judging without needing to download multi-gigabyte PDS files.

---

### Definition
**Image Registration** is the computational process of overlaying two or more images of the same scene captured at different times, from different vantage points, or by different sensors into a single, unified spatial coordinate system.

In lunar exploration (such as the Chandrayaan-2 mission):
* **Source Image (Moving Image / $I_{src}$)**: The newly acquired image (e.g., from Chandrayaan-2 Terrain Mapping Camera TMC-2 or Orbiter High Resolution Camera OHRC) that needs to be warped and geometrically transformed.
* **Reference Image (Fixed Image / $I_{ref}$)**: The baseline coordinate map (e.g., LRO WAC/NAC base maps or prior orbital mosaics) to which the source image must be aligned.

---

### Core Challenges in Lunar Image Registration

#### 1. Illumination Variations
* **Cause**: The Moon lacks an atmosphere, resulting in extreme shadow contrast. As the solar azimuth and elevation angles change across orbits, sunlit surface features shift dramatically while crater floors fall into deep shadows.
* **Impact**: Classical pixel-intensity direct matching fails because identical terrain points exhibit wildly different brightness values and shadow shapes across different illumination passes.

#### 2. Viewpoint Variations
* **Cause**: Spacecraft orbital inclination, spacecraft jitter, off-nadir pitch/roll angles, and camera sensor tilt introduce 3D perspective distortion.
* **Impact**: Craters appear circular when viewed top-down (nadir) but stretch into ellipses under oblique viewing angles. Linear planar homography or affine transforms alone cannot fully capture complex terrain relief.

#### 3. Scale Variations
* **Cause**: Multi-sensor imaging (e.g., OHRC at ~0.25 m/pixel vs TMC-2 at 5 m/pixel) and varying satellite altitudes (e.g., 100 km vs 50 km orbit) lead to significant scale ratio disparities.
* **Impact**: Feature detectors must identify identical morphological structures across multi-octave spatial resolution scales without losing fine sub-pixel accuracy.

#### 4. Spatial Uniformity & Sub-Pixel Precision Requirements
* **Requirement**: Correspondences must not simply cluster around high-contrast crater rims. They must be evenly distributed across low-contrast lunar maria (plains) to avoid spatial warping drift.
* **Sub-Pixel Precision**: The target registration error must be sub-pixel ($< 0.5$ pixels) across the source image to support precise photogrammetric DEM (Digital Elevation Model) generation and landing site selection.

---

## 2. Proposed Solution Architecture

Our solution employs a **Robust Hybrid Pipeline** combining multiscale feature extraction, uniform spatial sampling, robust outlier filtering, non-rigid warping, and local sub-pixel intensity refinement.

```
                  +-----------------------------------+
                  | Reference Image & Source Image    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Preprocessing & CLAHE Normalization|
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Multi-scale Feature Detection     |
                  | (SIFT / Scale-Space Pyramids)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Adaptive Non-Maximal Suppression   |
                  | (ANMS / Quadtree Spatial Grid)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Descriptor Matching & Ratio Test  |
                  | (FLANN + Lowe's Ratio Filter)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Robust Geometric Estimation       |
                  | (MAGSAC++ / Homography Filtering) |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Sub-Pixel Intensity Optimization  |
                  | (Lucas-Kanade / ECC Algorithm)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Warp Transformation               |
                  | (Thin Plate Splines / Homography) |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Evaluation & Metric Computation   |
                  | (RMSE, Inlier Ratio, SSIM, Uniform)|
                  +-----------------------------------+
```

---

## 3. Project Structure & Development Roadmap

### Project Structure
```
SIH2026/
├── android_app/                             # Complete Java Android Mobile App Project (Android Studio)
│   ├── build.gradle
│   ├── settings.gradle
│   └── app/src/main/
│       ├── AndroidManifest.xml
│       ├── res/layout/activity_main.xml     # Material Design UI layout
│       └── java/com/isro/chandrayaan2/lunarregistration/
│           ├── MainActivity.java            # App UI Controller
│           ├── LunarRegistrarEngine.java    # Java Sub-Pixel Registration Engine
│           ├── SyntheticLunarGenerator.java # Java Lunar Crater Procedural Generator
│           └── MetricsCalculator.java        # Java RMSE & Inlier Metrics Evaluator
├── PROBLEM_STATEMENT_AND_SOLUTION_GUIDE.md  # Detailed Hackathon Guide & Documentation
├── README.md                                 # Quickstart & Usage Instructions
├── requirements.txt                          # Python Dependencies
├── app.py                                    # Interactive Streamlit Web UI Dashboard
├── cli.py                                    # Command Line Tool for Batch Processing
└── lunar_reg/
    ├── __init__.py
    ├── core.py                               # Hybrid Registration Engine
    ├── generator.py                          # Synthetic Lunar Crater Generator
    ├── metrics.py                            # Quantitative Metrics Suite
    └── visualizer.py                         # Checkerboard & Error Plot Visualizer
```

### Roadmap
* **Phase 1**: Environment setup, dependency management, and synthetic lunar data creation.
* **Phase 2**: Core registration engine development (SIFT + ANMS + MAGSAC++ + ECC).
* **Phase 3**: Non-rigid interpolation (Thin Plate Spline) for non-flat lunar topography.
* **Phase 4**: Metric evaluation suite (RMSE, MAE, Inlier Ratio, Uniformity Metric).
* **Phase 5**: Interactive web interface (Streamlit) for real-time visualization and pitch demonstration.

---

## 4. Algorithms & Detailed Justifications

### 1. Preprocessing: Contrast Limited Adaptive Histogram Equalization (CLAHE)
* **What it does**: Enhances contrast locally across tiles in the image without amplifying global noise.
* **Why use it**: Lunar images often suffer from deep dark shadows and overexposed crater rims. Standard global histogram equalization ruins dark areas. CLAHE balances illumination variations locally.

### 2. Feature Extraction: Scale-Invariant Feature Transform (SIFT)
* **What it does**: Detects extrema in DoG (Difference of Gaussians) scale-space and computes 128-dimensional orientation-invariant gradient histograms.
* **Why use it**: SIFT is robust against scale variations ($>2\times$ scale changes) and rotation differences between orbits.

### 3. Spatial Uniformity: Adaptive Non-Maximal Suppression (ANMS / Quadtree NMS)
* **What it does**: Suppresses redundant keypoints in dense, high-contrast regions while retaining keypoints in low-contrast regions based on spatial distance radii or quadtree distribution.
* **Why use it**: Standard feature detectors cluster thousands of points on a single bright crater wall, leaving flat maria unconstrained. ANMS guarantees uniform spatial distribution across the entire frame.

### 4. Matching: FLANN Matcher with Lowe's Ratio Test
* **What it does**: Uses randomized KD-trees to find the nearest neighbor descriptors and applies Lowe's condition ($\frac{d_1}{d_2} < 0.75$).
* **Why use it**: Lunar surfaces contain repeating geometric patterns (many small circular craters look identical). Lowe's ratio test discards ambiguous matches where the first and second best descriptors are too similar.

### 5. Outlier Filtering: MAGSAC++ (Marginalized Sample Consensus)
* **What it does**: A state-of-the-art advancement over classical RANSAC that marginalizes over noise thresholds rather than using a hard-coded pixel threshold.
* **Why use it**: Extreme illumination changes produce high outlier ratios ($>70\%$). MAGSAC++ converges faster and yields significantly more accurate homography estimates than standard RANSAC.

### 6. Sub-Pixel Refinement: Enhanced Correlation Coefficient (ECC) Algorithm
* **What it does**: Optimizes the geometric transformation parameter vector by directly maximizing the normalized correlation coefficient of image pixel intensities using gradient descent.
* **Why use it**: Feature keypoint coordinates are initially discrete (integer pixels). ECC achieves sub-pixel accuracy ($< 0.1$ to $0.25$ pixels) by aligning local continuous intensity surfaces.

### 7. Transformation Model: Thin Plate Spline (TPS) / Homography
* **What it does**: Maps point correspondences globally (Homography $3\times3$) or non-rigidly (TPS smooth radial basis functions).
* **Why use it**: While flat lunar regions are well-modeled by Homography, mountainous or cratered regions with perspective tilt require TPS to handle local topographic relief displacement.

---

## 5. Topics to Study & Curated Learning Materials

To dominate the hackathon Q&A and technical evaluation, master the following core topics using these top-tier learning resources and academic references:

---

### Module 1: Preprocessing & Contrast Normalization (CLAHE)
* **Core Concept**: Local contrast enhancement to handle severe shadow shifts on lunar craters without amplifying background noise.
* **Top Study Materials & Links**:
  1. 📘 **OpenCV Tutorial**: [Histogram Equalization & CLAHE in OpenCV](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
  2. 🎓 **Textbook Reference**: *Digital Image Processing* (4th Edition) by Rafael C. Gonzalez & Richard E. Woods – Chapter 3 (Intensity Transformations & Spatial Filtering).
  3. 📺 **Video Explanation**: [Histogram Equalization & CLAHE Explained (YouTube - First Principles of Computer Vision by Shree Nayar)](https://www.youtube.com/results?search_query=shree+nayar+histogram+equalization)

---

### Module 2: Multi-Scale Feature Detection & Descriptors (SIFT, ORB, SuperPoint)
* **Core Concept**: Detecting Extrema in Scale-Space and extracting scale/rotation invariant feature descriptors.
* **Top Study Materials & Links**:
  1. 📄 **David Lowe's Landmark SIFT Paper (2004)**: [*Distinctive Image Features from Scale-Invariant Keypoints (IJCV)*](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) – **MUST READ for Hackathon Q&A!**
  2. 📘 **OpenCV SIFT Tutorial**: [Introduction to SIFT (Scale-Invariant Feature Transform)](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
  3. 📘 **OpenCV ORB Tutorial**: [ORB (Oriented FAST and Rotated BRIEF)](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
  4. 📄 **Adaptive Non-Maximal Suppression (ANMS)**: [*Efficient Adaptive Non-Maximal Suppression Algorithms for Feature Detection (Bailo et al., 2018)*](https://arxiv.org/abs/1807.11678)
  5. 📄 **Deep Learning Matcher**: [*SuperPoint: Self-Supervised Interest Point Detection and Description (DeTone et al., CVPR)*](https://arxiv.org/abs/1712.07629)

---

### Module 3: Descriptor Matching & Robust Outlier Rejection (FLANN, RANSAC, MAGSAC++)
* **Core Concept**: Discarding ambiguous crater matches (Lowe's Ratio Test) and robustly fitting transformation matrices under high noise ($>70\%$ outliers).
* **Top Study Materials & Links**:
  1. 📘 **OpenCV Feature Matching**: [Feature Matching with FLANN & Lowe's Ratio Test](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
  2. 📄 **MAGSAC++ Paper (CVPR 2020)**: [*MAGSAC++, a fast, reliable and accurate robust estimator*](https://arxiv.org/abs/1912.05904)
  3. 📘 **OpenCV Homography Guide**: [Feature Matching + findHomography with RANSAC/MAGSAC++](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)
  4. 🎓 **Interactive Visualizer**: [RANSAC Algorithm Step-by-Step Visualization](https://www.khoury.northeastern.edu/home/rkevers/ransac/)

---

### Module 4: Sub-Pixel Accuracy & Intensity Optimization (ECC & Lucas-Kanade)
* **Core Concept**: Maximizing the Enhanced Correlation Coefficient (ECC) via gradient descent to achieve sub-pixel ($< 0.5$ pixel) alignment.
* **Top Study Materials & Links**:
  1. 📄 **ECC Algorithm Original Paper**: [*Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization (Evangelidis & Psarakis, IEEE TPAMI 2008)*](https://hal.science/hal-00863641/document)
  2. 📘 **OpenCV ECC Image Alignment Tutorial**: [Image Alignment (ECC) in OpenCV C++ / Python](https://docs.opencv.org/4.x/dc/d8b/tutorial_py_lucas_kanade.html)
  3. 🎓 **Lucas-Kanade Optical Flow Paper**: [*An Iterative Image Registration Technique with an Application to Stereo Vision (Lucas & Kanade, IJCAI 1981)*](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf)

---

### Module 5: Projective Geometry & Geometric Transformations (Homography & TPS)
* **Core Concept**: Understanding 2D Homography ($3\times3$ matrix, 8 degrees of freedom) and Thin Plate Splines (non-rigid deformation).
* **Top Study Materials & Links**:
  1. 🎓 **The Bible of Computer Vision**: [*Multiple View Geometry in Computer Vision* by Richard Hartley & Andrew Zisserman](https://www.robots.ox.ac.uk/~vgg/hzbook/)
  2. 📘 **OpenCV Homography Matrix Explanation**: [Basic Concepts of Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
  3. 📄 **Thin Plate Spline Paper**: [*Principal Warps: Thin-Plate Splines and the Decomposition of Deformations (Bookstein, 1989)*](https://ieeexplore.ieee.org/document/24792)

---

### Module 6: Remote Sensing & ISRO/NASA Lunar Datasets
* **Core Concept**: Understanding Satellite Sensor Ground Sampling Distance (GSD), Orthorectification, and Lunar Orbital Sensors.
* **Top Study Materials & Links**:
  1. 🛰️ **ISRO PRADAN Portal**: [ISRO Chandrayaan-2 ISSDC Data Portal](https://pradan.issdc.gov.in/pradan/)
  2. 🛰️ **ISRO Science Payload Details**: [Chandrayaan-2 OHRC & TMC-2 Specifications](https://www.isro.gov.in/Chandrayaan2_Payloads.html)
  3. 🌐 **NASA Planetary Data System (PDS)**: [NASA PDS Imaging Node](https://pds-imaging.jpl.nasa.gov/)
  4. 🌐 **USGS Astrogeology**: [USGS Planetary GIS & Lunar Basemaps](https://astrogeology.usgs.gov/search)

---

## 6. Evaluation Metrics Summary

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left\| H \cdot x_i^{(src)} - x_i^{(ref)} \right\|^2}$$

* **RMSE (Root Mean Square Error)**: Measures spatial reprojection error of matched keypoints in pixels (Target: $< 0.5$ pixels).
* **Inlier Count**: Total number of geometrically consistent feature correspondences post-MAGSAC++.
* **Inlier Ratio**: Percentage of valid matches ($\frac{N_{inliers}}{N_{raw\_matches}} \times 100\%$).
* **SSIM (Structural Similarity Index)**: Evaluates luminance, contrast, and structural alignment quality between warped source and reference image ($0 \text{ to } 1$, higher is better).
* **PSNR (Peak Signal-to-Noise Ratio)**: Quantitative measure of image similarity post-registration in decibels (dB).
* **Uniformity Score**: Standard deviation of spatial point density across image grid cells (lower indicates more uniform spatial distribution).
