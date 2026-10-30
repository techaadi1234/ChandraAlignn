import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import pathlib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import maximum_filter

from config import Config
from modules.preprocessing import normalize_lunar_image, apply_clahe

# Windows compatibility fix for models saved on Linux (PosixPath)
if os.name == "nt":
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass


class ResNet50BasinFeatureExtractor:
    """
    Tier 2: Deep Invariant Features (ResNet-50).
    Uses torchvision.models.resnet50(weights=DEFAULT).
    Extracts activation maps from layer3 and layer4 to detect large crater basins (>50px)
    that remain invariant under harsh shadow reversals.
    """
    def __init__(self, device=Config.DEVICE):
        self.device = device
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights).to(device)
        except Exception:
            self.model = models.resnet50(pretrained=True).to(device)
            
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_basin_peaks(self, image, min_basin_diameter=50, top_k=25):
        """
        Extracts activation maps from layer3/layer4 to locate energy peaks for large basins (>50px).
        Uses memory-safe processing with dynamic scaling and automatic CUDA memory management.
        """
        gray = normalize_lunar_image(image)
        h_orig, w_orig = gray.shape[:2]
        
        # Scale to canonical resolution for deep feature maps to prevent CUDA OOM
        max_dim = 640
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / float(max(h_orig, w_orig))
            h_proc = max(64, int(round(h_orig * scale)))
            w_proc = max(64, int(round(w_orig * scale)))
            proc_gray = cv2.resize(gray, (w_proc, h_proc), interpolation=cv2.INTER_AREA)
        else:
            proc_gray = gray
            h_proc, w_proc = h_orig, w_orig

        bgr = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)
        tensor_in = self.transform(bgr).unsqueeze(0).to(self.device)
        
        activations = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        target_layers = ['layer3', 'layer4']
        for layer_name in target_layers:
            if hasattr(self.model, layer_name):
                layer_obj = getattr(self.model, layer_name)
                hooks.append(layer_obj.register_forward_hook(get_activation(layer_name)))

        try:
            with torch.inference_mode():
                if self.device == "cuda" and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        _ = self.model(tensor_in)
                else:
                    _ = self.model(tensor_in)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            # Fallback to CPU if GPU runs out of memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for h in hooks:
                h.remove()
            hooks = []
            cpu_model = self.model.cpu()
            for layer_name in target_layers:
                if hasattr(cpu_model, layer_name):
                    layer_obj = getattr(cpu_model, layer_name)
                    hooks.append(layer_obj.register_forward_hook(get_activation(layer_name)))
            with torch.inference_mode():
                _ = cpu_model(tensor_in.cpu())
            try:
                self.model.to(self.device)
            except Exception:
                pass
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        fused_energy = np.zeros((h_orig, w_orig), dtype=np.float32)
        valid_layers = 0

        for layer_name in target_layers:
            feat_map = activations.get(layer_name, None)
            if feat_map is None:
                continue

            if len(feat_map.shape) == 4:
                energy_map = torch.sum(torch.abs(feat_map), dim=1).squeeze(0).cpu().to(torch.float32).numpy()
            elif len(feat_map.shape) == 2:
                energy_map = torch.abs(feat_map).squeeze(0).cpu().to(torch.float32).numpy().reshape(1, -1)
            else:
                continue

            energy_map = energy_map.astype(np.float32)

            if energy_map.shape != (h_orig, w_orig):
                energy_resized = cv2.resize(energy_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            else:
                energy_resized = energy_map

            energy_norm = cv2.normalize(energy_resized, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            fused_energy += energy_norm
            valid_layers += 1

        if valid_layers == 0:
            return []

        fused_energy = (fused_energy / valid_layers * 255.0).astype(np.uint8)

        # Multi-scale filter windows to capture large basins up to 250px diameter
        filter_sizes = [35, 65, 100, 140, 180, 250]
        detected_peaks_mask = np.zeros((h_orig, w_orig), dtype=bool)

        for fs in filter_sizes:
            local_max = maximum_filter(fused_energy, size=fs) == fused_energy
            background = (fused_energy == 0)
            eroded_bg = cv2.erode(background.astype(np.uint8), np.ones((3, 3), np.uint8))
            detected_peaks_mask = detected_peaks_mask | (local_max ^ eroded_bg.astype(bool))

        thresh_val = np.percentile(fused_energy, 65)
        peak_mask_u8 = (detected_peaks_mask & (fused_energy >= thresh_val)).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peak_mask_u8)

        candidate_peaks = []
        margin = 20
        for lbl in range(1, num_labels):
            pts = np.argwhere(labels == lbl)
            best_pt = max(pts, key=lambda p: fused_energy[p[0], p[1]])
            by, bx = int(best_pt[0]), int(best_pt[1])
            if bx < margin or by < margin or bx > (w_orig - margin) or by > (h_orig - margin):
                continue
            candidate_peaks.append((bx, by, fused_energy[by, bx]))

        # Sort candidate peaks by energy descending
        candidate_peaks = sorted(candidate_peaks, key=lambda p: p[2], reverse=True)[:top_k * 2]

        peak_list = []
        for x, y, peak_val in candidate_peaks:
            roi_win = 125
            rx1, ry1 = max(0, x - roi_win), max(0, y - roi_win)
            rx2, ry2 = min(w_orig, x + roi_win), min(h_orig, y + roi_win)
            
            roi_energy = fused_energy[ry1:ry2, rx1:rx2]
            cx_c, cy_c = float(x), float(y)
            semi_maj, semi_min = float(min_basin_diameter / 2.0), float(min_basin_diameter / 2.0)
            angle_deg = 0.0

            if roi_energy.size > 0 and roi_energy.shape[0] >= 5 and roi_energy.shape[1] >= 5:
                _, thresh_blob = cv2.threshold(roi_energy, int(0.30 * peak_val), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_cnt = None
                max_area = 0
                for cnt in contours:
                    if len(cnt) >= 5:
                        area = cv2.contourArea(cnt)
                        if area > max_area:
                            max_area = area
                            best_cnt = cnt
                if best_cnt is not None:
                    try:
                        (cx_r, cy_r), (d1_r, d2_r), ang_r = cv2.fitEllipse(best_cnt)
                        s_min = float(min(d1_r, d2_r) / 2.0)
                        s_maj = float(max(d1_r, d2_r) / 2.0)
                        compactness = s_min / max(1.0, s_maj)
                        # Relax compactness constraint down to 0.55 for large polygonal/degraded candidate basins (>40px radius)
                        is_valid = (compactness >= 0.55) if s_maj >= 40.0 else (compactness >= 0.35)
                        if is_valid:
                            cx_c = float(rx1 + cx_r)
                            cy_c = float(ry1 + cy_r)
                            semi_maj = min(125.0, max(s_maj, min_basin_diameter / 2.0))
                            semi_min = min(125.0, max(s_min, min_basin_diameter / 4.0))
                            angle_deg = float(ang_r)
                    except cv2.error:
                        pass

            diam = float(semi_maj * 2.0)
            # Avoid duplicate basin predictions within 30px
            if not any(np.hypot(cx_c - p["centroid"][0], cy_c - p["centroid"][1]) < 30.0 for p in peak_list):
                peak_list.append({
                    "centroid": (float(cx_c), float(cy_c)),
                    "radius": float(semi_maj),
                    "semi_major": float(semi_maj),
                    "semi_minor": float(semi_min),
                    "angle": float(angle_deg),
                    "bbox": [float(cx_c - semi_maj), float(cy_c - semi_maj), float(cx_c + semi_maj), float(cy_c + semi_maj)],
                    "confidence": float(peak_val / 255.0),
                    "diameter": float(diam),
                    "layer_source": "ResNet-50"
                })

        peak_list = sorted(peak_list, key=lambda p: p["confidence"], reverse=True)[:top_k]
        return peak_list


def load_crater_detector(model_name=Config.HF_CRATER_MODEL, device=Config.DEVICE):
    """
    Tier 3: Macro Landmark Anchors (YOLOv5).
    Loads 'Gurveer05/moon-crater-boulder-detection-yolov5' directly via yolov5.load().
    Sets conf=0.15 and iou=0.35 defaults.
    """
    model = None
    
    # 1. Try loading via yolov5 package directly
    try:
        import yolov5
        model = yolov5.load(model_name)
    except Exception as e:
        print(f"[!] Notice: yolov5.load('{model_name}') skipped ({e}). Trying Hugging Face hub downloader...")

    # 2. Try loading via huggingface_hub + torch.hub
    if model is None:
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(repo_id=model_name, filename="best.pt")
            model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, trust_repo=True)
        except Exception as e:
            print(f"[!] Warning: hf_hub_download failed ({e}). Loading torch.hub direct...")
            try:
                model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name, trust_repo=True)
            except Exception as ex:
                print(f"[!] Warning: torch.hub load failed: {ex}")

    if model is not None:
        if hasattr(model, "conf"):
            model.conf = 0.15
        if hasattr(model, "iou"):
            model.iou = 0.35

        if device and hasattr(model, "to"):
            try:
                model.to(device)
            except Exception as e:
                print(f"[!] Warning: Could not move model to {device}: {e}")

        print(f"[+] Loaded YOLOv5 crater detector: {model_name} (conf=0.15, iou=0.35)")
    return model


_resnet_extractor = None

def get_resnet_extractor():
    global _resnet_extractor
    if _resnet_extractor is None:
        try:
            _resnet_extractor = ResNet50BasinFeatureExtractor()
        except Exception as e:
            print(f"[!] Notice: Could not initialize ResNet50BasinFeatureExtractor: {e}")
    return _resnet_extractor


def run_yolo_inference_at_1024(model, image, size=1024, conf=0.15, iou=0.35):
    """
    Runs YOLOv5 inference at resolution size=1024 with conf=0.15 and iou=0.35.
    Computes true geometrical centroids:
      x_c = (x_min + x_max) / 2
      y_c = (y_min + y_max) / 2
      radius = max(x_max - x_min, y_max - y_min) / 2
    """
    enhanced = apply_clahe(image, clip_limit=2.5)
    if len(enhanced.shape) == 2:
        bgr_in = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        bgr_in = enhanced.copy()

    if model is not None:
        if hasattr(model, "conf"):
            model.conf = float(conf)
        if hasattr(model, "iou"):
            model.iou = float(iou)

        try:
            results = model(bgr_in, size=size)
        except Exception:
            try:
                results = model(bgr_in)
            except Exception:
                results = None

        preds = []
        if results is not None:
            if hasattr(results, "xyxy") and len(results.xyxy) > 0:
                raw_preds = results.xyxy[0]
                preds = raw_preds.cpu().numpy() if hasattr(raw_preds, "cpu") else np.array(raw_preds)
            elif hasattr(results, "pandas"):
                df = results.pandas().xyxy[0]
                preds = df[["xmin", "ymin", "xmax", "ymax", "confidence", "class"]].to_numpy()

        craters = []
        for p in preds:
            if len(p) >= 5:
                x_min, y_min, x_max, y_max, cf = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x_c = (x_min + x_max) / 2.0
                y_c = (y_min + y_max) / 2.0
                w_b = max(1.0, x_max - x_min)
                h_b = max(1.0, y_max - y_min)
                radius = max(w_b, h_b) / 2.0
                diameter = radius * 2.0
                craters.append({
                    "centroid": (float(x_c), float(y_c)),
                    "radius": float(radius),
                    "semi_major": float(w_b / 2.0),
                    "semi_minor": float(h_b / 2.0),
                    "angle": 0.0,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float(cf),
                    "diameter": float(diameter),
                    "layer_source": "YOLOv5"
                })
        return craters
    return []


def extract_crater_landmarks(image, model=None, min_diameter=5, max_diameter=None, conf_thresh=0.15, enable_resnet=True):
    """
    3-Tier Multi-Scale Fusion Pipeline:
    - Tier 1: Micro Extrema (<12px) - Handled via SIFT keypoint extraction.
    - Tier 2: Deep Invariant Features - ResNet-50 activation maps (layer3/layer4) for large basins (>50px).
    - Tier 3: Macro Landmark Anchors - YOLOv5 inference at size=1024, conf=0.15, iou=0.35.
    """
    if model is None:
        model = load_crater_detector()

    actual_conf = float(conf_thresh) if conf_thresh is not None else 0.15

    # 1. Tier 3: YOLOv5 Inference at size=1024
    yolo_craters = run_yolo_inference_at_1024(model, image, size=1024, conf=actual_conf, iou=0.35)

    craters_metadata = []
    for c in yolo_craters:
        if c["diameter"] >= min_diameter:
            if max_diameter is None or c["diameter"] <= max_diameter:
                craters_metadata.append(c)

    # 2. Tier 2: ResNet-50 Basin Feature Extraction (>50px)
    if enable_resnet:
        extractor = get_resnet_extractor()
        if extractor is not None:
            resnet_peaks = extractor.extract_basin_peaks(image, min_basin_diameter=50, top_k=20)
            for peak in resnet_peaks:
                px, py = peak["centroid"]
                too_close = False
                for c in craters_metadata:
                    cx, cy = c["centroid"]
                    if np.hypot(px - cx, py - cy) < 15.0:
                        too_close = True
                        break
                if not too_close:
                    craters_metadata.append(peak)

    # 2b. Multi-Scale Polygonal Basin Ingestion:
    # For large candidate regions (>40px radius), relax circularity/compactness down to 0.55
    # Accommodate features up to 250px in diameter (r_maj <= 125.0px).
    gray_im = normalize_lunar_image(image)
    h_img, w_img = gray_im.shape[:2]
    blurred = cv2.GaussianBlur(gray_im, (11, 11), 2.5)

    for block_size in [51, 85, 125, 175]:
        for C in [2, 4]:
            th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1500 or area > (h_img * w_img * 0.45):
                    continue
                if len(cnt) >= 5:
                    try:
                        (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
                        r_maj = max(d1, d2) / 2.0
                        r_min = min(d1, d2) / 2.0
                        comp = r_min / max(1e-5, r_maj)
                        # Diameter up to 250px (r_maj <= 125px) and r_maj >= 40px, comp >= 0.55
                        if 40.0 <= r_maj <= 125.0 and comp >= 0.55:
                            # Avoid duplicate large basin predictions within 35px of an existing large basin
                            if not any(np.hypot(cx - c["centroid"][0], cy - c["centroid"][1]) < 35.0 and c.get("semi_major", 0) >= 35.0 for c in craters_metadata):
                                craters_metadata.append({
                                    "centroid": (float(cx), float(cy)),
                                    "radius": float(r_maj),
                                    "semi_major": float(r_maj),
                                    "semi_minor": float(r_min),
                                    "angle": float(angle),
                                    "bbox": [float(cx - r_maj), float(cy - r_maj), float(cx + r_maj), float(cy + r_maj)],
                                    "confidence": 0.95,
                                    "diameter": float(r_maj * 2.0),
                                    "layer_source": "ResNet-50"
                                })
                    except Exception:
                        pass

    # 3. Fallback: Morphological lunar crater & basin extraction if fewer than 8 craters detected
    if len(craters_metadata) < 8:
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 < area < (gray_im.shape[0] * gray_im.shape[1] * 0.4):
                if len(cnt) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        (cx, cy), (d1, d2), angle = ellipse
                        r_maj = max(d1, d2) / 2.0
                        r_min = min(d1, d2) / 2.0
                        if 8 < r_maj < 350 and (r_min / max(1e-5, r_maj)) > 0.35:
                            if not any(np.hypot(cx - c["centroid"][0], cy - c["centroid"][1]) < 25 for c in craters_metadata):
                                craters_metadata.append({
                                    "centroid": (float(cx), float(cy)),
                                    "radius": float(r_maj),
                                    "semi_major": float(r_maj),
                                    "semi_minor": float(r_min),
                                    "angle": float(angle),
                                    "bbox": [float(cx - r_maj), float(cy - r_maj), float(cx + r_maj), float(cy + r_maj)],
                                    "confidence": 0.88,
                                    "diameter": float(r_maj * 2.0),
                                    "layer_source": "YOLOv5-Morphology"
                                })
                    except Exception:
                        pass

    # 4. Strict Geometric & Photometric Post-Processing Pipeline
    # - Stricter Eccentricity (e <= 0.78) & Aspect Ratio (AR <= 1.6)
    # - Rim Profile Verification (Perimeter Gradient Check on 32 equidistant samples)
    # - Scale Caps: a <= 0.35 * min(W, H)
    # - Rotated NMS (Polygon/Rotated BBox IoU at thresh = 0.30)
    # - Enclosure Suppression: suppress large ellipse enclosing >=2 smaller craters unless rim grad >= 1.5x terrain mean
    # - Diagnostic Logging: Small (<15px), Medium (15-50px), Large (>50px)
    craters_metadata, scale_counts = filter_and_validate_crater_candidates(craters_metadata, image)

    # 5. Render 3-Color Visual Overlays
    gray_orig = normalize_lunar_image(image)
    annotated_img = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR) if len(gray_orig.shape) == 2 else gray_orig.copy()

    for c in craters_metadata:
        x_c, y_c = c["centroid"]
        semi_major = c["semi_major"]
        semi_minor = c["semi_minor"]
        angle = c["angle"]
        diameter = c["diameter"]
        source = c.get("layer_source", "YOLOv5")

        # Color-coded categories strictly matching prompt requirement:
        # Green (<12px micro-craters), Yellow (12–25px craters), Red (>25px basins)
        if diameter > 25.0 or source == "ResNet-50":
            color_bgr = (0, 0, 255)    # 🔴 Red (>25px basins)
            thickness = 2
        elif diameter >= 12.0:
            color_bgr = (0, 255, 255)  # 🟡 Yellow (12–25px craters)
            thickness = 2
        else:
            color_bgr = (0, 255, 0)    # 🟢 Green (<12px micro-craters)
            thickness = 1

        center = (int(round(x_c)), int(round(y_c)))
        axes = (int(max(2, round(semi_major))), int(max(2, round(semi_minor))))
        cv2.ellipse(annotated_img, center, axes, int(round(angle)), 0, 360, color_bgr, thickness, cv2.LINE_AA)
        cv2.circle(annotated_img, center, 2 if diameter < 12 else 3, color_bgr, -1)

    return craters_metadata, annotated_img


def compute_rotated_iou(box1, box2):
    """
    Computes IoU between two rotated rectangles using cv2.rotatedRectangleIntersection.
    box: ((cx, cy), (width, height), angle_deg)
    """
    ret, pts = cv2.rotatedRectangleIntersection(box1, box2)
    if ret == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0
    area1 = float(box1[1][0] * box1[1][1])
    area2 = float(box2[1][0] * box2[1][1])
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    if ret == cv2.INTERSECT_FULL:
        return float(min(area1, area2) / max(1e-5, max(area1, area2)))
    inter_area = float(cv2.contourArea(pts))
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def filter_and_validate_crater_candidates(candidates, image):
    """
    Crater Ellipse Post-Processing & Filtering Pipeline:
    1. Stricter Eccentricity & Aspect Ratio Constraints:
       - AR = max(a, b) / min(a, b) <= 1.6
       - e = sqrt(1 - (b/a)^2) <= 0.78
    2. Rim Profile Verification (Perimeter Gradient Check):
       - Sample 32 equidistant points along the parameterized perimeter:
         x(t) = xc + a*cos(t)*cos(theta) - b*sin(t)*sin(theta)
         y(t) = yc + a*sin(t)*cos(theta) + b*cos(t)*sin(theta)
       - Compute Sobel gradient magnitude |grad I| = sqrt(Gx^2 + Gy^2)
       - Discard candidates if more than 35% of perimeter points fall below edge threshold (flat terrain filter).
    3. Scale Caps & Scale-Dependent NMS:
       - Max Dimension Limit: a <= 0.35 * min(W, H)
       - Rotated NMS (Polygon / Rotated BBox IoU) with threshold 0.30 across all scales.
       - Enclosure Suppression: If larger candidate encloses >= 2 smaller valid craters,
         suppress large ellipse unless perimeter gradient strength >= 1.5 * background terrain mean.
    4. Diagnostic Logging:
       - Categorized detection counts: Small (<15px), Medium (15-50px), Large (>50px).
    """
    if not candidates:
        return [], {"small": 0, "medium": 0, "large": 0, "total": 0}

    gray = normalize_lunar_image(image)
    h_img, w_img = gray.shape[:2]
    max_semi_major = 0.35 * float(min(w_img, h_img))

    # Pre-calculate Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    mean_terrain_grad = float(np.mean(grad_mag))
    edge_strength_thresh = max(8.0, 0.40 * mean_terrain_grad)

    t_vals = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    cos_t = np.cos(t_vals)
    sin_t = np.sin(t_vals)

    valid_candidates = []

    for c in candidates:
        a = float(c.get("semi_major", c.get("radius", 5.0)))
        b = float(c.get("semi_minor", c.get("radius", 5.0)))
        theta_deg = float(c.get("angle", 0.0))

        # Canonical ordering: a is semi-major (larger), b is semi-minor (smaller)
        if a < b:
            a, b = b, a
            theta_deg = float((theta_deg + 90.0) % 180.0)

        # 1. Scale Cap: a <= 0.35 * min(W, H)
        if a > max_semi_major or a <= 1.0 or b <= 0.5:
            continue

        # 2. Stricter Eccentricity & Aspect Ratio Constraints
        ar = a / max(1e-5, b)
        if ar > 1.6:
            continue
        e = np.sqrt(max(0.0, 1.0 - (b / a)**2))
        if e > 0.78:
            continue

        # 3. Rim Profile Verification (Perimeter Gradient Check on 32 points)
        xc, yc = float(c["centroid"][0]), float(c["centroid"][1])
        if xc < 0 or yc < 0 or xc >= w_img or yc >= h_img:
            continue

        theta_rad = np.radians(theta_deg)
        cos_th = np.cos(theta_rad)
        sin_th = np.sin(theta_rad)

        px = xc + a * cos_t * cos_th - b * sin_t * sin_th
        py = yc + a * sin_t * cos_th + b * cos_t * sin_th

        px_clipped = np.clip(np.round(px).astype(int), 0, w_img - 1)
        py_clipped = np.clip(np.round(py).astype(int), 0, h_img - 1)
        sampled_grads = grad_mag[py_clipped, px_clipped]

        weak_edge_count = int(np.sum(sampled_grads < edge_strength_thresh))
        if (weak_edge_count / 32.0) > 0.35:
            # Discard candidates traversing flat terrain
            continue

        mean_rim_grad = float(np.mean(sampled_grads))

        item = dict(c)
        item["semi_major"] = a
        item["semi_minor"] = b
        item["angle"] = theta_deg
        item["diameter"] = float(a * 2.0)
        item["radius"] = a
        item["mean_rim_grad"] = mean_rim_grad
        item["eccentricity"] = float(e)
        item["aspect_ratio"] = float(ar)
        valid_candidates.append(item)

    # 4. Rotated NMS (Rotated BBox IoU at threshold 0.30)
    valid_candidates.sort(key=lambda x: float(x.get("confidence", 0.5)), reverse=True)

    rboxes = [
        ((float(c["centroid"][0]), float(c["centroid"][1])),
         (float(c["semi_major"] * 2.0), float(c["semi_minor"] * 2.0)),
         float(c.get("angle", 0.0)))
        for c in valid_candidates
    ]

    keep_indices = []
    order = list(range(len(valid_candidates)))

    while len(order) > 0:
        curr_idx = order[0]
        keep_indices.append(curr_idx)
        remaining = []
        for next_idx in order[1:]:
            iou = compute_rotated_iou(rboxes[curr_idx], rboxes[next_idx])
            if iou <= 0.30:
                remaining.append(next_idx)
        order = remaining

    nms_filtered = [valid_candidates[i] for i in keep_indices]

    # 5. Enclosure Suppression:
    # If a larger candidate fully encloses multiple (>=2) smaller valid craters,
    # suppress the large ellipse unless its perimeter gradient strength >= 1.5 * terrain mean.
    final_craters = []
    for i, c_large in enumerate(nms_filtered):
        a_L = float(c_large["semi_major"])
        b_L = float(c_large["semi_minor"])
        xc_L, yc_L = float(c_large["centroid"][0]), float(c_large["centroid"][1])
        theta_L = np.radians(float(c_large.get("angle", 0.0)))
        cos_L = np.cos(theta_L)
        sin_L = np.sin(theta_L)

        # Only evaluate large candidates for enclosure
        if a_L >= 35.0:
            enclosed_count = 0
            for j, c_small in enumerate(nms_filtered):
                if i == j:
                    continue
                a_S = float(c_small["semi_major"])
                if a_S < 0.75 * a_L:
                    dx = float(c_small["centroid"][0]) - xc_L
                    dy = float(c_small["centroid"][1]) - yc_L
                    x_rot = dx * cos_L + dy * sin_L
                    y_rot = -dx * sin_L + dy * cos_L
                    if ((x_rot / a_L)**2 + (y_rot / b_L)**2) <= 1.0:
                        enclosed_count += 1

            if enclosed_count >= 2:
                if c_large.get("mean_rim_grad", 0.0) < 1.5 * mean_terrain_grad:
                    # Enclosure suppression: phantom macro ellipse over terrain cluster
                    continue

        final_craters.append(c_large)

    # 6. Diagnostic Logging categorized by scale
    counts = {"small": 0, "medium": 0, "large": 0, "total": len(final_craters)}
    for c in final_craters:
        diam = c["diameter"]
        if diam < 15.0:
            counts["small"] += 1
        elif diam <= 50.0:
            counts["medium"] += 1
        else:
            counts["large"] += 1

    print(f"[Crater Post-Processing] Validated {len(final_craters)} / {len(candidates)} craters "
          f"(Small <15px: {counts['small']}, Medium 15-50px: {counts['medium']}, Large >50px: {counts['large']})")

    return final_craters, counts


# Backward Compatibility Wrappers
def load_crater_model(checkpoint_path=None, device=Config.DEVICE):
    return load_crater_detector(model_name=Config.HF_CRATER_MODEL, device=device)

def predict_craters(img, model=None, checkpoint_path=None, thresh=0.15, min_radius=2, max_radius=200):
    craters_metadata, annotated_img = extract_crater_landmarks(img, model=model, min_diameter=min_radius * 2, conf_thresh=thresh)
    h, w = img.shape[:2]
    mask_binary = np.zeros((h, w), dtype=np.uint8)
    for c in craters_metadata:
        cx, cy = c["centroid"]
        r = c["radius"]
        cv2.circle(mask_binary, (int(cx), int(cy)), int(r), 255, -1)
    return mask_binary, mask_binary, craters_metadata
