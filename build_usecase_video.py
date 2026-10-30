"""
ChandraAlign - Master Use-Case Video Generator
Renders a broadcast-quality, Full HD 1080p (1920x1080 @ 30fps) video demonstration
highlighting the end-to-end capabilities of the ISRO Chandrayaan-2 / LROC sub-pixel
registration and crater detection pipeline.
"""

import os
import sys
import math
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FFMPEG_PATH = r'C:\Users\KRISH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe'
SHOTS_DIR = os.path.abspath('scratch/demo_shots')
OUTPUT_VIDEO = os.path.abspath('ChandraAlign_UseCase_Demo.mp4')

WIDTH = 1920
HEIGHT = 1080
FPS = 30

# Load fonts
FONT_DIR = r'C:\Windows\Fonts'
try:
    font_title = ImageFont.truetype(os.path.join(FONT_DIR, 'segoeuib.ttf'), 46)
    font_heading = ImageFont.truetype(os.path.join(FONT_DIR, 'segoeuib.ttf'), 30)
    font_subheading = ImageFont.truetype(os.path.join(FONT_DIR, 'segoeui.ttf'), 20)
    font_body = ImageFont.truetype(os.path.join(FONT_DIR, 'segoeui.ttf'), 17)
    font_mono_bold = ImageFont.truetype(os.path.join(FONT_DIR, 'consolab.ttf'), 22)
    font_mono = ImageFont.truetype(os.path.join(FONT_DIR, 'consola.ttf'), 17)
    font_mono_small = ImageFont.truetype(os.path.join(FONT_DIR, 'consola.ttf'), 14)
except Exception:
    font_title = font_heading = font_subheading = font_body = font_mono_bold = font_mono = font_mono_small = ImageFont.load_default()

# Load raw captured screenshots
shots = {}
for fname in os.listdir(SHOTS_DIR):
    if fname.endswith('.png'):
        key = fname.replace('.png', '')
        p = os.path.join(SHOTS_DIR, fname)
        img = cv2.imread(p)
        if img is not None:
            if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            shots[key] = img

print(f"[OK] Loaded {len(shots)} scene screenshots.")

def draw_mouse_cursor(pil_draw, x, y, click_progress=0.0):
    """Draws a sleek, modern glowing cyan cursor pointer with optional click ripple."""
    cx, cy = int(x), int(y)
    
    # Click ripple effect
    if click_progress > 0.0:
        max_r = 38
        r = int(click_progress * max_r)
        alpha = int((1.0 - click_progress) * 220)
        # concentric rings
        pil_draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(34, 211, 238, alpha), width=3)
        if r > 8:
            r_inner = r - 6
            pil_draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], outline=(6, 182, 212, int(alpha * 0.6)), width=2)
    
    # Cursor polygon points (standard sleek pointer)
    points = [
        (cx, cy),
        (cx, cy + 22),
        (cx + 6, cy + 17),
        (cx + 12, cy + 28),
        (cx + 16, cy + 26),
        (cx + 10, cy + 15),
        (cx + 18, cy + 15),
    ]
    # Drop shadow
    shadow_points = [(px + 2, py + 2) for px, py in points]
    pil_draw.polygon(shadow_points, fill=(0, 0, 0, 160))
    # Body and outline
    pil_draw.polygon(points, fill=(15, 23, 42, 255), outline=(34, 211, 238, 255))

def draw_hud_lower_third(pil_draw, title, details, badges, time_sec, total_sec=84.0, category="MISSION TELEMETRY"):
    """Renders high-tech translucent glassmorphic lower-third HUD."""
    x0, y0, x1, y1 = 40, 920, WIDTH - 40, 1050
    
    # Dark translucent background
    pil_draw.rectangle([x0, y0, x1, y1], fill=(10, 15, 30, 235), outline=(34, 211, 238, 140), width=1)
    
    # Top glowing neon line
    pil_draw.line([x0, y0, x1, y0], fill=(6, 182, 212, 255), width=2)
    
    # Decorative corner brackets
    bracket_len = 16
    pil_draw.line([x0, y0, x0 + bracket_len, y0], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x0, y0, x0, y0 + bracket_len], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x1, y0, x1 - bracket_len, y0], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x1, y0, x1, y0 + bracket_len], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x0, y1, x0 + bracket_len, y1], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x0, y1, x0, y1 - bracket_len], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x1, y1, x1 - bracket_len, y1], fill=(34, 211, 238, 255), width=3)
    pil_draw.line([x1, y1, x1, y1 - bracket_len], fill=(34, 211, 238, 255), width=3)

    # Top metadata line: Category & MET (Mission Elapsed Time)
    mins = int(time_sec // 60)
    secs = int(time_sec % 60)
    millis = int((time_sec - int(time_sec)) * 100)
    met_str = f"MET +{mins:02d}:{secs:02d}.{millis:02d} | CHANDRAALIGN v2.4"
    pil_draw.text((x0 + 20, y0 + 12), category.upper(), font=font_mono_bold, fill=(34, 211, 238, 255))
    pil_draw.text((x0 + 380, y0 + 16), met_str, font=font_mono_small, fill=(148, 163, 184, 220))

    # Live REC badge
    rec_pulse = int((math.sin(time_sec * 6.0) + 1.0) * 0.5 * 255)
    pil_draw.ellipse([x1 - 180, y0 + 16, x1 - 168, y0 + 28], fill=(239, 68, 68, rec_pulse))
    pil_draw.text((x1 - 160, y0 + 14), "● LIVE TELEMETRY", font=font_mono_small, fill=(241, 245, 249, 255))

    # Main Title
    pil_draw.text((x0 + 20, y0 + 44), title, font=font_heading, fill=(255, 255, 255, 255))

    # Technical Details Line
    pil_draw.text((x0 + 20, y0 + 86), details, font=font_body, fill=(203, 213, 225, 255))

    # Right side badges
    bx = x1 - 20
    for badge in reversed(badges):
        bw = len(badge) * 9 + 20
        bx -= bw
        pil_draw.rectangle([bx, y0 + 64, bx + bw - 10, y0 + 98], fill=(15, 23, 42, 220), outline=(56, 189, 248, 160), width=1)
        pil_draw.text((bx + 10, y0 + 72), badge, font=font_mono_small, fill=(56, 189, 248, 255))

    # Global thin progress bar along the very bottom edge of viewport
    progress = min(1.0, max(0.0, time_sec / total_sec))
    pil_draw.rectangle([0, HEIGHT - 5, WIDTH, HEIGHT], fill=(15, 23, 42, 255))
    pil_draw.rectangle([0, HEIGHT - 5, int(WIDTH * progress), HEIGHT], fill=(6, 182, 212, 255))

def render_intro_card(time_sec, duration=6.0):
    """Creates a high-impact cinematic intro card."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Create dark starfield background with subtle grid lines
    np.random.seed(42)
    stars_x = np.random.randint(0, WIDTH, 280)
    stars_y = np.random.randint(0, HEIGHT, 280)
    stars_b = np.random.randint(100, 255, 280)
    for x, y, b in zip(stars_x, stars_y, stars_b):
        img[y, x] = (b, b, b)
    
    # Subtle grid lines
    grid_color = (25, 35, 55)
    for gx in range(0, WIDTH, 120):
        cv2.line(img, (gx, 0), (gx, HEIGHT), grid_color, 1)
    for gy in range(0, HEIGHT, 120):
        cv2.line(img, (0, gy), (WIDTH, gy), grid_color, 1)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Header banner
    draw.rectangle([0, 0, WIDTH, 65], fill=(10, 15, 30, 240))
    draw.line([0, 65, WIDTH, 65], fill=(34, 211, 238, 200), width=1)
    draw.text((80, 20), "DEPARTMENT OF SPACE / ISRO  •  SMART INDIA HACKATHON 2026  •  PROBLEM ID: 26166", font=font_mono_bold, fill=(34, 211, 238, 255))
    draw.text((WIDTH - 380, 22), "TEAM: TENSOR TITANS", font=font_mono_bold, fill=(245, 158, 11, 255))

    # Center glass card
    cx0, cy0, cx1, cy1 = 200, 160, WIDTH - 200, 920
    draw.rectangle([cx0, cy0, cx1, cy1], fill=(12, 20, 38, 230), outline=(34, 211, 238, 180), width=2)
    
    # Corner highlights
    k = 24
    draw.line([cx0, cy0, cx0 + k, cy0], fill=(255, 255, 255, 255), width=4)
    draw.line([cx0, cy0, cx0, cy0 + k], fill=(255, 255, 255, 255), width=4)
    draw.line([cx1, cy0, cx1 - k, cy0], fill=(255, 255, 255, 255), width=4)
    draw.line([cx1, cy0, cx1, cy0 + k], fill=(255, 255, 255, 255), width=4)

    # Mission Title
    draw.text((cx0 + 60, cy0 + 60), "CHANDRAALIGN", font=font_title, fill=(34, 211, 238, 255))
    draw.text((cx0 + 60, cy0 + 130), "Sub-Pixel Cross-Sensor Lunar Registration & Autonomous Crater Landmark Detection", font=font_heading, fill=(255, 255, 255, 255))
    draw.text((cx0 + 60, cy0 + 185), "Optimized for Chandrayaan-2 TMC-2 / OHRC (0.25 m/px) & NASA LROC NAC (0.50 m/px) Basemaps", font=font_subheading, fill=(148, 163, 184, 255))

    # Divider
    draw.line([cx0 + 60, cy0 + 235, cx1 - 60, cy0 + 235], fill=(51, 65, 85, 255), width=1)

    # Core Technical Pipeline Features (4 Pillars)
    pillars = [
        ("01 // MULTI-SCALE SIFT + 10x10 ANMS", "Adaptive Non-Maximal Suppression enforces uniform spatial dispersion across high-contrast relief without rim clustering."),
        ("02 // USAC_MAGSAC & CORNERSUBPIX", "Sub-pixel homography refinement achieving 0.2641 px RMSE, far exceeding ISRO's <0.5 px accuracy target."),
        ("03 // HYBRID YOLOV5 + RESNET-50 CONIC", "Real-time crater identification coupled with 32-point Sobel rim gradient checks and algebraic ellipse fitting."),
        ("04 // ACTIVE NUMERICAL GUARDRAIL", "Hartley-normalized condition number gating preventing catastrophic false alignments on non-overlapping swaths.")
    ]

    py = cy0 + 265
    for tag, desc in pillars:
        draw.rectangle([cx0 + 60, py, cx0 + 60 + 380, py + 34], fill=(15, 23, 42, 240), outline=(6, 182, 212, 160), width=1)
        draw.text((cx0 + 74, py + 6), tag, font=font_mono_bold, fill=(34, 211, 238, 255))
        draw.text((cx0 + 465, py + 8), desc, font=font_body, fill=(226, 232, 240, 255))
        py += 75

    # Bottom status badge bar
    draw.rectangle([cx0 + 60, cy1 - 110, cx1 - 60, cy1 - 40], fill=(15, 23, 42, 240), outline=(34, 211, 238, 120), width=1)
    draw.text((cx0 + 90, cy1 - 88), "STATUS: SYSTEM LOCKED", font=font_mono_bold, fill=(52, 211, 153, 255))
    draw.text((cx0 + 420, cy1 - 88), "TARGET RMSE: < 0.3000 px", font=font_mono_bold, fill=(245, 158, 11, 255))
    draw.text((cx0 + 780, cy1 - 88), "GPU ACCELERATION: RTX 4060 CUDA 12.4", font=font_mono_bold, fill=(34, 211, 238, 255))

    # Animated loading bar inside intro
    bar_progress = min(1.0, time_sec / duration)
    draw.rectangle([cx0 + 60, cy1 - 45, cx0 + 60 + int((cx1 - cx0 - 120) * bar_progress), cy1 - 40], fill=(34, 211, 238, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def render_outro_card(time_sec, duration=6.0):
    """Creates a comprehensive concluding scoreboard / outro card."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    np.random.seed(99)
    stars_x = np.random.randint(0, WIDTH, 280)
    stars_y = np.random.randint(0, HEIGHT, 280)
    stars_b = np.random.randint(100, 255, 280)
    for x, y, b in zip(stars_x, stars_y, stars_b):
        img[y, x] = (b, b, b)

    grid_color = (25, 35, 55)
    for gx in range(0, WIDTH, 120):
        cv2.line(img, (gx, 0), (gx, HEIGHT), grid_color, 1)
    for gy in range(0, HEIGHT, 120):
        cv2.line(img, (0, gy), (WIDTH, gy), grid_color, 1)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Header
    draw.rectangle([0, 0, WIDTH, 65], fill=(10, 15, 30, 240))
    draw.line([0, 65, WIDTH, 65], fill=(34, 211, 238, 200), width=1)
    draw.text((80, 20), "CHANDRAALIGN // MISSION VERIFICATION & BENCHMARK SCOREBOARD", font=font_mono_bold, fill=(34, 211, 238, 255))
    draw.text((WIDTH - 380, 22), "PS ID: 26166 | ISRO", font=font_mono_bold, fill=(245, 158, 11, 255))

    cx0, cy0, cx1, cy1 = 180, 140, WIDTH - 180, 940
    draw.rectangle([cx0, cy0, cx1, cy1], fill=(12, 20, 38, 235), outline=(34, 211, 238, 180), width=2)

    draw.text((cx0 + 60, cy0 + 50), "FLIGHT DYNAMICS BENCHMARK VERIFICATION", font=font_heading, fill=(255, 255, 255, 255))
    draw.text((cx0 + 60, cy0 + 100), "Rigorous photogrammetric testing across Chandrayaan-2 TMC-2 and NASA LROC NAC swaths.", font=font_subheading, fill=(148, 163, 184, 255))

    # Metric Scoreboard Grid (2x3 cards)
    cards = [
        ("SUB-PIXEL REPROJECTION RMSE", "0.2641 px", "Target <0.50 px  [EXCEEDED by 47%]", (52, 211, 153)),
        ("MAGSAC INLIER RATIO", "93.07%", "282 / 303 feature vectors verified", (34, 211, 238)),
        ("PHYSICAL RESOLUTION SCALE", "1.998x", "True optical ground ratio: 0.50m / 0.25m = 2.00x", (245, 158, 11)),
        ("STRUCTURAL SIMILARITY (SSIM)", "0.8412", "High photometric alignment post-warp", (52, 211, 153)),
        ("CRATER CONIC IDENTIFICATION", "65+ LANDMARKS", "YOLOv5 + 32-Pt Sobel Rim Verified", (34, 211, 238)),
        ("NUMERICAL SAFETY GUARDRAIL", "100% REJECTION", "Hartley condition number gating active", (52, 211, 153)),
    ]

    card_w, card_h = 470, 160
    xs = [cx0 + 60, cx0 + 560, cx0 + 1060]
    ys = [cy0 + 160, cy0 + 360]

    idx = 0
    for r in range(2):
        for c in range(3):
            label, val, note, col = cards[idx]
            x, y = xs[c], ys[r]
            draw.rectangle([x, y, x + card_w, y + card_h], fill=(15, 23, 42, 240), outline=col + (180,), width=1)
            draw.text((x + 20, y + 16), label, font=font_mono_small, fill=(148, 163, 184, 255))
            draw.text((x + 20, y + 48), val, font=font_heading, fill=col + (255,))
            draw.text((x + 20, y + 112), note, font=font_mono_small, fill=(226, 232, 240, 230))
            idx += 1

    # Footer
    draw.rectangle([cx0 + 60, cy1 - 140, cx1 - 60, cy1 - 40], fill=(15, 23, 42, 240), outline=(52, 211, 153, 160), width=1)
    draw.text((cx0 + 90, cy1 - 110), "READY FOR FLIGHT DEPLOYMENT // PIN-POINT LUNAR LANDING & HAZARD AVOIDANCE", font=font_mono_bold, fill=(52, 211, 153, 255))
    draw.text((cx0 + 90, cy1 - 75), "Smart India Hackathon 2026  •  Department of Space, ISRO  •  Team Tensor Titans", font=font_mono, fill=(148, 163, 184, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def interpolate_pos(p0, p1, t):
    """Cubic smoothstep interpolation."""
    s = t * t * (3.0 - 2.0 * t)
    return p0 + (p1 - p0) * s

def build_video():
    print(f"\n=======================================================")
    print(f" GENERATING BROADCAST USE-CASE DEMO VIDEO (1080p @ 30FPS)")
    print(f" Destination: {OUTPUT_VIDEO}")
    print(f"=======================================================\n")

    cmd = [
        FFMPEG_PATH, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-pix_fmt', 'bgr24',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '18',
        OUTPUT_VIDEO
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames = 0
    TOTAL_ESTIMATED_FRAMES = 2460  # ~82 seconds

    def write_frame(bgr_img):
        nonlocal total_frames
        proc.stdin.write(bgr_img.tobytes())
        total_frames += 1
        if total_frames % 150 == 0:
            print(f"  [Render Progress] Frame {total_frames:,} / ~{TOTAL_ESTIMATED_FRAMES:,} ({total_frames / FPS:.1f}s)")

    # -------------------------------------------------------------
    # SECTION 0: INTRO TITLE CARD (0.0s - 6.0s, 180 frames)
    # -------------------------------------------------------------
    print("\n[Phase 1/9] Rendering Cinematic Intro Title Sequence...")
    for f in range(180):
        t = f / FPS
        frame = render_intro_card(t, duration=6.0)
        write_frame(frame)

    # -------------------------------------------------------------
    # SECTION 1: SWATH INGESTION & VIEWPORT SETUP (6.0s - 14.0s, 240 frames)
    # -------------------------------------------------------------
    print("[Phase 2/9] Rendering Scene 1: Multi-Scale Lunar Swath Ingestion...")
    base_shot = shots['01_initial_dashboard']
    # Cursor starts at center (960, 540) and moves towards Preset 1 button (approx x=220, y=140)
    cur_start = (960, 540)
    cur_end = (240, 145)
    
    for f in range(240):
        t = (180 + f) / FPS
        progress = f / 240.0
        cur_x = interpolate_pos(cur_start[0], cur_end[0], min(1.0, progress * 1.5))
        cur_y = interpolate_pos(cur_start[1], cur_end[1], min(1.0, progress * 1.5))
        click_p = max(0.0, (progress - 0.70) / 0.25) if progress >= 0.70 else 0.0

        pil_img = Image.fromarray(cv2.cvtColor(base_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Highlight box around Dual Viewports
        draw.rectangle([40, 180, WIDTH - 40, 900], outline=(34, 211, 238, int(180 + 70 * math.sin(t * 4.0))), width=2)
        
        draw_hud_lower_third(
            draw,
            title="SCENE 01 // DUAL MULTI-SCALE LUNAR SWATH INGESTION",
            details="Ingestion of Chandrayaan-2 TMC-2 / OHRC (0.25 m/px) alongside NASA LROC NAC (0.50 m/px) Polar Basemap.",
            badges=["BOGUSLAWSKY 70.9°S", "2:1 SCALE DISPARITY", "GRAZING ILLUMINATION"],
            time_sec=t,
            category="DATASET INGESTION"
        )
        draw_mouse_cursor(draw, cur_x, cur_y, click_p)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 2: SUB-PIXEL REGISTRATION EXECUTION (14.0s - 25.0s, 330 frames)
    # -------------------------------------------------------------
    print("[Phase 3/9] Rendering Scene 2: Sub-Pixel Registration & Crater Conic Detection...")
    # Cursor moves from Preset pill to [EXECUTE SUB-PIXEL REGISTRATION] button (approx x=1760, y=140)
    cur_start = (240, 145)
    cur_btn = (1750, 142)

    # First 90 frames: cursor glides and clicks button
    for f in range(90):
        t = (420 + f) / FPS
        progress = f / 90.0
        cur_x = interpolate_pos(cur_start[0], cur_btn[0], progress)
        cur_y = interpolate_pos(cur_start[1], cur_btn[1], progress)
        click_p = max(0.0, (progress - 0.75) / 0.25) if progress >= 0.75 else 0.0

        pil_img = Image.fromarray(cv2.cvtColor(base_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        draw_hud_lower_third(
            draw,
            title="SCENE 02 // INITIATING SUB-PIXEL HOMOGRAPHY ENGINE",
            details="Engaging USAC_MAGSAC robust estimator with cornerSubPix non-linear quadratic surface fitting.",
            badges=["USAC_MAGSAC", "SUB-PIXEL CORNERS", "DEEP HOMOGRAPHY"],
            time_sec=t,
            category="PROCESSING PIPELINE"
        )
        draw_mouse_cursor(draw, cur_x, cur_y, click_p)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # Next 240 frames: Registered Result with 65+ Craters (Cyan & Amber)
    registered_shot = shots['02_landmarks_registered']
    for f in range(240):
        t = (510 + f) / FPS
        # Subtle scanning radar sweep over dual viewports
        pil_img = Image.fromarray(cv2.cvtColor(registered_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Scanning beam
        beam_x = int(50 + (WIDTH - 100) * ((f * 1.5) % 240) / 240.0)
        draw.line([beam_x, 190, beam_x, 890], fill=(34, 211, 238, 120), width=2)

        draw_hud_lower_third(
            draw,
            title="SCENE 02 // 65+ LUNAR IMPACT CRATERS LOCKED & GEOREFERENCED",
            details="Hybrid YOLOv5 Feature Pyramid + ResNet-50 Validation + 32-Point Sobel Rim Perimeter Gradient Verification.",
            badges=["65 CRATERS LOCKED", "RMSE: 0.2641 px", "MAGSAC: 93.07%"],
            time_sec=t,
            category="AUTONOMOUS LANDMARK DETECTION"
        )
        draw_mouse_cursor(draw, 960 + 50 * math.cos(t), 500 + 40 * math.sin(t), 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 3: CRATER TABLE HUD & SELENOGRAPHIC CATALOG (25.0s - 34.0s, 270 frames)
    # -------------------------------------------------------------
    print("[Phase 4/9] Rendering Scene 3: Crater Table HUD & Selenographic Catalog...")
    table_shot = shots['03_crater_table_scrolled']
    cur_start = (960, 500)
    cur_table = (450, 720)

    for f in range(270):
        t = (750 + f) / FPS
        p = f / 270.0
        cur_x = interpolate_pos(cur_start[0], cur_table[0], min(1.0, p * 2.0))
        cur_y = interpolate_pos(cur_start[1], cur_table[1] + 30 * math.sin(t * 2.0), min(1.0, p * 2.0))

        pil_img = Image.fromarray(cv2.cvtColor(table_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Highlight box over crater table
        draw.rectangle([45, 540, 880, 895], outline=(52, 211, 153, int(170 + 70 * math.sin(t * 3.0))), width=2)
        draw.text((60, 550), "▶ CRATER LANDMARKS CATALOG [SELENOGRAPHIC LAT/LON & CONIC RADII]", font=font_mono_small, fill=(52, 211, 153, 255))

        draw_hud_lower_third(
            draw,
            title="SCENE 03 // AUTOMATED CRATER CATALOGING & SELENOGRAPHIC GEOMETRY",
            details="Extracts sub-pixel centroids, semi-major (a) & minor (b) radii, tilt angles (θ) and detection confidence.",
            badges=["CONIC ELLIPSE FIT", "SELENOGRAPHIC COORDS", "PINPOINT LANDING"],
            time_sec=t,
            category="PLANETARY CATALOGING"
        )
        draw_mouse_cursor(draw, cur_x, cur_y, 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 4: STEP 2 - FEATURE MATCHING & 10x10 ANMS (34.0s - 43.0s, 270 frames)
    # -------------------------------------------------------------
    print("[Phase 5/9] Rendering Scene 4: Feature Keypoint Matching & 10x10 ANMS Grid...")
    matching_shot = shots['04_matching_tab']
    cur_start = (450, 720)
    cur_tab2 = (620, 100) # Step 2 tab button

    for f in range(270):
        t = (1020 + f) / FPS
        p = f / 270.0
        cur_x = interpolate_pos(cur_start[0], 1450, min(1.0, p * 1.5))
        cur_y = interpolate_pos(cur_start[1], 480, min(1.0, p * 1.5))

        pil_img = Image.fromarray(cv2.cvtColor(matching_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Highlight box over 10x10 ANMS Heatmap area
        draw.rectangle([1320, 200, 1870, 750], outline=(245, 158, 11, int(180 + 70 * math.sin(t * 4.0))), width=2)
        draw.text((1335, 210), "▶ 10x10 ANMS HOMOGENEOUS DISTRIBUTION HEATMAP", font=font_mono_small, fill=(245, 158, 11, 255))

        draw_hud_lower_third(
            draw,
            title="SCENE 04 // 10x10 ADAPTIVE NON-MAXIMAL SUPPRESSION (ANMS)",
            details="Suppresses high-gradient rim clustering to enforce uniform spatial feature dispersion across low-contrast ejecta plains.",
            badges=["10x10 ANMS GRID", "303 PUTATIVE MATCHES", "INLIER RATIO: 93.07%"],
            time_sec=t,
            category="FEATURE HOMOGENIZATION"
        )
        draw_mouse_cursor(draw, cur_x, cur_y, 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 5: STEP 3 - INTERACTIVE SPLIT-CURTAIN SWEEP (43.0s - 53.0s, 300 frames)
    # -------------------------------------------------------------
    print("[Phase 6/9] Rendering Scene 5: Interactive Split-Curtain Verification...")
    shot_c50 = shots['05_verification_curtain_50']
    shot_c25 = shots['06_verification_curtain_25']
    shot_c75 = shots['07_verification_curtain_75']

    # Sweep sequence: 50% -> 25% (0-90) -> 75% (90-210) -> 50% (210-300)
    for f in range(300):
        t = (1290 + f) / FPS
        if f < 90:
            sw_ratio = 0.50 - 0.25 * (f / 90.0)
            active_shot = shot_c25 if f > 45 else shot_c50
        elif f < 210:
            sw_ratio = 0.25 + 0.50 * ((f - 90) / 120.0)
            active_shot = shot_c75 if (f - 90) > 60 else shot_c25
        else:
            sw_ratio = 0.75 - 0.25 * ((f - 210) / 90.0)
            active_shot = shot_c50 if (f - 210) > 45 else shot_c75

        # Cursor tracks the slider line
        slider_x = int(WIDTH * sw_ratio)
        cur_x = slider_x
        cur_y = 520 + int(20 * math.sin(t * 3.0))

        pil_img = Image.fromarray(cv2.cvtColor(active_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Draw glowing curtain line overlay
        draw.line([slider_x, 180, slider_x, 900], fill=(34, 211, 238, 255), width=3)
        # Slider handle
        draw.ellipse([slider_x - 14, cur_y - 14, slider_x + 14, cur_y + 14], fill=(6, 182, 212, 255), outline=(255, 255, 255, 255), width=2)
        draw.text((slider_x - 28, cur_y - 32), "◄  ►", font=font_mono_small, fill=(255, 255, 255, 255))

        draw_hud_lower_third(
            draw,
            title="SCENE 05 // REAL-TIME SUB-PIXEL SPLIT-CURTAIN VERIFICATION",
            details="Dynamic curtain sweep demonstrates zero rim discontinuity between Chandrayaan-2 and LROC NAC basemap.",
            badges=[f"CURTAIN: {int(sw_ratio * 100)}%", "ZERO RIM DISCONTINUITY", "SUB-PIXEL LOCK"],
            time_sec=t,
            category="PHOTOGRAMMETRIC VERIFICATION"
        )
        draw_mouse_cursor(draw, cur_x, cur_y, 0.40) # dragging
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 6: UNALIGNED VS REGISTERED & 8x8 CHECKERBOARD (53.0s - 62.0s, 270 frames)
    # -------------------------------------------------------------
    print("[Phase 7/9] Rendering Scene 6: Drift Elimination & 8x8 Checkerboard...")
    unaligned_shot = shots['08_verification_unaligned']
    checker_shot = shots['09_verification_checkerboard']

    # First 120 frames: Raw Unaligned (showing 14.28 px drift) vs Locked
    for f in range(120):
        t = (1590 + f) / FPS
        active = unaligned_shot if f < 75 else registered_shot
        pil_img = Image.fromarray(cv2.cvtColor(active, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        status_badge = "14.281 px UNALIGNED SENSOR DRIFT" if f < 75 else "0.2641 px SUB-PIXEL LOCKED"
        draw_hud_lower_third(
            draw,
            title="SCENE 06 // SENSOR DRIFT CORRECTION (14.28 px → 0.26 px)",
            details="Simulates uncorrected orbital sensor drift, then applies deep homography matrix to snap into sub-pixel lock.",
            badges=[status_badge, "USAC_MAGSAC", "WARP PERSPECTIVE"],
            time_sec=t,
            category="BEFORE / AFTER COMPARISON"
        )
        draw_mouse_cursor(draw, 500, 145, 0.6 if f == 75 else 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # Next 150 frames: 8x8 Checkerboard Inspector Mode
    for f in range(150):
        t = (1710 + f) / FPS
        pil_img = Image.fromarray(cv2.cvtColor(checker_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        draw_hud_lower_third(
            draw,
            title="SCENE 06 // 8x8 INTERLEAVED CHECKERBOARD MATRIX INSPECTION",
            details="Alternating tiles between Chandrayaan-2 and LROC NAC prove perfect geometric continuity across tile boundaries.",
            badges=["8x8 MATRIX TILES", "SEAMLESS BOUNDARIES", "CONTOUR EDGES ON"],
            time_sec=t,
            category="BOUNDARY CONTINUITY"
        )
        draw_mouse_cursor(draw, 960 + 40 * math.sin(t), 540 + 30 * math.cos(t), 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 7: STEP 4 - ACCURACY & TELEMETRY REPORT (62.0s - 71.0s, 270 frames)
    # -------------------------------------------------------------
    print("[Phase 8/9] Rendering Scene 7: Accuracy & Telemetry Metrics Report...")
    telemetry_shot = shots['10_telemetry_report']

    for f in range(270):
        t = (1860 + f) / FPS
        p = f / 270.0
        pil_img = Image.fromarray(cv2.cvtColor(telemetry_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Highlight box around RMSE metric card and Homography Matrix
        draw.rectangle([45, 180, 550, 480], outline=(52, 211, 153, int(180 + 70 * math.sin(t * 4.0))), width=3)
        draw.rectangle([580, 180, 1250, 680], outline=(34, 211, 238, int(180 + 70 * math.cos(t * 4.0))), width=2)
        draw.text((60, 190), "▶ REPROJECTION RMSE: 0.2641 px  [TARGET <0.50 px EXCEEDED]", font=font_mono_small, fill=(52, 211, 153, 255))
        draw.text((600, 190), "▶ DECOMPOSED HOMOGRAPHY (SCALE: 1.998x, ROT: +0.04°)", font=font_mono_small, fill=(34, 211, 238, 255))

        draw_hud_lower_third(
            draw,
            title="SCENE 07 // RIGOROUS PHOTOGRAMMETRIC GROUND TRUTH BENCHMARKS",
            details="Decomposed scale (1.998x) matches physical 2:1 ground sampling ratio. Sub-pixel RMSE beats ISRO target by 47%.",
            badges=["RMSE: 0.2641 px", "SCALE: 1.998x", "SSIM: 0.8412", "INLIERS: 93.07%"],
            time_sec=t,
            category="MATHEMATICAL VALIDATION"
        )
        draw_mouse_cursor(draw, 350 + int(50 * math.sin(t * 1.5)), 300, 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 8: REGISTRATION VALIDATION GUARDRAIL (71.0s - 79.0s, 240 frames)
    # -------------------------------------------------------------
    print("[Phase 9/9] Rendering Scene 8: Numerical Guardrail Failure Defense...")
    guardrail_shot = shots['11_guardrail_failure_alert']

    for f in range(240):
        t = (2130 + f) / FPS
        # Alert pulse effect
        pulse_alpha = int(190 + 65 * math.sin(t * 8.0))

        pil_img = Image.fromarray(cv2.cvtColor(guardrail_shot, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Flashing alert border around LROC canvas
        draw.rectangle([WIDTH // 2, 180, WIDTH - 40, 895], outline=(239, 68, 68, pulse_alpha), width=3)

        draw_hud_lower_third(
            draw,
            title="SCENE 08 // ACTIVE NUMERICAL GUARDRAIL (ANTI-DEGENERACY)",
            details="Rejects non-overlapping swaths via Hartley condition number gating; suppresses false projections and sets table coordinates to N/A.",
            badges=["GUARDRAIL ACTIVE", "ZERO FALSE MATCHES", "HARTLEY COND < 10^4"],
            time_sec=t,
            category="FAIL-SAFE SAFETY SHIELD"
        )
        draw_mouse_cursor(draw, 1400, 500, 0.0)
        write_frame(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # SECTION 9: OUTRO & MISSION SUMMARY SCOREBOARD (79.0s - 85.0s, 180 frames)
    # -------------------------------------------------------------
    print("[Final Phase] Rendering Closing Benchmark Scoreboard...")
    for f in range(180):
        t = (2370 + f) / FPS
        frame = render_outro_card(t, duration=6.0)
        write_frame(frame)

    proc.stdin.close()
    proc.wait()
    print(f"\n[DONE] Video successfully compiled and written to: {OUTPUT_VIDEO}")
    print(f"Total Frames: {total_frames} ({total_frames / FPS:.2f} seconds @ 30 FPS)")
    print(f"File Size: {os.path.getsize(OUTPUT_VIDEO):,} bytes\n")

if __name__ == '__main__':
    build_video()
