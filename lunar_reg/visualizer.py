import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_checkerboard(img1, img2, square_size=50):
    """
    Creates an alternating checkerboard pattern between img1 (ref) and img2 (warped src)
    to visually verify alignment accuracy across feature boundaries.
    """
    h, w = img1.shape[:2]
    checkerboard = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            y_end = min(y + square_size, h)
            x_end = min(x + square_size, w)
            
            # Alternate squares
            if ((y // square_size) + (x // square_size)) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]
                
    return checkerboard

def create_residual_error_map(ref_img, warped_src):
    """
    Generates a colorized difference/residual error map highlighting registration offsets.
    """
    diff = cv2.absdiff(ref_img, warped_src)
    # Apply color map for visual clarity
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    # Mask out region outside warped source bounding box
    mask = (warped_src == 0)
    diff_colored[mask] = [0, 0, 0]
    return diff_colored

def draw_matches_side_by_side(ref_img, src_img, ref_pts, src_pts, max_draw=100):
    """
    Draws feature correspondences between reference and source image.
    """
    h1, w1 = ref_img.shape[:2]
    h2, w2 = src_img.shape[:2]
    
    out_h = max(h1, h2)
    out_w = w1 + w2
    
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Convert grayscale to BGR canvas
    ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img
    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR) if len(src_img.shape) == 2 else src_img
    
    canvas[:h1, :w1] = ref_bgr
    canvas[:h2, w1:w1+w2] = src_bgr
    
    num_draw = min(len(ref_pts), max_draw)
    indices = np.random.choice(len(ref_pts), num_draw, replace=False) if len(ref_pts) > max_draw else range(len(ref_pts))
    
    for idx in indices:
        pt1 = (int(ref_pts[idx][0]), int(ref_pts[idx][1]))
        pt2 = (int(src_pts[idx][0] + w1), int(src_pts[idx][1]))
        
        # Color based on position for distinction
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        )
        
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        
    return canvas

def create_uniformity_heatmap(pts, img_shape, grid_size=8):
    """
    Generates a 2D spatial keypoint distribution heatmap.
    """
    h, w = img_shape[:2]
    counts, xedges, yedges = np.histogram2d(
        pts[:, 1], pts[:, 0], 
        bins=[grid_size, grid_size], 
        range=[[0, h], [0, w]]
    )
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(counts, cmap='YlOrRd', interpolation='nearest')
    ax.set_title("Keypoint Spatial Density Distribution")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Save figure to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data
