import numpy as np
import cv2
import math

class LunarImageGenerator:
    """
    Generates synthetic lunar crater scenes with controllable illumination,
    scale variations, rotation, perspective tilt, and sensor noise.
    Used for testing and benchmark evaluation of registration algorithms.
    """
    def __init__(self, width=800, height=800, seed=42):
        self.width = width
        self.height = height
        self.seed = seed

    def generate_lunar_surface(self, num_craters=40, sun_angle_deg=45, noise_level=5.0):
        """
        Generates a synthetic lunar heightmap and renders lighting based on sun angle.
        """
        np.random.seed(self.seed)
        
        # Base regolith texture with Perlin-like fractal noise
        surface = np.ones((self.height, self.width), dtype=np.float32) * 128.0
        
        # Add background terrain roughness
        noise_base = np.random.normal(0, 8.0, (self.height, self.width)).astype(np.float32)
        noise_base = cv2.GaussianBlur(noise_base, (31, 31), 10.0)
        surface += noise_base
        
        # Generate random crater parameters
        craters = []
        for _ in range(num_craters):
            cx = np.random.uniform(50, self.width - 50)
            cy = np.random.uniform(50, self.height - 50)
            r = np.random.uniform(12, 70)
            depth = np.random.uniform(25, 90)
            craters.append((cx, cy, r, depth))
            
        # Draw craters onto heightmap/surface with illumination shadow
        sun_rad = math.radians(sun_angle_deg)
        dx = math.cos(sun_rad)
        dy = math.sin(sun_rad)
        
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        
        for (cx, cy, r, depth) in craters:
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            r_sq = r**2
            
            # Interior bowl mask
            inside = dist_sq <= r_sq
            rim = (dist_sq > r_sq) & (dist_sq <= (1.3 * r)**2)
            
            # Crater bowl profile
            bowl = -depth * np.cos((np.sqrt(dist_sq[inside]) / r) * (np.pi / 2))
            surface[inside] += bowl
            
            # Ejecta rim profile
            rim_dist = np.sqrt(dist_sq[rim]) - r
            rim_height = (depth * 0.25) * np.exp(-rim_dist / (r * 0.2))
            surface[rim] += rim_height

        # Calculate lighting gradients based on sun direction (directional derivative)
        grad_x = cv2.Sobel(surface, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(surface, cv2.CV_32F, 0, 1, ksize=5)
        
        illumination = 128.0 + (grad_x * dx + grad_y * dy) * 1.5
        
        # Add sensor shot noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, (self.height, self.width))
            illumination += noise

        illumination = np.clip(illumination, 0, 255).astype(np.uint8)
        return illumination, craters

    def generate_pair(self, scale=1.1, angle_deg=12.0, tx=25.0, ty=-15.0, 
                      sun_angle1=45, sun_angle2=65, perspective_strength=0.0003):
        """
        Creates a synthetic pair: Reference Image and Source (moving) Image
        with known geometric transform ground truth and illumination difference.
        """
        # Reference image
        ref_img, craters = self.generate_lunar_surface(num_craters=45, sun_angle_deg=sun_angle1, noise_level=3.0)
        
        # Base source image with different sun angle (illumination variation)
        src_base, _ = self.generate_lunar_surface(num_craters=45, sun_angle_deg=sun_angle2, noise_level=5.0)
        
        # Construct transformation matrix H
        cx, cy = self.width / 2.0, self.height / 2.0
        rad = math.radians(angle_deg)
        
        # Similarity / Rigid matrix
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        # Build 3x3 Homography with perspective distortion
        H_gt = np.eye(3, dtype=np.float32)
        H_gt[0:2, :] = M_rot
        H_gt[2, 0] = perspective_strength
        H_gt[2, 1] = -perspective_strength * 0.5
        
        # Warp base source image using Homography
        src_warped = cv2.warpPerspective(src_base, H_gt, (self.width, self.height), 
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=30)
        
        return ref_img, src_warped, H_gt
