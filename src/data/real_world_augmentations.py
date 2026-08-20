import random
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

def apply_perspective_warp(image: Image.Image, max_distortion: float = 0.05) -> Image.Image:
    """
    Simulates taking a photo of a document from a slight angle.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    # Source points: Corners of the image
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Target points: perturbed corners
    dx1 = random.uniform(0, max_distortion * w)
    dy1 = random.uniform(0, max_distortion * h)
    dx2 = random.uniform(0, max_distortion * w)
    dy2 = random.uniform(0, max_distortion * h)
    dx3 = random.uniform(0, max_distortion * w)
    dy3 = random.uniform(0, max_distortion * h)
    dx4 = random.uniform(0, max_distortion * w)
    dy4 = random.uniform(0, max_distortion * h)
    
    dst_pts = np.float32([
        [dx1, dy1],
        [w - dx2, dy2],
        [w - dx3, h - dy3],
        [dx4, h - dy4]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)

def apply_lighting_shadow(image: Image.Image) -> Image.Image:
    """
    Simulates non-uniform ambient room lighting or cast shadow across the paper.
    """
    img_np = np.array(image, dtype=np.float32)
    h, w = img_np.shape[:2]
    
    # Create random linear or radial gradient
    if random.random() < 0.5:
        # Linear lighting gradient
        x = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), w)
        y = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), h)
        xx, yy = np.meshgrid(x, y)
        gradient = (xx + yy) / 2.0
    else:
        # Radial shadow (e.g. phone shadow or overhead lamp)
        cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(w**2 + h**2)
        gradient = 1.0 - (dist / max_dist) * random.uniform(0.15, 0.35)
        
    gradient = np.clip(gradient, 0.5, 1.2)[:, :, np.newaxis]
    lit_img = np.clip(img_np * gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(lit_img)

def apply_camera_sensor_noise(image: Image.Image, noise_level: float = 0.02) -> Image.Image:
    """
    Simulates smartphone camera sensor ISO grain / noise.
    """
    img_np = np.array(image, dtype=np.float32) / 255.0
    
    # Gaussian noise
    sigma = random.uniform(0.005, noise_level)
    gauss = np.random.normal(0, sigma, img_np.shape)
    
    noisy = np.clip(img_np + gauss, 0.0, 1.0) * 255.0
    return Image.fromarray(noisy.astype(np.uint8))

def apply_jpeg_recompression(image: Image.Image, quality_range=(40, 90)) -> Image.Image:
    """
    Simulates sending the document over WhatsApp/Telegram or saving at lower JPEG quality.
    """
    quality = random.randint(quality_range[0], quality_range[1])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_camera_capture_pipeline(image: Image.Image) -> Image.Image:
    """
    Full real-world degradation pipeline simulating physical capture.
    """
    # 1. Perspective tilt
    if random.random() < 0.7:
        image = apply_perspective_warp(image, max_distortion=0.04)
        
    # 2. Lighting / shadows
    if random.random() < 0.8:
        image = apply_lighting_shadow(image)
        
    # 3. Slight camera blur (optional out-of-focus)
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        
    # 4. Sensor noise
    if random.random() < 0.7:
        image = apply_camera_sensor_noise(image)
        
    # 5. JPEG recompression
    image = apply_jpeg_recompression(image, quality_range=(50, 88))
    
    return image
