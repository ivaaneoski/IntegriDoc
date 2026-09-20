import cv2
import numpy as np
from PIL import Image

def calculate_gaussian_residual(image: Image.Image, sigma: float = 2.0) -> dict:
    """
    Calculates the Gaussian high-pass residual.
    residual = image - gaussian_blur(image, sigma)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_arr = np.array(image, dtype=np.float32)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img_arr, (0, 0), sigma)
    
    # Calculate residual
    residual = img_arr - blurred
    
    # Take absolute magnitude
    magnitude = np.abs(residual)
    
    # Normalize to 0-255 for visualization
    max_val = np.max(magnitude)
    if max_val == 0:
        max_val = 1.0
        
    normalized = (magnitude / max_val) * 255.0
    res_image = Image.fromarray(normalized.astype(np.uint8))
    
    # Compute mean anomaly scalar and variance
    score = float(np.mean(magnitude))
    variance = float(np.var(magnitude))
    
    return {
        "residual_image": res_image,
        "residual_array": magnitude,
        "score": round(score, 4),
        "variance": round(variance, 4)
    }

def calculate_laplacian_residual(image: Image.Image) -> dict:
    """
    Calculates the Laplacian residual.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Convert to grayscale for laplacian to keep it simple
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    magnitude = np.abs(laplacian)
    
    # Normalize
    max_val = np.max(magnitude)
    if max_val == 0:
        max_val = 1.0
        
    normalized = (magnitude / max_val) * 255.0
    res_image = Image.fromarray(normalized.astype(np.uint8), mode='L')
    
    score = float(np.mean(magnitude))
    variance = float(np.var(magnitude))
    
    return {
        "residual_image": res_image,
        "residual_array": magnitude,
        "score": round(score, 4),
        "variance": round(variance, 4)
    }
