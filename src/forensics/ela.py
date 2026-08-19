import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io

def calculate_ela(image: Image.Image, quality: int = 90, percentile: float = 95.0) -> dict:
    """
    Calculates Error Level Analysis (ELA) for an image.
    
    Args:
        image: PIL Image in RGB format
        quality: JPEG compression quality for ELA
        percentile: Percentile for normalization
        
    Returns:
        dict containing:
        - ela_image: PIL Image of the ELA heatmap
        - ela_array: Normalized numpy array
        - max_diff: Maximum difference value
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Recompress the image
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    
    # Calculate absolute difference
    diff = ImageChops.difference(image, recompressed)
    
    # Get max difference for normalization
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
        
    # Calculate scale dynamically based on percentile
    diff_arr = np.array(diff)
    norm_val = np.percentile(diff_arr, percentile)
    if norm_val == 0:
        norm_val = max_diff
    
    scale = 255.0 / norm_val
    
    # Enhance difference to make it visible
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)
    
    # Convert to grayscale for consistent array output
    ela_gray = ela_image.convert('L')
    ela_array = np.array(ela_gray) / 255.0
    
    return {
        "ela_image": ela_image,
        "ela_array": ela_array,
        "max_diff": max_diff,
        "norm_val": norm_val
    }
