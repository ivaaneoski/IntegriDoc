import io
import base64
from typing import Union, Tuple, Optional
from PIL import Image, ImageOps

# Guard against decompression bomb vulnerabilities
Image.MAX_IMAGE_PIXELS = 60_000_000

def is_pdf(file_bytes: bytes) -> bool:
    """Check if the provided byte sequence starts with the PDF magic header."""
    return file_bytes.startswith(b"%PDF")

def load_document_image(
    file_bytes: bytes, 
    page_index: int = 0,
    render_scale: float = 2.0
) -> Tuple[Image.Image, int]:
    """
    Safely loads and pre-processes document bytes (Images or PDFs).
    
    Returns:
        Tuple of (PIL.Image in RGB mode with corrected EXIF orientation, total_page_count)
    
    Raises:
        ValueError: If file data is invalid, empty, or cannot be decoded.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded document payload is empty (0 bytes).")

    # Handle PDF Documents
    if is_pdf(file_bytes):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            total_pages = len(pdf)
            if total_pages == 0:
                raise ValueError("PDF document has no pages.")
            
            # Select target page (bounded)
            target_idx = max(0, min(page_index, total_pages - 1))
            page = pdf[target_idx]
            
            # Render to bitmap at crisp resolution
            bitmap = page.render(scale=render_scale)
            pil_image = bitmap.to_pil().convert('RGB')
            return pil_image, total_pages
        except ImportError:
            raise ValueError(
                "PDF format detected, but 'pypdfium2' is not available. Please install pypdfium2."
            )
        except Exception as e:
            raise ValueError(f"Failed to parse and render PDF document: {str(e)}")

    # Handle Standard Image formats (JPEG, PNG, WEBP, TIFF, BMP, etc.)
    try:
        raw_img = Image.open(io.BytesIO(file_bytes))
        
        # Apply EXIF rotation if present (e.g. smartphone identity scans)
        try:
            transposed_img = ImageOps.exif_transpose(raw_img)
            img = transposed_img if transposed_img is not None else raw_img
        except Exception:
            img = raw_img
            
        img = img.convert('RGB')
        
        # Ensure dimensions are valid
        w, h = img.size
        if w < 10 or h < 10:
            raise ValueError(f"Image dimensions are too small to analyze ({w}x{h} px).")
            
        return img, 1
    except Exception as e:
        raise ValueError(f"Failed to decode image file. Unsupported or corrupted format: {str(e)}")

def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Encodes a PIL Image into a base64 string."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG":
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
