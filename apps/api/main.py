import os
import io
import base64
from typing import Optional, Dict, Any
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, Field

load_dotenv()

from apps.api.services.pipeline import (
    run_full_analysis,
    get_model,
    device,
    THRESHOLD_LOW,
    THRESHOLD_HIGH
)

# Create FastAPI app
app = FastAPI(
    title="IntegriDoc Forensic API",
    description="Explainable document forgery detection, multi-modal localization, and VLM analysis.",
    version="1.1.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure frontend directory exists for static mounting
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if not os.path.exists(FRONTEND_DIR):
    os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount the static frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    model_type: str
    segmentation_loaded: bool
    vlm_configured: bool
    thresholds: Dict[str, float]

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns detailed API health and inference engine telemetry."""
    model, m_type, seg_model = get_model()
    return HealthResponse(
        status="ok",
        version="1.1.0",
        device=str(device),
        model_type=m_type,
        segmentation_loaded=seg_model is not None,
        vlm_configured=bool(os.environ.get("GEMINI_API_KEY")),
        thresholds={"low": THRESHOLD_LOW, "high": THRESHOLD_HIGH}
    )

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the main Forensic Workstation UI."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>IntegriDoc Frontend Initializing...</h1>", status_code=200)

@app.get("/documentation", response_class=HTMLResponse)
def serve_docs_ui():
    path = os.path.join(FRONTEND_DIR, "documentation.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Documentation</h1>", status_code=200)

@app.get("/support", response_class=HTMLResponse)
def serve_support_ui():
    path = os.path.join(FRONTEND_DIR, "support.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Support</h1>", status_code=200)

@app.get("/privacy", response_class=HTMLResponse)
def serve_privacy_ui():
    path = os.path.join(FRONTEND_DIR, "privacy.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Privacy Policy</h1>", status_code=200)

@app.post("/v1/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0, description="Target page index for multi-page PDF documents (0-indexed)."),
    threshold_low: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional custom lower threshold for REAL verdict."),
    threshold_high: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional custom upper threshold for TAMPERED verdict.")
):
    """
    Accepts an image or PDF document upload and executes the full forensic analysis pipeline.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
            
        t_low = threshold_low if threshold_low is not None else THRESHOLD_LOW
        t_high = threshold_high if threshold_high is not None else THRESHOLD_HIGH
        
        result = run_full_analysis(
            document_bytes=content,
            page_index=page,
            threshold_low=t_low,
            threshold_high=t_high
        )
        return result
    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Forensic Processing Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
