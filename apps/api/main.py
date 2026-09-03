import os
import io
import base64
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

load_dotenv()

from apps.api.services.pipeline import run_full_analysis

# Create FastAPI app
app = FastAPI(title="IntegriDoc Forensic API", version="1.0.0")


# Ensure frontend directory exists for static mounting
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if not os.path.exists(FRONTEND_DIR):
    os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount the static frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the API health status."""
    return HealthResponse(status="ok", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the main Forensic Workstation UI."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/v1/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts an image upload and executes the full forensic analysis pipeline.
    """
    content = await file.read()
    return run_full_analysis(content)

if __name__ == "__main__":
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)

