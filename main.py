import os
import io
import time
import hashlib
import psutil
from collections import deque
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    Form, 
    Header, 
    HTTPException, 
    Security, 
    Depends, 
    Request, 
    status,
    Query
)
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY", "my_secure_api_key_12345")
ADMIN_KEY = os.getenv("ADMIN_KEY", "my_secure_admin_key_67890")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.6"))
DEFAULT_TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.4"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "100"))

NUM_THREADS = int(os.getenv("NUM_THREADS", "0"))
if NUM_THREADS > 0:
    torch.set_num_threads(NUM_THREADS)


# Server state
SERVER_START_TIME = time.time()
TOTAL_REQUESTS = 0
CACHE_HITS = 0

# In-Memory History / Deduplication store (circular deque + fast lookup dict)
history_deque: deque = deque(maxlen=HISTORY_LIMIT)
history_lookup: Dict[str, Dict[str, Any]] = {}
history_id_counter = 0

# Model globals
# Load CLIP model on startup
print(f"Loading CLIP model '{MODEL_NAME}' on {DEVICE}...", flush=True)
t_load_start = time.time()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
clip_model.eval()
print(f"Model loaded successfully in {time.time() - t_load_start:.2f}s.", flush=True)

def extract_tensor(out):
    if hasattr(out, 'text_embeds') and out.text_embeds is not None:
        return out.text_embeds
    elif hasattr(out, 'image_embeds') and out.image_embeds is not None:
        return out.image_embeds
    elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :]
    elif isinstance(out, (list, tuple)):
        return out[0]
    return out

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Shutting down embedding server...", flush=True)

app = FastAPI(
    title="Lightweight Image & Text Embedding Server",
    description="High-performance embedding service powered by OpenAI CLIP with deduplication and in-memory history.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Query(None, alias="api_key")
):
    key = x_api_key or query_key
    if not key or key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Provide 'X-API-Key' header or '?api_key=' parameter."
        )
    return key

def verify_admin_key(
    x_admin_key: Optional[str] = Security(admin_key_header),
    query_key: Optional[str] = Query(None, alias="admin_key")
):
    key = x_admin_key or query_key
    if not key or key != ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing Admin Key. Provide 'X-Admin-Key' header or '?admin_key=' parameter."
        )
    return key

# Helper: Compute SHA-256 for caching & deduplication
def compute_sha256(image_bytes: Optional[bytes] = None, text: Optional[str] = None) -> str:
    hasher = hashlib.sha256()
    if image_bytes:
        hasher.update(b"IMG:")
        hasher.update(image_bytes)
    if text:
        hasher.update(b"|TXT:")
        hasher.update(text.strip().lower().encode("utf-8"))
    return hasher.hexdigest()

def record_history(
    ip: str, 
    req_type: str, 
    sha: str, 
    vector: List[float], 
    text: Optional[str] = None
) -> Dict[str, Any]:
    global history_id_counter
    history_id_counter += 1
    
    entry = {
        "id": history_id_counter,
        "ip": ip,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "type": req_type,
        "sha256": sha,
        "text": text if text else None,
        "vector": vector
    }
    
    # If history is full, remove oldest from lookup table
    if len(history_deque) == history_deque.maxlen:
        oldest = history_deque[0]
        history_lookup.pop(oldest["sha256"], None)
        
    history_deque.append(entry)
    history_lookup[sha] = entry
    return entry

# --- ENDPOINTS ---

@app.get("/health", tags=["System"])
async def health_check():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=None)
    uptime_sec = round(time.time() - SERVER_START_TIME, 1)
    
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "uptime_seconds": uptime_sec,
        "memory_rss_mb": round(ram_mb, 1),
        "cpu_usage_percent": cpu_percent,
        "total_requests": TOTAL_REQUESTS,
        "cache_hits": CACHE_HITS,
        "in_memory_history_count": len(history_deque)
    }

class TextEmbedRequest(BaseModel):
    text: str = Field(..., description="Text query to embed", min_length=1)

@app.post("/embed", tags=["Embedding"])
async def embed(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image file upload"),
    text: Optional[str] = Form(None, description="Optional text or search query"),
    image_weight: float = Form(DEFAULT_IMAGE_WEIGHT, description="Weight for image in fused vector"),
    text_weight: float = Form(DEFAULT_TEXT_WEIGHT, description="Weight for text in fused vector"),
    _auth: str = Depends(verify_api_key)
):
    """
    Unified embedding endpoint. Accepts:
    - Image only (multipart file)
    - Text only (form field 'text')
    - Image + Text (multipart file + form field 'text') -> returns fused vector
    """
    global TOTAL_REQUESTS, CACHE_HITS
    TOTAL_REQUESTS += 1
    client_ip = request.client.host if request.client else "unknown"

    image_bytes = None
    if file is not None:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            image_bytes = None

    clean_text = text.strip() if text and text.strip() else None

    if not image_bytes and not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either an image file, a text string, or both."
        )

    # Determine type and SHA256
    if image_bytes and clean_text:
        req_type = "image+text"
    elif image_bytes:
        req_type = "image"
    else:
        req_type = "text"

    sha = compute_sha256(image_bytes, clean_text)

    # In-memory deduplication check
    if sha in history_lookup:
        CACHE_HITS += 1
        cached_entry = history_lookup[sha]
        return {
            "success": True,
            "type": req_type,
            "sha256": sha,
            "dim": len(cached_entry["vector"]),
            "cached": True,
            "latency_ms": 0.0,
            "embedding": cached_entry["vector"]
        }

    # Generate Embeddings via CLIP
    t0 = time.time()
    try:
        image_vec = None
        text_vec = None

        if image_bytes:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat_img = extract_tensor(clip_model.get_image_features(**img_inputs))
                feat_img = feat_img / feat_img.norm(dim=-1, keepdim=True)
                image_vec = feat_img.cpu().numpy()[0]

        if clean_text:
            txt_inputs = clip_processor(text=[clean_text], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat_txt = extract_tensor(clip_model.get_text_features(**txt_inputs))
                feat_txt = feat_txt / feat_txt.norm(dim=-1, keepdim=True)
                text_vec = feat_txt.cpu().numpy()[0]

        # Combine or select final vector
        if image_vec is not None and text_vec is not None:
            combined = (image_weight * image_vec) + (text_weight * text_vec)
            norm = np.linalg.norm(combined)
            final_vec = (combined / norm if norm > 0 else combined).tolist()
        elif image_vec is not None:
            final_vec = image_vec.tolist()
        else:
            final_vec = text_vec.tolist()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    latency_ms = round((time.time() - t0) * 1000, 2)

    # Store in last 100 history cache
    record_history(
        ip=client_ip,
        req_type=req_type,
        sha=sha,
        vector=final_vec,
        text=clean_text
    )

    return {
        "success": True,
        "type": req_type,
        "sha256": sha,
        "dim": len(final_vec),
        "cached": False,
        "latency_ms": latency_ms,
        "embedding": final_vec
    }

# Also support direct JSON payload for pure text queries: POST /embed/text
@app.post("/embed/text", tags=["Embedding"])
async def embed_text_json(
    request: Request,
    payload: TextEmbedRequest,
    _auth: str = Depends(verify_api_key)
):
    """Convenience JSON endpoint for text-only search embeddings."""
    return await embed(
        request=request,
        file=None,
        text=payload.text,
        _auth=_auth
    )

# --- ADMIN ENDPOINTS ---

@app.get("/admin/history", tags=["Admin"])
async def get_history(
    include_vectors: bool = Query(True, description="Whether to include full vector arrays"),
    limit: int = Query(100, ge=1, le=100, description="Max entries to return"),
    _admin: str = Depends(verify_admin_key)
):
    """Returns the last 100 in-memory embedding records with IP, SHA, and vectors."""
    records = list(history_deque)[-limit:]
    records.reverse()  # Latest first
    
    if not include_vectors:
        return [
            {
                "id": r["id"],
                "ip": r["ip"],
                "timestamp": r["timestamp"],
                "type": r["type"],
                "sha256": r["sha256"],
                "text": r["text"],
                "dim": len(r["vector"])
            }
            for r in records
        ]
        
    return records

@app.delete("/admin/history", tags=["Admin"])
async def clear_history(_admin: str = Depends(verify_admin_key)):
    """Clears the in-memory history and deduplication cache."""
    history_deque.clear()
    history_lookup.clear()
    return {"success": True, "message": "In-memory history and deduplication cache cleared."}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
