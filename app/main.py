"""
FastAPI application exposing the Speech-to-Text model as a REST API.

Endpoints:
    GET  /                  -> API info
    GET  /health            -> health check + model status
    POST /transcribe        -> upload an audio file, get transcript back
    GET  /docs              -> auto-generated Swagger UI (built into FastAPI)
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import SpeechToTextEngine

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best_model.pt")
ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
MAX_FILE_SIZE_MB = 25

engine: SpeechToTextEngine | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP (not per-request — this is the key FastAPI pattern)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print(f"Loading model from {CHECKPOINT_PATH} ...")
    engine = SpeechToTextEngine(CHECKPOINT_PATH)
    print(f"Model loaded. Epoch={engine.epoch}  Params={engine.n_params:,}")
    yield
    engine = None


app = FastAPI(
    title="Speech-to-Text API",
    description=(
        "A from-scratch Automatic Speech Recognition API built with a "
        "CNN + Bidirectional GRU + CTC architecture, trained on LibriSpeech. "
        "Upload an audio file and receive a transcript."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser-based clients (e.g. a frontend demo) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Speech-to-Text API",
        "architecture": "CNN + BiGRU x5 + CTC (greedy decoding)",
        "dataset": "LibriSpeech dev-clean (~5.4h)",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "transcribe": "POST /transcribe (multipart/form-data, field name: file)"
        }
    }


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "epoch": engine.epoch,
        "val_wer": engine.val_wer,
        "val_cer": engine.val_cer,
        "parameters": engine.n_params,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Save upload to a temp file (torchaudio needs a path, not a stream)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        os.unlink(tmp_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB."
        )

    try:
        result = engine.transcribe(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)
