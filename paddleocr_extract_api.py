"""Standalone PDF text-extraction API using PaddleOCR.

Single, self-contained file — no dependency on any other file in this
project. Drop it into another project as-is.

Install:
    pip install fastapi "uvicorn[standard]" python-multipart pymupdf pillow numpy paddlepaddle paddleocr

Run:
    uvicorn paddleocr_extract_api:app --host 0.0.0.0 --port 8002
    # or:  python paddleocr_extract_api.py

Usage:
    curl -X POST http://localhost:8002/extract-text \
      -F "files=@scanned.pdf" \
      -F "files=@digital.pdf"

Each page is classified automatically: if it already has enough embedded
text it's read directly (fast, no OCR); otherwise it's rendered to an
image and run through PaddleOCR. Response includes per-page and overall
processing time in seconds (queue-wait time excluded from per-page
figures — see _ocr_worker).

Notes carried over from testing this in a larger project:
- PaddlePaddle has no GPU/MPS support on Apple Silicon (CPU-only there).
  Whether PaddleOCR is faster or slower than an alternative like EasyOCR
  depends heavily on document content/language — it was measured slower
  on a simple English page, but about the same speed (and visibly more
  accurate) on a dense multilingual page. Benchmark on your own documents.
- OCR runs in separate worker *processes* (not threads) via
  ProcessPoolExecutor with the "spawn" start method forced explicitly
  (not left to the OS default) — this is deliberate, not incidental.
  Running heavy, natively-threaded OCR libraries concurrently in threads
  within one process was found to crash reproducibly in a related EasyOCR
  setup (no traceback, consistent with a native thread-safety bug, not a
  clean OOM kill); separate processes avoid sharing that native state.
  "spawn" (rather than the Linux default "fork") avoids a related hazard:
  forking a process that already has native-threaded libraries loaded is
  a known source of deadlocks/crashes, and spawn is also the only option
  Windows supports, so it keeps behavior identical across platforms.
- PADDLE_OCR_WORKERS defaults to 1. Raising it does not reliably speed up
  a single multi-page document on a machine with a mixed performance/
  efficiency core layout (e.g. Apple M-series) — two heavy OCR workers end
  up contending for the same handful of real fast cores regardless of
  thread-count tuning. It can still help availability when separate
  concurrent requests arrive, just not single-document latency, on that
  class of hardware.
"""

import asyncio
import io
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Union

import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Config (environment variables)
# --------------------------------------------------------------------------

OCR_DPI = int(os.environ.get("OCR_DPI", "200"))
DIGITAL_TEXT_MIN_CHARS = int(os.environ.get("DIGITAL_TEXT_MIN_CHARS", "20"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
# See the module docstring re: why raising this may not help on Apple
# Silicon-class hardware.
PADDLE_OCR_WORKERS = int(os.environ.get("PADDLE_OCR_WORKERS", "1"))

PDF_MAGIC = b"%PDF"

# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class PageResult(BaseModel):
    page_number: int
    method: str  # "digital" or "ocr"
    text: str
    confidence: Optional[float] = None
    processing_time_seconds: float


class ExtractionError(BaseModel):
    filename: str
    error: str


class TextExtractionResult(BaseModel):
    filename: str
    num_pages: int
    pages: list[PageResult]
    full_text: str
    processing_time_seconds: float


class TextExtractionResponse(BaseModel):
    results: list[TextExtractionResult]
    errors: list[ExtractionError] = Field(default_factory=list)


# --------------------------------------------------------------------------
# PDF page classification: digital text vs. needs-OCR
# --------------------------------------------------------------------------

# Court-filing systems and scanners commonly stamp a short text watermark
# (e.g. "NOT A CERTIFIED COPY", a filing header) as real embedded text on
# top of an otherwise fully scanned page image. A character-count threshold
# alone can't tell that apart from a genuinely digital page, so any page
# carrying an image covering roughly the full page is treated as scanned
# regardless of how much stray text sits on top of it.
_FULL_PAGE_IMAGE_AREA_RATIO = 0.85


@dataclass
class _PreparedPage:
    page_number: int
    kind: Literal["digital", "ocr"]
    payload: Union[str, bytes]
    prepare_seconds: float


def _extract_digital_text(page: "fitz.Page") -> str:
    return page.get_text().strip()


def _has_full_page_image(page: "fitz.Page") -> bool:
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False
    for img in page.get_images(full=True):
        xref = img[0]
        for bbox in page.get_image_rects(xref):
            if (bbox.width * bbox.height) / page_area >= _FULL_PAGE_IMAGE_AREA_RATIO:
                return True
    return False


def _render_page_to_png(page: "fitz.Page", dpi: int) -> bytes:
    # PNG-encoded here (not raw pixels) since this payload gets pickled and
    # piped to a worker process — a compressed scan is typically a fraction
    # of the size of its raw RGB buffer.
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def _prepare_pages(content: bytes, dpi: int, min_chars: int) -> list[_PreparedPage]:
    # MuPDF forbids touching pages of the same document from multiple threads
    # at once, so this whole pass (open -> read/render -> close) runs
    # single-threaded. Only the OCR inference step afterwards fans out.
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        prepared: list[_PreparedPage] = []
        for i in range(doc.page_count):
            page_start = time.perf_counter()
            page = doc[i]
            page_number = i + 1
            digital_text = _extract_digital_text(page)
            if len(digital_text) >= min_chars and not _has_full_page_image(page):
                prepared.append(_PreparedPage(page_number, "digital", digital_text, time.perf_counter() - page_start))
            else:
                png_bytes = _render_page_to_png(page, dpi)
                prepared.append(_PreparedPage(page_number, "ocr", png_bytes, time.perf_counter() - page_start))
        return prepared
    finally:
        doc.close()


# --------------------------------------------------------------------------
# PaddleOCR engine (runs inside pool worker processes)
# --------------------------------------------------------------------------

_reader_lock = threading.Lock()
_reader: "Optional[PaddleOCR]" = None

# Lightest available PP-OCR combo: tiny detection + English mobile
# recognition. Swap these for other model names (e.g. "PP-OCRv6_medium_det"
# / "PP-OCRv6_medium_rec") for higher accuracy at the cost of speed, or a
# language-specific *_rec model for non-English documents.
_DET_MODEL = "PP-OCRv6_tiny_det"
_REC_MODEL = "en_PP-OCRv4_mobile_rec"


def _get_reader() -> "PaddleOCR":
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=_DET_MODEL,
                text_recognition_model_name=_REC_MODEL,
            )
    return _reader


def _warmup_worker() -> None:
    """Runs inside a pool worker process to preload its model at startup."""
    _get_reader()


def _ocr_worker(png_bytes: bytes) -> tuple[str, float, float]:
    """Runs inside a pool worker process: decode + OCR one page image.

    Timed here, inside the worker, rather than by the caller around the
    executor submission — that would also count time the task spent queued
    waiting for a free worker, not just its own actual compute time.
    """
    start = time.perf_counter()
    reader = _get_reader()
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    array = np.array(image)

    lines: list[str] = []
    scores: list[float] = []
    for res in reader.predict(array):
        lines.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))

    elapsed = time.perf_counter() - start
    if not lines:
        return "", 0.0, elapsed
    return "\n".join(lines), round(sum(scores) / len(scores), 4), elapsed


# --------------------------------------------------------------------------
# Process pool management
# --------------------------------------------------------------------------

# Force "spawn" everywhere rather than relying on the OS default ("fork" on
# Linux) — see the module docstring for why.
_mp_context = multiprocessing.get_context("spawn")

_pool: "Optional[ProcessPoolExecutor]" = None
_pool_lock = asyncio.Lock()


def _new_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=PADDLE_OCR_WORKERS, mp_context=_mp_context)


async def _start_pool() -> None:
    global _pool
    _pool = _new_pool()
    loop = asyncio.get_running_loop()
    await asyncio.gather(*(loop.run_in_executor(_pool, _warmup_worker) for _ in range(PADDLE_OCR_WORKERS)))


async def _stop_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=True)
        _pool = None


async def _run_ocr(png_bytes: bytes) -> tuple[str, float, float]:
    global _pool
    async with _pool_lock:
        pool = _pool
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(pool, _ocr_worker, png_bytes)
    except BrokenProcessPool:
        # A worker crashed and poisoned the whole pool. Replace it so later
        # requests recover; this task's page/file is still reported as a
        # failure by the caller.
        async with _pool_lock:
            if _pool is pool:
                _pool = _new_pool()
        raise


# --------------------------------------------------------------------------
# PDF processing
# --------------------------------------------------------------------------


async def _resolve_page(item: _PreparedPage) -> PageResult:
    if item.kind == "digital":
        return PageResult(
            page_number=item.page_number,
            method="digital",
            text=item.payload,
            processing_time_seconds=round(item.prepare_seconds, 3),
        )

    text, confidence, ocr_seconds = await _run_ocr(item.payload)
    total_seconds = item.prepare_seconds + ocr_seconds
    return PageResult(
        page_number=item.page_number,
        method="ocr",
        text=text,
        confidence=confidence,
        processing_time_seconds=round(total_seconds, 3),
    )


async def _extract_text(filename: str, content: bytes) -> TextExtractionResult:
    start = time.perf_counter()

    prepared_pages = await asyncio.to_thread(_prepare_pages, content, OCR_DPI, DIGITAL_TEXT_MIN_CHARS)
    pages = await asyncio.gather(*(_resolve_page(item) for item in prepared_pages))

    ordered_pages = sorted(pages, key=lambda p: p.page_number)
    full_text = "\n\n".join(p.text for p in ordered_pages)
    elapsed = time.perf_counter() - start

    return TextExtractionResult(
        filename=filename,
        num_pages=len(ordered_pages),
        pages=ordered_pages,
        full_text=full_text,
        processing_time_seconds=round(elapsed, 3),
    )


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _start_pool()
    try:
        yield
    finally:
        await _stop_pool()


app = FastAPI(title="PaddleOCR Text Extraction API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results: list[TextExtractionResult] = []
    errors: list[ExtractionError] = []

    async def handle(file: UploadFile) -> None:
        content = await file.read()

        if not content.startswith(PDF_MAGIC):
            errors.append(ExtractionError(filename=file.filename, error="Not a valid PDF file"))
            return

        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            errors.append(ExtractionError(filename=file.filename, error=f"File exceeds {MAX_UPLOAD_MB}MB limit"))
            return

        try:
            results.append(await _extract_text(file.filename, content))
        except Exception as exc:  # malformed/corrupt PDFs, etc.
            errors.append(ExtractionError(filename=file.filename, error=str(exc)))

    await asyncio.gather(*(handle(f) for f in files))

    return TextExtractionResponse(results=results, errors=errors)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("paddleocr_extract_api:app", host="0.0.0.0", port=8002)
