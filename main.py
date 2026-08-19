"""
PDF Extraction API — PaddleOCR
=================================
Upload single or multiple PDFs (digital or scanned). Returns page-wise
extracted text, confidence scores, and per-PDF processing time, as JSON.

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to test it interactively.
"""

import os
import tempfile
import time
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MIN_TEXT_CHARS_FOR_DIGITAL = 20   # below this char count, a page is treated as scanned
OCR_DPI = 200                     # render resolution for OCR — 200 is a good speed/accuracy balance

# Loaded once when the server starts, reused for every request.
# Doc-orientation/unwarp/textline-orientation are switched off for speed —
# turn them back on if your scans are rotated or skewed.
print("Loading PaddleOCR model... (first run downloads the model, may take a minute)")
ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)
print("PaddleOCR model loaded.")


# ----------------------------------------------------------------------
# Response schema
# ----------------------------------------------------------------------
class PageOut(BaseModel):
    page_number: int
    method: str                         # "digital" or "ocr"
    text: str
    confidence: Optional[float] = None  # 0-100 (100 for digital pages)


class PdfResult(BaseModel):
    filename: str
    status: str                         # "success" or "error"
    error: Optional[str] = None
    total_pages: int = 0
    time_seconds: float = 0.0
    pages: List[PageOut] = []


class ExtractResponse(BaseModel):
    total_files: int
    total_time_seconds: float
    results: List[PdfResult]


# ----------------------------------------------------------------------
# Extraction logic
# ----------------------------------------------------------------------
def extract_digital_text(page) -> str:
    """Fast path: read the PDF's real text layer directly."""
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (round(b[1], 1), b[0]))  # top-to-bottom, left-to-right
    return "\n".join(b[4].strip() for b in blocks if b[4].strip())


def ocr_page(page) -> tuple[str, float]:
    """Fallback path: rasterize the page and run PaddleOCR on it."""
    zoom = OCR_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    results = ocr_engine.predict(img_arr)
    if not results:
        return "", 0.0

    res = results[0]
    texts = res["rec_texts"]
    scores = res["rec_scores"]
    if not texts:
        return "", 0.0

    text = "\n".join(texts)
    avg_conf = round(sum(scores) / len(scores) * 100, 2)
    return text, avg_conf


def process_pdf(pdf_path: str) -> tuple[int, List[PageOut]]:
    doc = fitz.open(pdf_path)
    pages: List[PageOut] = []

    for i in range(len(doc)):
        page = doc[i]
        digital_text = extract_digital_text(page)

        if len(digital_text) >= MIN_TEXT_CHARS_FOR_DIGITAL:
            pages.append(PageOut(
                page_number=i + 1, method="digital", text=digital_text, confidence=100.0
            ))
        else:
            text, confidence = ocr_page(page)
            pages.append(PageOut(
                page_number=i + 1, method="ocr", text=text, confidence=confidence
            ))

    n_pages = len(doc)
    doc.close()
    return n_pages, pages


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API", version="1.0.0")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "PDF Extraction API is running. See /docs for usage."}


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files (form field name: 'files').
    Returns page-wise text, OCR confidence, and per-file timing.
    """
    overall_start = time.time()
    results: List[PdfResult] = []

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            results.append(PdfResult(filename=upload.filename, status="error", error="Not a PDF file"))
            continue

        content = await upload.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(content)
        tmp.close()

        start = time.time()
        try:
            n_pages, pages = process_pdf(tmp.name)
            results.append(PdfResult(
                filename=upload.filename,
                status="success",
                total_pages=n_pages,
                time_seconds=round(time.time() - start, 3),
                pages=pages,
            ))
        except Exception as e:
            results.append(PdfResult(filename=upload.filename, status="error", error=str(e)))
        finally:
            os.unlink(tmp.name)

    return ExtractResponse(
        total_files=len(files),
        total_time_seconds=round(time.time() - overall_start, 3),
        results=results,
    )
