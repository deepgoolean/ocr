
# PDF Extraction API

Upload one or more PDFs (digital or scanned) and get back page-wise text,
OCR confidence scores, and per-file timing — all as JSON.

- Digital PDFs: text is read directly (fast, exact, confidence = 100).
- Scanned PDFs: pages are rasterized and run through PaddleOCR, with a
  per-page confidence score.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

The first request after startup will be slower than usual — PaddleOCR
downloads and caches its model files on first use. After that it stays warm.

Open **http://127.0.0.1:8000/docs** for an interactive test UI (Swagger),
or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "files=@scan1.pdf" \
  -F "files=@digital2.pdf"
```

## Example response

```json
{
  "total_files": 2,
  "total_time_seconds": 4.1,
  "results": [
    {
      "filename": "scan1.pdf",
      "status": "success",
      "total_pages": 3,
      "time_seconds": 3.8,
      "pages": [
        { "page_number": 1, "method": "ocr", "text": "...", "confidence": 94.2 },
        { "page_number": 2, "method": "ocr", "text": "...", "confidence": 91.7 },
        { "page_number": 3, "method": "ocr", "text": "...", "confidence": 96.0 }
      ]
    },
    {
      "filename": "digital2.pdf",
      "status": "success",
      "total_pages": 1,
      "time_seconds": 0.05,
      "pages": [
        { "page_number": 1, "method": "digital", "text": "...", "confidence": 100.0 }
      ]
    }
  ]
}
```

## Notes

- Files are processed sequentially, one PDF then the next, so total time
  scales with how many files you send in one request. If you need to
  process large batches faster, that's the first thing to change (running
  files or pages concurrently) — flag it if that becomes a bottleneck.
- `MIN_TEXT_CHARS_FOR_DIGITAL` and `OCR_DPI` in `main.py` are the two knobs
  worth tuning first if accuracy or speed needs adjusting for your specific
  documents.






