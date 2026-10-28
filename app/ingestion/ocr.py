"""Pluggable OCR backends for scanned answer sheets / images.

OCR_PROVIDER (env):
  auto      -> vision if an OCR/LLM key is set, else tesseract   (default)
  vision    -> vision-LLM transcription via the OpenAI-compatible endpoint
               (OCR_MODEL; "gemini" is accepted as an alias)
  tesseract -> local pytesseract (needs the Tesseract binary)
  bodhan    -> Bodhan IndicOCR (stub: not implemented yet, see `_bodhan_page`)

Vision LLMs read handwriting far better than Tesseract and need no system
binary, so they also work on serverless/container deployments. Text may be English,
any Indic script or code-mixed; it is transcribed in the script it is written in.
"""
import base64
import io
import re
import time
from typing import Any, List, Sequence, Tuple

from app.config import settings
from app.llm import OutputBudgetError, _is_auth_error, check_output_budget

OCR_PROMPT = (
    "Transcribe ALL text in this image of an exam document exactly as written. "
    "The text may be in English, in any Indian language script (Devanagari, Tamil, "
    "Telugu, Bengali, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, ...) or "
    "code-mixed, for example Hindi words written in Latin letters (Hinglish) or "
    "English terms inside a Hindi sentence. "
    "Rules: write every word in the script it is written in; NEVER translate or "
    "transliterate anything; keep the original wording, spelling mistakes, digits "
    "(including Indic digits such as १ or ௧), equations and units; do NOT correct, "
    "solve or summarise anything; keep question labels such as 'Question 1', 'Q2', "
    "'3)', 'प्रश्न 1' or 'கேள்வி 1' on their own lines exactly as they appear; write "
    "math in plain text (e.g. x^2, sqrt(x), a/b); mark unreadable parts as "
    "[illegible]. If the image contains no text at all, output exactly [no text]. "
    "Output only the transcription, no commentary."
)

_MAX_SIDE = 2000
_MAX_PAGES = 10  # max PDF pages rendered for OCR (cost + latency guard)


class OCRError(RuntimeError):
    """OCR failed (backend unavailable, request error, too many pages)."""


class NoTextError(OCRError):
    """The image was read but contains no text (blank page, photo without writing)."""


_ALIASES = {"gemini": "vision"}


def _provider():
    p = _ALIASES.get(settings.ocr_provider, settings.ocr_provider)
    if p == "auto":
        return "vision" if settings.ocr_api_key else "tesseract"
    return p


def provider_name() -> str:
    """Return the OCR backend that will be used (vision, tesseract or bodhan)."""
    return _provider()


def backend_label(backend: str) -> str:
    """Return a human-readable OCR source.

    Args:
        backend: Backend name, as returned by `ocr_pages`.

    Returns:
        The vision model's name for the vision backend, else the backend's name.
    """
    if backend == "vision":
        return settings.ocr_model
    return {"tesseract": "Tesseract", "bodhan": "Bodhan"}.get(backend, str(backend))


def _to_jpeg_b64(img_or_bytes):
    """Downscale large images and encode as base64 JPEG (keeps requests small)."""
    from PIL import Image

    img = img_or_bytes
    if isinstance(img_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_or_bytes))
    img = img.convert("RGB")
    w, h = img.size
    scale = _MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _vision_page(img_or_bytes):
    from openai import OpenAI

    client = OpenAI(base_url=settings.ocr_base_url, api_key=settings.ocr_api_key)
    b64 = _to_jpeg_b64(img_or_bytes)
    last = None
    for attempt in range(settings.max_retries):
        try:
            kwargs = dict(
                model=settings.ocr_model,
                temperature=0.0,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
            )
            if settings.ocr_extra_body:
                kwargs["extra_body"] = dict(settings.ocr_extra_body)
            elif settings.ocr_disable_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
            resp = client.chat.completions.create(**kwargs)
            check_output_budget(resp.choices[0], settings.ocr_model)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last = "empty response"
        except OutputBudgetError as exc:
            raise OCRError(str(exc)) from exc  # retrying the same request cannot help
        except Exception as exc:  # rate limits / transient errors
            if _is_auth_error(exc):
                raise OCRError(f"invalid API key for {settings.ocr_base_url}") from exc
            last = exc
        time.sleep(min(5 * (2 ** attempt), 60))
    raise OCRError(f"Vision OCR ({settings.ocr_model}) failed: {last}")


def _tesseract_page(img_or_bytes):
    import pytesseract
    from PIL import Image

    img = img_or_bytes
    if isinstance(img_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_or_bytes))
    return pytesseract.image_to_string(img)


def _bodhan_page(img_or_bytes):
    """Bodhan IndicOCR backend (stub).

    Bodhan's IndicOCR reads printed and handwritten Indian-language scripts. It is
    configured with BODHAN_BASE_URL, BODHAN_OCR_KEY and BODHAN_OCR_MODEL, but the
    request itself is not implemented yet because the API reference was not
    available. The intended routing, once it is: with OCR_PROVIDER=auto, read with the
    vision model first and, when the result contains Indic script or several
    [illegible] parts and BODHAN_OCR_KEY is set, re-read with Bodhan (results cached
    by image hash, keeping the vision text plus a warning on any Bodhan error).
    """
    raise OCRError(
        "Bodhan IndicOCR is not implemented yet; set OCR_PROVIDER=vision (or auto) to "
        "use the vision model, which also reads Indic scripts."
    )


_BACKENDS = {"vision": _vision_page, "tesseract": _tesseract_page, "bodhan": _bodhan_page}


def ocr_pages(pages: Sequence[Any]) -> Tuple[str, str]:
    """OCR pages with the configured backend.

    Args:
        pages: PIL images or raw image bytes, one per page.

    Returns:
        (text, backend_used), with the pages joined by newlines.

    Raises:
        OCRError: If the backend is unknown or fails.
        NoTextError: If no page contains any text.
    """
    provider = _provider()
    fn = _BACKENDS.get(provider)
    if fn is None:
        raise OCRError(f"Unknown OCR_PROVIDER '{provider}'")
    out = []
    for i, p in enumerate(pages):
        out.append(fn(p))
        if provider == "vision" and settings.request_pacing and i < len(pages) - 1:
            time.sleep(settings.request_pacing)
    text = "\n".join(out)
    text = re.sub(r"(?im)^\s*\[no text\]\s*$", "", text)
    if not text.strip():
        raise NoTextError("OCR found no text")
    return text, provider


def render_pdf_pages(data: bytes, scale: float = 2.0) -> List[Any]:
    """Render each PDF page to a PIL image for OCR.

    Args:
        data: The PDF file contents.
        scale: Render scale (2.0 is about 144 dpi).

    Returns:
        One PIL image per page.

    Raises:
        OCRError: If the PDF has more than 10 pages.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        n_pages = len(pdf)
        if n_pages > _MAX_PAGES:
            raise OCRError(
                f"This PDF has {n_pages} pages; OCR supports at most {_MAX_PAGES}. "
                "Upload a shorter PDF, a text/text-layer PDF, or paste the text."
            )
        return [page.render(scale=scale).to_pil() for page in pdf]
    finally:
        pdf.close()
