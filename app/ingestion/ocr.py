"""Pluggable OCR backends for scanned answer sheets / images.

OCR_PROVIDER (env):
  auto      -> vision if an OCR/LLM key is set, else tesseract   (default)
  vision    -> vision-LLM transcription via the OpenAI-compatible endpoint
               (OCR_MODEL; "gemini" is accepted as an alias)
  tesseract -> local pytesseract (needs the Tesseract binary)
  bodhan    -> Bodhan OCR API (placeholder until course access is enabled)

Vision LLMs read handwriting far better than Tesseract and need no system
binary, so they also work on serverless/container deployments.
"""
import base64
import io
import re
import time

from app.config import settings
from app.llm import OutputBudgetError, _is_auth_error, check_output_budget

OCR_PROMPT = (
    "Transcribe ALL text in this image of an exam document exactly as written. "
    "Rules: keep the original wording, spelling mistakes, equations and units; "
    "do NOT correct, solve or summarise anything; keep question labels such as "
    "'Question 1', 'Q2', '3)' on their own lines exactly as they appear; write "
    "math in plain text (e.g. x^2, sqrt(x), a/b); mark unreadable parts as "
    "[illegible]. If the image contains no text at all, output exactly [no text]. "
    "Output only the transcription, no commentary."
)

_MAX_SIDE = 2000
_MAX_PAGES = 10  # max PDF pages rendered for OCR (cost + latency guard)


class OCRError(RuntimeError):
    pass


class NoTextError(OCRError):
    """The image was read but contains no text (blank page, photo without writing)."""


_ALIASES = {"gemini": "vision"}


def _provider():
    p = _ALIASES.get(settings.ocr_provider, settings.ocr_provider)
    if p == "auto":
        return "vision" if settings.ocr_api_key else "tesseract"
    return p


def provider_name() -> str:
    """The OCR backend that will be used (vision / tesseract / bodhan)."""
    return _provider()


def backend_label(backend) -> str:
    """Human-readable OCR source: the vision model's name, or the backend name."""
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
    # TODO: wire up once Bodhan OCR credits/docs are available (console.bodhan.ai).
    raise OCRError("Bodhan OCR is not configured yet - set OCR_PROVIDER=vision for now.")


_BACKENDS = {"vision": _vision_page, "tesseract": _tesseract_page, "bodhan": _bodhan_page}


def ocr_pages(pages):
    """OCR a list of PIL images or raw image bytes. Returns (text, backend_used)."""
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


def render_pdf_pages(data: bytes, scale=2.0):
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
