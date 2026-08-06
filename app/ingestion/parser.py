"""Ingestion & parsing: extract text from PDFs/images/text and segment by question."""
import io
import json
import re

from app.models import QuestionSpec, RubricCriterion, StudentAnswer


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Text extraction (PDF / images / plain text)
# ---------------------------------------------------------------------------

def extract_text(filename, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
        return _ocr_pdf(data)
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")):
        return _ocr_image(data)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")


def _ocr_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except Exception as exc:
        raise ParseError(
            f"OCR required for this image but is unavailable: {exc}. "
            "Install `pip install -r requirements-extras.txt` plus the Tesseract "
            "binary, or upload a text/PDF-with-text file."
        ) from exc


def _ocr_pdf(data: bytes) -> str:
    try:
        import pytesseract
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(data)
        pages = []
        for page in pdf:
            bmp = page.render(scale=2.0)
            img = bmp.to_pil()
            pages.append(pytesseract.image_to_string(img))
        pdf.close()
        if not any(p.strip() for p in pages):
            raise ValueError("OCR produced no text")
        return "\n".join(pages)
    except Exception as exc:
        raise ParseError(
            f"PDF appears scanned and OCR is unavailable: {exc}. "
            "Install `pip install -r requirements-extras.txt` plus the Tesseract "
            "binary, or upload a PDF that contains a text layer."
        ) from exc


# ---------------------------------------------------------------------------
# Question segmentation helpers
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"(?mi)^\s*(?:question|q\.?|problem)\s*(\d+)\b[^\n]*")
_NUMBERED_RE = re.compile(r"(?m)^\s*(\d+)\s*[.)]\s+")
_MARKS_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:marks?|points?|pts?)")
_CRITERION_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(?:mark|point|pt)s?\s*(?:for|if|-|:)\s*([^\n;,]+)"
)


def _split_segments(text):
    """Return list of (question_number, body) tuples, or None if no anchors."""
    markers = list(_HEADER_RE.finditer(text))
    if not markers:
        markers = list(_NUMBERED_RE.finditer(text))
    if not markers:
        return None
    segments = []
    for i, m in enumerate(markers):
        header = m.group(0)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segments.append((m.group(1), header, text[start:end].strip()))
    return segments


def _parse_structured_json(text):
    """Try to interpret text as a canonical JSON list of questions."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    items = data if isinstance(data, list) else [data]
    questions = []
    for item in items:
        if not isinstance(item, dict) or "question_text" not in item:
            return None
        criteria = [
            RubricCriterion(criterion=c.get("criterion", ""), marks=float(c.get("marks", 0)))
            for c in item.get("rubric_criteria", [])
        ]
        questions.append(
            QuestionSpec(
                question_id=str(item.get("question_id", item.get("id", len(questions) + 1))),
                question_text=item["question_text"],
                max_marks=float(item.get("max_marks", sum(c.marks for c in criteria) or 1)),
                rubric_criteria=criteria,
                model_answer=item.get("model_answer", ""),
            )
        )
    return questions or None


# ---------------------------------------------------------------------------
# Domain parsers
# ---------------------------------------------------------------------------

def parse_question_paper(text):
    """Parse a question paper into a list of QuestionSpec."""
    structured = _parse_structured_json(text)
    if structured is not None:
        return structured

    segments = _split_segments(text)
    if not segments:
        raise ParseError(
            "Could not identify questions in the question paper. Use headers like "
            "'Question 1 (5 marks)', a numbered list, or structured JSON."
        )
    questions = []
    for qid, header, body in segments:
        marks = None
        m = _MARKS_RE.search(header)
        if not m:
            m = _MARKS_RE.search(body[:300])
        if m:
            marks = float(m.group(1))
        if marks is None or marks <= 0:
            marks = 1.0
        questions.append(
            QuestionSpec(
                question_id=qid,
                question_text=body,
                max_marks=marks,
                rubric_criteria=[],
                model_answer="",
            )
        )
    return questions


def parse_rubric(text):
    """Parse rubric text into {question_id: [RubricCriterion]}. Best effort."""
    segments = _split_segments(text)
    if not segments:
        segments = [("1", "", text)]
    result = {}
    for qid, _header, body in segments:
        criteria = []
        for m in _CRITERION_RE.finditer(body):
            criteria.append(RubricCriterion(criterion=m.group(2).strip(), marks=float(m.group(1))))
        if criteria:
            result[qid] = criteria
    return result


def parse_model_answers(text):
    """Parse model answers into {question_id: text}. Best effort."""
    segments = _split_segments(text)
    if not segments:
        return {"1": text}
    return {qid: body for qid, _h, body in segments}


def parse_answer_sheet(text, question_ids):
    """Segment the student answer sheet by question anchors."""
    segments = _split_segments(text)
    if not segments:
        return None, "No question anchors found in the answer sheet; entire text attached to the first question."
    matched = {qid: body for qid, _h, body in segments if qid in question_ids}
    if not matched:
        return None, "Answer sheet anchors did not match question IDs; entire text attached to the first question."
    return (
        {qid: StudentAnswer(question_id=qid, student_answer_text=body) for qid, body in matched.items()},
        None,
    )


def default_criterion(max_marks):
    return RubricCriterion(criterion="Overall answer quality", marks=float(max_marks))
