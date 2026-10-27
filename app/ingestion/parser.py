"""Ingestion & parsing: extract text from PDFs/images/text and segment by question."""
import io
import json
import os
import re

from app.models import QuestionSpec, RubricCriterion, StudentAnswer


class ParseError(Exception):
    pass


class NoTextError(ParseError):
    """The file was read successfully but contains no text."""


# ---------------------------------------------------------------------------
# Text extraction (PDF / images / plain text)
# ---------------------------------------------------------------------------

TEXT_EXTS = (".txt", ".md", ".csv", ".json", ".tex", ".text")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")
_SUPPORTED = "text (.txt/.md), PDF, or an image (.png/.jpg/.jpeg/.webp/.tif/.bmp)"


def _sniff(data: bytes):
    """Detect the real format from magic bytes (filenames can lie or be missing)."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
            or data[:4] in (b"II*\x00", b"MM\x00*") or data[:2] == b"BM"):
        return "image"
    return None


def _check_image(filename, data: bytes):
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()
    except ImportError:
        return
    except Exception as exc:
        raise ParseError(f"Could not read the image '{filename}': the file is corrupt or not a supported image.") from exc


def _pdf_text(filename, data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise ParseError(f"'{filename}' is a password-protected PDF. Remove the password and upload it again.")
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            f"Could not read the PDF '{filename}': the file is corrupt or not a valid PDF."
        ) from exc


def _decode_text(filename, data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise ParseError(f"'{filename}' does not look like a text file. Upload {_SUPPORTED}, or paste the text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract_text_meta(filename, data: bytes):
    """Return (text, ocr_backend_or_None).

    Raises ParseError for unsupported / corrupt files and NoTextError (a subclass)
    when the file is readable but contains no text.
    """
    from app.ingestion.ocr import NoTextError as OCRNoText
    from app.ingestion.ocr import OCRError, ocr_pages, render_pdf_pages

    name = filename or "upload"
    ext = os.path.splitext(name.lower())[1]
    kind = _sniff(data)
    if kind is None:
        if ext == ".pdf":
            raise ParseError(f"Could not read the PDF '{name}': the file is corrupt or not a valid PDF.")
        if ext in IMAGE_EXTS:
            raise ParseError(f"Could not read the image '{name}': the file is corrupt or not a supported image.")
        if ext and ext not in TEXT_EXTS:
            raise ParseError(f"Unsupported file type '{ext}' for '{name}'. Upload {_SUPPORTED}, or paste the text.")
        kind = "text"

    try:
        if kind == "pdf":
            text = _pdf_text(name, data)
            if text.strip():
                return text, None
            try:
                pages = render_pdf_pages(data)
            except OCRError:
                raise
            except Exception as exc:
                raise ParseError(f"Could not render the PDF '{name}' for OCR: the file may be corrupt.") from exc
            return ocr_pages(pages)
        if kind == "image":
            _check_image(name, data)
            return ocr_pages([data])
    except OCRNoText as exc:
        raise NoTextError(f"No text could be found in '{name}' (blank page or unreadable image).") from exc
    except (OCRError, ImportError) as exc:
        raise ParseError(
            f"Could not OCR '{name}': {exc}. Set an LLM API key (Gemini OCR) or "
            "install Tesseract, or upload a text / text-layer PDF."
        ) from exc

    text = _decode_text(name, data)
    if not text.strip():
        raise NoTextError(f"'{name}' is empty.")
    return text, None


def extract_text(filename, data: bytes) -> str:
    return extract_text_meta(filename, data)[0]


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
    """Return list of (question_number, header, body) tuples, or None if no anchors."""
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


def _qkey(qid):
    return (0, int(qid)) if str(qid).isdigit() else (1, str(qid))


def _qlist(ids):
    return ", ".join(f"Q{q}" for q in sorted(set(ids), key=_qkey))


def _dedupe(segments, what, warnings=None, merge=False):
    """Resolve repeated question numbers. merge=True joins the repeated parts (a
    student continuing an answer later); otherwise the first occurrence wins."""
    seen, order, dups = {}, [], []
    for qid, header, body in segments:
        if qid in seen:
            dups.append(qid)
            if merge:
                q0, h0, b0 = seen[qid]
                seen[qid] = (q0, h0, (b0 + "\n\n" + body).strip())
            continue
        seen[qid] = (qid, header, body)
        order.append(qid)
    if dups and warnings is not None:
        action = "their parts were joined" if merge else "the first occurrence was used"
        warnings.append(f"The {what} repeats question number(s) {_qlist(dups)}; {action}.")
    return [seen[q] for q in order]


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
        try:
            criteria = [
                RubricCriterion(criterion=str(c.get("criterion", "")), marks=float(c.get("marks", 0)))
                for c in item.get("rubric_criteria", []) or []
            ]
            crit_total = sum(c.marks for c in criteria)
            has_marks = item.get("max_marks") not in (None, "")
            max_marks = float(item["max_marks"]) if has_marks else (crit_total or 1.0)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ParseError(f"Structured JSON question has invalid marks: {exc}") from exc
        questions.append(
            QuestionSpec(
                question_id=str(item.get("question_id", item.get("id", len(questions) + 1))),
                question_text=str(item["question_text"]),
                max_marks=max_marks if max_marks > 0 else 1.0,
                rubric_criteria=criteria,
                model_answer=str(item.get("model_answer", "") or ""),
                marks_specified=has_marks and max_marks > 0,
            )
        )
    return questions or None


# ---------------------------------------------------------------------------
# Domain parsers
# ---------------------------------------------------------------------------

def parse_question_paper(text, warnings=None):
    """Parse a question paper into a list of QuestionSpec."""
    structured = _parse_structured_json(text)
    if structured is not None:
        segments = [(q.question_id, "", q) for q in structured]
        return [q for _id, _h, q in _dedupe(segments, "question paper", warnings)]

    segments = _split_segments(text)
    if not segments:
        raise ParseError(
            "Could not identify questions in the question paper. Use headers like "
            "'Question 1 (5 marks)', a numbered list, or structured JSON."
        )
    questions = []
    for qid, header, body in _dedupe(segments, "question paper", warnings):
        marks = None
        m = _MARKS_RE.search(header)
        if not m:
            m = _MARKS_RE.search(body[:300])
        if m:
            marks = float(m.group(1))
        specified = marks is not None and marks > 0
        questions.append(
            QuestionSpec(
                question_id=qid,
                question_text=body,
                max_marks=marks if specified else 1.0,
                rubric_criteria=[],
                model_answer="",
                marks_specified=specified,
            )
        )
    return questions


def parse_rubric(text, warnings=None):
    """Parse rubric text into {question_id: [RubricCriterion]}. Best effort."""
    segments = _split_segments(text)
    if not segments:
        segments = [("1", "", text)]
    result = {}
    for qid, _header, body in _dedupe(segments, "rubric", warnings):
        criteria = []
        for m in _CRITERION_RE.finditer(body):
            criteria.append(RubricCriterion(criterion=m.group(2).strip(), marks=float(m.group(1))))
        if criteria:
            result[qid] = criteria
    return result


def parse_model_answers(text, warnings=None):
    """Parse model answers into {question_id: text}. Best effort."""
    segments = _split_segments(text)
    if not segments:
        return {"1": text}
    return {qid: body for qid, _h, body in _dedupe(segments, "model answers", warnings)}


def parse_answer_sheet(text, question_ids):
    """Segment the student answer sheet by question anchors.

    Returns (answers_or_None, warnings). Answers to questions that are not in the
    question paper are dropped with a warning; repeated question numbers are joined.
    """
    warnings = []
    segments = _split_segments(text)
    if not segments:
        warnings.append("No question anchors found in the answer sheet; entire text attached to the first question.")
        return None, warnings
    segments = _dedupe(segments, "answer sheet", warnings, merge=True)
    matched = {qid: body for qid, _h, body in segments if qid in question_ids}
    extra = [qid for qid, _h, _b in segments if qid not in question_ids]
    if not matched:
        warnings.append("Answer sheet anchors did not match question IDs; entire text attached to the first question.")
        return None, warnings
    if extra:
        warnings.append(
            f"The answer sheet has answers for {_qlist(extra)}, which are not in the "
            "question paper; they were not graded."
        )
    return (
        {qid: StudentAnswer(question_id=qid, student_answer_text=body) for qid, body in matched.items()},
        warnings,
    )


def default_criterion(max_marks):
    return RubricCriterion(criterion="Overall answer quality", marks=float(max_marks))


def apply_rubric(q: QuestionSpec, criteria):
    """Attach rubric criteria to `q`, reconciling them with the paper's marks.

    Rule: the question paper's marks are authoritative. If the rubric criteria add
    up to a different total, they are scaled proportionally to the paper's marks.
    If the paper gives no marks, the rubric total becomes the maximum; with neither,
    the question is worth 1 mark. Returns a list of warnings.
    """
    warnings = []
    if not criteria:
        if not q.marks_specified:
            warnings.append(
                f"Q{q.question_id}: the question paper gives no marks and there is no rubric, "
                f"so it was graded out of {q.max_marks:g} mark. Add marks such as "
                "'Question 1 (5 marks)'."
            )
        q.rubric_criteria = [default_criterion(q.max_marks)]
        return warnings
    total = sum(c.marks for c in criteria)
    if not q.marks_specified and total > 0:
        q.max_marks = total
        warnings.append(
            f"Q{q.question_id}: the question paper gives no marks; using the rubric total ({total:g})."
        )
    elif total > 0 and abs(total - q.max_marks) > 1e-6:
        scale = q.max_marks / total
        criteria = [
            RubricCriterion(criterion=c.criterion, marks=round(c.marks * scale, 2)) for c in criteria
        ]
        warnings.append(
            f"Q{q.question_id}: the rubric criteria add up to {total:g} marks but the question "
            f"paper gives {q.max_marks:g}; the criteria were scaled to {q.max_marks:g} "
            "(the question paper's marks take precedence)."
        )
    q.rubric_criteria = criteria
    return warnings
