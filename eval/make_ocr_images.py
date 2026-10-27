"""Generate the OCR benchmark images.

    python -m eval.make_ocr_images

For every item in ``eval/dataset.py`` the answer text is rendered in three capture
styles: clean typed, handwriting-like font, and a "phone photo" (handwriting plus
rotation, blur, noise, uneven lighting and JPEG q~60). The bundled sample answer
sheets are rendered as multi-question documents, and one scanned (image-only) PDF is
produced from the good sheet. Ground truth is always the original text.

Outputs go to ``eval/ocr_data/`` together with ``manifest.json``. The folder
``eval/ocr_data/real/`` is created for your own photos (drop ``<name>.jpg`` with a
matching ``<name>.txt`` transcription).
"""
import io
import json
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from eval.dataset import ITEMS
from eval.ocr_dataset import (
    IMAGE_EXTS,
    MANIFEST_PATH,
    OCR_DATA_DIR,
    REAL_DIR,
    SHEETS,
    STYLES,
)

CLEAN_FONTS = ("C:/Windows/Fonts/arial.ttf",)
HAND_FONTS = (
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/Inkfree.ttf",
    "C:/Windows/Fonts/LHANDW.TTF",
    "C:/Windows/Fonts/comic.ttf",
)

INK = (18, 28, 92)
PAPER = (250, 249, 244)


def _font(size, handwriting):
    for path in (HAND_FONTS if handwriting else CLEAN_FONTS):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    lines = []
    for para in (text.splitlines() or [""]):
        words = para.split(" ")
        cur = ""
        for word in words:
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def _render(text, style, width=1100):
    handwriting = style in ("handwriting", "photo")
    size = 34 if handwriting else 30
    font = _font(size, handwriting)
    line_h = int(size * 1.75)
    pad = 70
    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    lines = _wrap(probe, text, font, width - 2 * pad)
    height = max(420, 2 * pad + line_h * len(lines))
    img = Image.new("RGB", (width, height), PAPER if handwriting else "white")
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=INK if handwriting else "black", font=font)
        y += line_h
    if style == "photo":
        img = _degrade(img)
    return img


def _degrade(img):
    """Simulate a phone photo: small rotation, blur, noise, uneven light, JPEG q~60."""
    import numpy as np

    angle = random.uniform(-2.0, 2.0)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=PAPER)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    grad_y = np.linspace(0.78, 1.12, h)[:, None, None]
    grad_x = np.linspace(1.08, 0.82, w)[None, :, None]
    arr = arr * grad_y * grad_x
    arr = arr + np.random.normal(0, 6.0, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _photo_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    return buf.getvalue()


def _write_real_readme():
    path = os.path.join(REAL_DIR, "README.md")
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Real OCR photos\n\n"
            "Drop your own photos here, one image plus a matching transcription:\n\n"
            "    <name>.jpg   (or .png / .jpeg / .webp)\n"
            "    <name>.txt   (the exact ground-truth transcription of that image)\n\n"
            "`python -m eval.ocr_run` picks up every image that has a `.txt` next to it\n"
            "and reports it under the `real` style. This folder is empty in the repo.\n"
        )


def main():
    os.makedirs(OCR_DATA_DIR, exist_ok=True)
    os.makedirs(REAL_DIR, exist_ok=True)
    _write_real_readme()

    # Rebuild generated artefacts only; never touch real/.
    for name in os.listdir(OCR_DATA_DIR):
        path = os.path.join(OCR_DATA_DIR, name)
        if os.path.isfile(path) and (name.lower().endswith(IMAGE_EXTS) or name.lower().endswith(".pdf")):
            os.remove(path)

    random.seed(1234)
    entries = []

    def emit(doc_id, text, kind, item_id):
        for style in STYLES:
            img = _render(text, style)
            if style == "photo":
                fname = f"{doc_id}_photo.jpg"
                with open(os.path.join(OCR_DATA_DIR, fname), "wb") as f:
                    f.write(_photo_bytes(img))
            else:
                fname = f"{doc_id}_{style}.png"
                img.save(os.path.join(OCR_DATA_DIR, fname), "PNG")
            entries.append(
                {
                    "id": f"{doc_id}_{style}",
                    "doc_id": doc_id,
                    "item_id": item_id,
                    "style": style,
                    "kind": kind,
                    "image": fname,
                    "ground_truth": text,
                }
            )

    for item in ITEMS:
        emit(item["id"], item["answer"], "item", item["id"])

    for doc_id, fname in SHEETS:
        with open(os.path.join("sample_data", fname), encoding="utf-8") as f:
            emit(doc_id, f.read(), "sheet", None)

    # One scanned (image-only) PDF from the good sheet -> no text layer, forces OCR.
    with open(os.path.join("sample_data", "answer_sheet_good.txt"), encoding="utf-8") as f:
        good = f.read()
    pdf_name = "scanned_answersheet.pdf"
    _render(good, "clean", width=1240).save(
        os.path.join(OCR_DATA_DIR, pdf_name), "PDF", resolution=150.0
    )
    entries.append(
        {
            "id": "scanned_pdf",
            "doc_id": "scanned_pdf",
            "item_id": None,
            "style": "pdf",
            "kind": "pdf",
            "image": pdf_name,
            "ground_truth": good,
        }
    )

    manifest = {
        "generated_by": "eval/make_ocr_images.py",
        "styles": list(STYLES),
        "n_entries": len(entries),
        "entries": entries,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(entries)} benchmark images + manifest.json to {OCR_DATA_DIR}/")
    print(f"Real-photo folder ready at {REAL_DIR}/ (drop <name>.jpg + <name>.txt pairs).")


if __name__ == "__main__":
    main()
