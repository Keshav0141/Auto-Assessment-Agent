"""Generate the OCR benchmark images.

    python -m eval.make_ocr_images              # English + Hindi + Tamil
    python -m eval.make_ocr_images --skip-indic # English only (no browser needed)

For every item in ``eval/dataset.py`` the answer text is rendered in three capture
styles: clean typed, handwriting-like font, and a "phone photo" (handwriting plus
rotation, blur, noise, uneven lighting and JPEG q~60). The bundled sample answer
sheets are rendered as multi-question documents, and one scanned (image-only) PDF is
produced from the good sheet. Ground truth is always the original text.

The Hindi and Tamil answers from ``eval/multilingual.py`` are rendered in the same
three styles with Noto Sans Devanagari / Noto Sans Tamil. Pillow cannot shape Indic
scripts without libraqm (vowel signs and conjuncts come out wrong), so these images
are laid out by a headless Edge/Chromium through Playwright, which shapes text with
HarfBuzz, and then styled with Pillow. Generation stops with an error if the font
did not load. Needs ``pip install playwright`` and Microsoft Edge (or
``playwright install chromium``) plus network access for Google Fonts.

Outputs go to ``eval/ocr_data/`` together with ``manifest.json``. The folder
``eval/ocr_data/real/`` is created for your own photos (drop ``<name>.jpg`` with a
matching ``<name>.txt`` transcription).
"""
import argparse
import html
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

# Indic answers rendered in the browser: language -> (Noto family, a character to check).
INDIC_FONTS = {"hi": ("Noto Sans Devanagari", "क"), "ta": ("Noto Sans Tamil", "க")}
_GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500"
    "&family=Noto+Sans+Devanagari:wght@400;500&family=Noto+Sans+Tamil:wght@400;500&display=block"
)


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


def _indic_html(text, family, style):
    """Build the HTML page for one Indic answer.

    The handwriting-like style jitters, slants and resizes each word a little
    (deterministically, from the module's random seed).
    """
    handwriting = style in ("handwriting", "photo")
    words = []
    for word in text.split(" "):
        if handwriting:
            css = (
                f"display:inline-block;transform:translateY({random.uniform(-3, 3):.1f}px) "
                f"rotate({random.uniform(-2.5, 2.5):.1f}deg) skewX({random.uniform(-9, -3):.1f}deg);"
                f"font-size:{random.uniform(32, 37):.1f}px;font-weight:{random.choice((400, 500))}"
            )
            words.append(f'<span style="{css}">{html.escape(word)}</span>')
        else:
            words.append(html.escape(word))
    bg = "#faf9f4" if handwriting else "#ffffff"
    ink = "#121c5c" if handwriting else "#000000"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{_GOOGLE_FONTS}">
<style>
  html, body {{ margin: 0; background: {bg}; }}
  #page {{ width: 1100px; min-height: 420px; box-sizing: border-box; padding: 70px;
          font-family: "{family}", "Noto Sans", sans-serif; font-size: 30px; line-height: 1.75;
          color: {ink}; background: {bg}; word-spacing: 0.15em; }}
</style></head><body><div id="page">{" ".join(words)}</div></body></html>"""


class IndicRenderer:
    """Renders Indic text with correct shaping in a headless browser (Playwright)."""

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(channel="msedge")
        except Exception:  # noqa: BLE001 - fall back to Playwright's own Chromium
            self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(viewport={"width": 1100, "height": 600})
        return self

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()

    def render(self, text: str, lang: str, style: str) -> Image.Image:
        """Return a PIL image of `text` in `style`; raises if the Noto font did not load."""
        family, probe = INDIC_FONTS[lang]
        self._page.set_content(_indic_html(text, family, style), wait_until="networkidle")
        self._page.evaluate("document.fonts.ready")
        if not self._page.evaluate(f"document.fonts.check('30px \"{family}\"', '{probe}')"):
            raise RuntimeError(f"{family} did not load; check the network connection.")
        png = self._page.locator("#page").screenshot(type="png")
        img = Image.open(io.BytesIO(png)).convert("RGB")
        if style == "handwriting":
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
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


def main() -> None:
    """Build the OCR benchmark images and manifest in eval/ocr_data/."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-indic", action="store_true", help="skip the browser-rendered Hindi/Tamil images")
    args = parser.parse_args()
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

    def emit(doc_id, text, kind, item_id, lang="en", renderer=None):
        for style in STYLES:
            img = renderer.render(text, lang, style) if renderer else _render(text, style)
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
                    "lang": lang,
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
            "lang": "en",
            "image": pdf_name,
            "ground_truth": good,
        }
    )

    if not args.skip_indic:
        from eval.multilingual import ITEM_IDS, TRANSLATIONS

        with IndicRenderer() as renderer:
            for lang in INDIC_FONTS:
                for item_id in ITEM_IDS:
                    emit(f"{item_id}_{lang}", TRANSLATIONS[item_id][lang], "item", item_id, lang, renderer)

    manifest = {
        "generated_by": "eval/make_ocr_images.py",
        "styles": list(STYLES),
        "n_entries": len(entries),
        "entries": entries,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} benchmark images + manifest.json to {OCR_DATA_DIR}/")
    print(f"Real-photo folder ready at {REAL_DIR}/ (drop <name>.jpg + <name>.txt pairs).")


if __name__ == "__main__":
    main()
