"""Definitions + loaders for the OCR benchmark.

Ground truth for every generated image is the original text (``item["answer"]`` or the
bundled answer sheet). Images are produced by ``python -m eval.make_ocr_images``.

Real photos can be dropped into ``eval/ocr_data/real/``; each image must have a
side-by-side ``<name>.txt`` transcription and is picked up automatically.
"""
import json
import os

from eval.dataset import ITEMS

OCR_DATA_DIR = os.path.join("eval", "ocr_data")
REAL_DIR = os.path.join(OCR_DATA_DIR, "real")
MANIFEST_PATH = os.path.join(OCR_DATA_DIR, "manifest.json")

STYLES = ("clean", "handwriting", "photo")
STYLE_LABELS = {
    "clean": "clean typed",
    "handwriting": "handwriting",
    "photo": "phone photo",
    "pdf": "scanned PDF",
    "real": "real photo",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

SHEETS = (
    ("sheet_good", "answer_sheet_good.txt"),
    ("sheet_poor", "answer_sheet_poor.txt"),
)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def item_answers():
    """(item_id, ground_truth_text) for every labelled item."""
    return [(it["id"], it["answer"]) for it in ITEMS]


def sheet_answers():
    """(doc_id, ground_truth_text) for the bundled multi-question sheets."""
    return [(doc_id, _read(os.path.join("sample_data", fname))) for doc_id, fname in SHEETS]


def load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def real_entries():
    """Real photos in eval/ocr_data/real/ that have a matching .txt transcription."""
    entries = []
    if not os.path.isdir(REAL_DIR):
        return entries
    for name in sorted(os.listdir(REAL_DIR)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        txt = os.path.join(REAL_DIR, stem + ".txt")
        if not os.path.isfile(txt):
            continue
        entries.append(
            {
                "id": "real_" + stem,
                "doc_id": "real_" + stem,
                "item_id": None,
                "style": "real",
                "kind": "real",
                "image": os.path.join(REAL_DIR, name),
                "ground_truth": _read(txt),
            }
        )
    return entries


def all_entries(include_real=True):
    """All benchmark entries (generated manifest + optional real photos).

    Each entry: id, doc_id, item_id, style, kind, image (resolved path), ground_truth.
    """
    entries = []
    for e in load_manifest()["entries"]:
        e = dict(e)
        if not os.path.isabs(e["image"]):
            e["image"] = os.path.join(OCR_DATA_DIR, e["image"])
        entries.append(e)
    if include_real:
        entries += real_entries()
    return entries
