# Real OCR photos

Drop your own photos here, one image plus a matching transcription:

    <name>.jpg   (or .png / .jpeg / .webp)
    <name>.txt   (the exact ground-truth transcription of that image)

`python -m eval.ocr_run` picks up every image that has a `.txt` next to it
and reports it under the `real` style. This folder is empty in the repo.
