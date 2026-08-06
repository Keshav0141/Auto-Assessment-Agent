# Confidence-Aware Auto-Assessment

[![Live Demo on Vercel](https://img.shields.io/badge/Live_Demo-Vercel-black?logo=vercel)](https://auto-assessment-agent-nine.vercel.app/)

*Try the live application here:* [auto-assessment-agent-nine.vercel.app](https://auto-assessment-agent-nine.vercel.app/)

A web app that grades a student's answer sheet against a rubric and model answers,
producing per-question marks + rubric-defensible feedback **and** a 0–1 confidence
score per grading decision. Low-confidence gradings are flagged for human review.

The confidence layer is the differentiating piece: it estimates uncertainty from
**black-box LLM outputs only** (no logprobs/logits), using sampling-based
consistency.

> **Note:** This is a course project for *Recent Advances in Generative AI* that applies black-box uncertainty quantification techniques to a real product (auto-grading). While it draws theoretical inspiration from my separate Master's Thesis Project (MTP) on LLM confidence bounds (Lin et al. 2023, *Generating with Confidence*), this repository is a separate applied project and not the theoretical MTP itself.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# set your provider (OpenAI-compatible, incl. OpenCode Zen / local servers)
copy .env.example .env                # then edit OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000, upload the sample files from `sample_data/`
(`question_paper.txt`, `answer_sheet_good.txt`, `rubric.txt`, `model_answers.txt`)
and hit **Grade answer sheet**.

> No API key set? The app falls back to a deterministic **mock grader** so you can
> exercise the whole UI/pipeline free of cost (it says so in the UI). Real grading
> needs `OPENAI_API_KEY`.

## Pipeline

1. **Ingestion & parsing** (`app/ingestion/parser.py`)
   - Text, PDF (text layer via pypdf) or scanned PDF/images (OCR via pytesseract +
     pypdfium2, installed from `requirements-extras.txt`).
   - Segments question paper / rubric / model answers / answer sheet by question
     anchors (`Question 1 (5 marks)` or a numbered list), extracting marks and
     rubric criteria. Accepts structured JSON too.
2. **Grading engine** (`app/grading/engine.py`)
   - Per question, prompts the LLM with question + rubric criteria + model answer +
     student answer, requesting a strict JSON schema. Malformed JSON is extracted
     defensively and calls retry with backoff. Outputs are clamped to valid bounds.
3. **Confidence scoring** (`app/confidence/scorer.py`) — see below.
4. **Output** — per-question marks, criterion-level breakdown, feedback, confidence,
   flagged status, totals; download as JSON, view in the web UI.

## How confidence is computed (black-box UQ)

For each question the grading call is sampled **N = 5 times** (configurable) at
`temperature > 0`. Three measures are derived from the outputs only:

- **Score variance** — the spread of `total_awarded` across the N samples,
  normalised by `max_marks`: `confidence_var = 1 - std(totals)/max_marks`.
- **Semantic consistency** — the N feedback texts are embedded (sentence-transformers
  if installed, else a hashed character n-gram fallback) and greedily clustered by
  cosine similarity. Two Lin et al. style measures are combined:
  - *NumSet-style*: `1 - (clusters - 1)/(N - 1)` — fewer distinct clusters ⇒ confident.
  - *Deg-style*: fraction of samples in the largest cluster.
- **EigV (optional, disabled by default)** — algebraic connectivity (2nd smallest eigenvalue of
  the normalised similarity-graph Laplacian): ≈1 for a tightly connected set of
  agreeing samples, ≈0 when samples form disconnected clusters. (Disabled by default due to graph-laplacian overhead).

These are combined into a single 0–1 score (weighted average, weights in `.env`),
and any question scoring **below `CONFIDENCE_THRESHOLD` (default 0.6)** is flagged
`needs human review`.

The scorer is a self-contained module — swap or extend it without touching grading
prompt logic.

## Evaluation harness (the core deliverable)

A labeled set of 16 items (some genuinely borderline/ambiguous) lives in
`eval/dataset.py`. Running:

```bash
python -m eval.run --samples 5
```

writes `eval/results/metrics.json` and `eval/results/calibration_curve.png`, reporting:

- **Grading accuracy** — MAE, Pearson r, agreement rate between predicted and
  ground-truth marks.
- **Confidence calibration** — AUROC (does low confidence predict grading errors?)
  and a calibration curve of predicted-confidence bins vs. observed error rate,
  compared against the perfect-calibration line `1 - confidence`.

This mirrors the MTP eval methodology and is the core deliverable to showcase for this course assignment demo.

## Project layout

```
app/
  main.py            FastAPI app + /api/grade, /api/health, static frontend
  config.py          env-driven settings
  models.py          pydantic schemas
  llm.py             OpenAI-compatible client, retry/backoff, JSON extraction
  embeddings.py      semantic embeddings (sentence-transformers, n-gram fallback)
  ingestion/parser.py  text/PDF/OCR extraction + question/rubric segmentation
  grading/engine.py  grading prompts, JSON validation, mock fallback
  confidence/scorer.py  sampling-based confidence (variance + semantic + EigV)
frontend/            vanilla HTML/JS/CSS single page served by FastAPI
eval/
  dataset.py         labeled items incl. borderline cases
  harness.py         metrics: MAE, Pearson, AUROC, calibration curve
  run.py             CLI entry point
sample_data/         demo question paper / rubric / model answers / answer sheets
```

## Configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `mock` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | *(empty)* | leave empty to use mock |
| `LLM_MODEL` | `gpt-4o-mini` | model name |
| `GRADING_TEMPERATURE` | `0.7` | sampling temperature for confidence |
| `CONFIDENCE_SAMPLES` | `5` | N samples per question |
| `CONFIDENCE_THRESHOLD` | `0.6` | flagging threshold |
| `W_VARIANCE` / `W_SEMANTIC` | `0.5` / `0.5` | combination weights |
| `CLUSTER_THRESHOLD` | `0.85` | semantic clustering cosine threshold |
| `ERROR_MARGIN` | `0.5` | marks difference counted as an error (eval) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |

## API

- `GET /api/health` — status, mock flag, model, sampling config.
- `POST /api/grade` — multipart form with optional files `question_paper`,
  `answer_sheet`, `rubric`, `model_answers` (or matching `*_text` form fields),
  optional `structured_json`, and `n_samples` / `threshold`. Returns a
  `PipelineResult` JSON document.

## References

- Lin, Z. et al. (2023). **Generating with Confidence: Uncertainty Quantification
  for Black-Box Large Language Models.** (EigV / NumSet / Deg / Ecc methods.)
- Related black-box UQ literature on semantic consistency and NLI-based
  self-consistency for LLM outputs.
