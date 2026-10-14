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
   - Text, PDF (text layer via pypdf) or scanned PDF/images via a pluggable OCR
     layer (`app/ingestion/ocr.py`, `OCR_PROVIDER`): Gemini vision transcription by
     default (handles handwriting, no system binary), Tesseract as a local fallback,
     and a Bodhan OCR slot pending course access. OCR'd inputs are flagged in the UI.
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

For each question the grading call is sampled **N times** (`N` defaults to
`CONFIDENCE_SAMPLES`, 5 unless overridden) at `temperature > 0`. Three measures
are derived from the outputs only:

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

A labeled set of 26 items (some genuinely borderline/ambiguous) lives in
`eval/dataset.py`. Running:

```bash
python -m eval.run --samples 5
python -m eval.run --samples 5 --repeats 3
```

writes `eval/results/metrics.json` and `eval/results/calibration_curve.png`, reporting:

- **Grading accuracy** — MAE, Pearson r, agreement rate between predicted and
  ground-truth marks, before *and* after adjudication.
- **Confidence calibration** — AUROC (does low confidence predict grading errors?)
  against both pre- and post-adjudication errors, per-component ablation AUROCs
  (agree / variance / semantic / cross / combined), and a calibration curve of
  predicted-confidence bins vs. observed error rate, compared against the
  perfect-calibration line `1 - confidence`.
- **Flagging** — error rate among questions flagged at `CONFIDENCE_THRESHOLD` vs.
  unflagged questions.
- `--repeats N` runs the eval N times and reports mean ± std for every metric
  above (calibration curve plotted from the pooled runs).

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
| `LLM_DISABLE_THINKING` | `false` | disable reasoning/thinking tokens (e.g. DeepSeek V4.1 via Goedel Machines) |
| `LLM_NATIVE_N` | `false` | request `n` gradings in one call instead of `n` sequential calls |
| `GRADING_TEMPERATURE` | `0.7` | sampling temperature for confidence |
| `GRADE_WORKERS` | `4` | parallel threads grading questions |
| `ADJUDICATE` | `true` | referee the primary + second grading when they disagree materially |
| `CONFIDENCE_SAMPLES` | `5` | N samples per question |
| `CONFIDENCE_THRESHOLD` | `0.85` | flagging threshold |
| `W_AGREE` / `W_VARIANCE` / `W_SEMANTIC` / `W_CROSS` | `0.4` / `0.2` / `0` / `0.3` | combination weights (missing components are skipped and weights renormalised) |
| `CLUSTER_THRESHOLD` | `0.72` | semantic clustering cosine threshold |
| `ERROR_MARGIN` | `0.5` | marks difference counted as an error (eval) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `JUDGE2_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | second-grader endpoint (cross-model check) |
| `JUDGE2_API_KEY` | *(empty)* | enable the second grader; empty skips it |
| `JUDGE2_MODEL` | `gemini-3.5-flash-lite` | second-grader model |
| `OCR_PROVIDER` | `auto` | `auto` / `gemini` / `tesseract` / `bodhan` |
| `OCR_BASE_URL` | = `OPENAI_BASE_URL` | OCR/vision endpoint |
| `OCR_API_KEY` | = `OPENAI_API_KEY` | OCR/vision key — the text-only DeepSeek endpoint cannot OCR, so this must point at a vision model (e.g. Gemini) |
| `OCR_MODEL` | = `LLM_MODEL` | vision model used for OCR |

## Deploy on Vercel

The app is a single FastAPI function (`app/main.py`) that also serves the static
frontend. `vercel.json` uses the platform's default function timeout — the Hobby
default (300s) is used, which is enough for a full multi-question grading run.

1. Import the repo at [vercel.com/new](https://vercel.com/new) (framework preset: **Other**).
2. Add the environment variables below (Project → Settings → Environment Variables),
   then redeploy.

**Secrets** (add as *Sensitive* — never committed; `.vercelignore` excludes `.env`):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | primary grading LLM key |
| `OCR_API_KEY` | vision/OCR key (Gemini recommended; the text-only DeepSeek endpoint cannot OCR) |
| `JUDGE2_API_KEY` | second grader (optional; empty disables the cross-model check) |

**Non-secret configuration** (copy the values from `.env.example`):

`LLM_PROVIDER`, `OPENAI_BASE_URL`, `LLM_MODEL`, `LLM_DISABLE_THINKING`,
`LLM_NATIVE_N`, `LLM_REQUEST_PACING`, `GRADING_TEMPERATURE`, `LLM_MAX_RETRIES`,
`GRADE_WORKERS`, `ADJUDICATE`, `CONFIDENCE_SAMPLES`, `CONFIDENCE_THRESHOLD`,
`W_AGREE`, `W_VARIANCE`, `W_SEMANTIC`, `W_CROSS`, `CLUSTER_THRESHOLD`,
`ERROR_MARGIN`, `EMBEDDING_MODEL`, `OCR_PROVIDER`, `OCR_BASE_URL`, `OCR_MODEL`,
`JUDGE2_BASE_URL`, `JUDGE2_MODEL`.

Notes for serverless: `requirements.txt` intentionally has **no `torch` /
`sentence-transformers`** (heavy native wheels blow the bundle limit) — the embedding
layer falls back to a dependency-free character n-gram vector, and background warm-up
threads are disabled when `VERCEL` is set. Vercel also caps the request body at ~4.5 MB,
so the frontend downscales any uploaded image to a max of 2000 px on the long side
(re-encoded as JPEG q0.85) and refuses to send if the request still exceeds 4 MB; the
server enforces the same 4 MB per-file limit (and a 10-page cap for OCR'd PDFs).

## API

- `GET /api/health` — status, mock flag, provider + models, sampling config,
  second-grader on/off, adjudication on/off (no keys).
- `GET /api/sample` — the bundled `sample_data/` inputs for the "Try with sample data" button.
- `POST /api/grade` — multipart form with optional files `question_paper`,
  `answer_sheet`, `rubric`, `model_answers` (or matching `*_text` form fields),
  optional `structured_json`, and `n_samples` / `threshold`. Returns a
  `PipelineResult` JSON document. Files are limited to 4 MB each and OCR'd PDFs to
  10 pages; images are downscaled in the browser before upload. The primary LLM falls
  back to the second grader (with a warning) if it fails after retries.

## Agentic flow

Each question is graded by a small agentic loop that decides whether a human needs to
look at the result:

1. **Sample N** gradings from the primary model (temperature > 0).
2. **Second grader** — an independent model (e.g. Gemini) grades the same answer once
   (temperature 0) and its agreement with the representative total becomes the
   cross-model confidence component.
3. **Referee on disagreement** — when the two totals differ by `0.25 × max_marks`
   or more (and `ADJUDICATE=true`), the primary model is called once more
   (temperature 0) with both gradings and decides the final mark per criterion; that
   result replaces the representative. Confidence is *not* raised by adjudication.
4. **Flag** — questions whose combined confidence is below `CONFIDENCE_THRESHOLD` are
   flagged for human review.

## References

- Lin, Z. et al. (2023). **Generating with Confidence: Uncertainty Quantification
  for Black-Box Large Language Models.** (EigV / NumSet / Deg / Ecc methods.)
- Related black-box UQ literature on semantic consistency and NLI-based
  self-consistency for LLM outputs.
