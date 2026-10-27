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

# set your provider (any OpenAI-compatible endpoint, incl. local servers)
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
2. **Guards** (`app/grading/guards.py`) — cheap pre-checks on every answer before any
   LLM call (blank, prompt injection, restated question, gibberish, OCR `[illegible]`
   markers, length) plus informational answer-language detection. See [Robustness](#robustness).
3. **Grading engine** (`app/grading/engine.py`)
   - Per question, prompts the LLM with question + rubric criteria + model answer +
     student answer, requesting a strict JSON schema. Malformed JSON is extracted
     defensively and calls retry with backoff. Outputs are clamped to valid bounds.
4. **Confidence scoring** (`app/confidence/scorer.py`) — see below.
5. **Output** — per-question marks, criterion-level breakdown, feedback, confidence,
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
and any question scoring **below `CONFIDENCE_THRESHOLD` (default 0.85)** is flagged
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
- **Adjudication** — how many items were refereed, how often the referee actually
  changed the mark (`n_referee_changed`), and which model refereed.
- `--repeats N` runs the eval N times and reports mean ± std for every metric
  above (calibration curve plotted from the pooled runs).

This mirrors the MTP eval methodology and is the core deliverable to showcase for this course assignment demo.

## OCR evaluation

Scanned answer sheets pass through `app/ingestion/ocr.py` before grading, so OCR errors
propagate into marks. This benchmark measures that effect end to end.

Build the images (26 eval items x 3 capture styles, the two sample sheets, and one
scanned image-only PDF; ground truth is always the original text):

```bash
python -m eval.make_ocr_images
```

This writes `eval/ocr_data/` (excluded via `.vercelignore`) with a `manifest.json`. The
three styles are `clean` typed, `handwriting` (script font), and `phone photo`
(handwriting plus ±2° rotation, blur, noise, uneven lighting and JPEG q~60). Drop your
own photos into `eval/ocr_data/real/` as `<name>.jpg` + `<name>.txt` (matching
transcription) and they join the run automatically.

Run a provider (the endpoint comes from env vars; `--provider` is a report label):

```bash
# current Goedel setup (values from .env)
python -m eval.ocr_run --provider goedel --limit 8

# OpenRouter: export its endpoint/key/model first (these win over .env for the run)
export OCR_BASE_URL=https://openrouter.ai/api/v1 OCR_API_KEY=... OCR_MODEL=...
python -m eval.ocr_run --provider openrouter --limit 8

# local Tesseract, if installed
python -m eval.ocr_run --provider tesseract
```

`OCR_BASE_URL` / `OCR_API_KEY` / `OCR_MODEL` (plus `OCR_EXTRA_BODY` and
`OCR_DISABLE_THINKING`) exported in the shell override `.env` for that run.
`OCR_EXTRA_BODY` is a JSON object merged into the provider request body and takes
precedence over the `OCR_DISABLE_THINKING` shortcut, so OpenRouter / Goedel / Gemini can
each send their own flags (e.g. `OCR_EXTRA_BODY='{"reasoning":{"enable":false}}'`).

Reported per style and overall (mean ± std when `--repeats` > 1):

- **CER / WER** — Levenshtein character / word error rate against the ground truth.
- **Question-label recall** — fraction of `Question N` anchors that survive OCR
  (computed on the multi-question sheets / PDF; `-` for single-answer items).
- **Latency** and **failure** counts per image.
- **End-to-end impact** — each item's ground-truth text and OCR text are graded by the
  same grader (`--samples`, default 2); reports mean |Δmarks|, the share of items whose
  mark moves by `>= ERROR_MARGIN`, and mean combined confidence for GT vs OCR text.
  `--no-grade` skips this for a cheaper OCR-only run.

Results go to `eval/results/ocr_metrics.json`, and a comparison table is printed.

## Robustness

### Prompt-injection defence

- **Per-request delimiters.** Every grading, second-grader and referee request wraps
  the student answer in fresh random markers (`<<<ANSWER_3f9a1c>>> … <<<END_3f9a1c>>>`).
  The system prompt says everything between them is untrusted student text to grade,
  never instructions. Marker-like runs (`<<<`, `>>>`) inside the answer are broken
  up, so a student cannot fake a closing marker, and the random suffix cannot be guessed.
- **Regex guards** (`app/grading/guards.py`) catch instruction-like phrases in English
  and Hindi ("ignore previous instructions", "award full marks", "give me 5/5",
  "you are now", `system:` / `SYSTEM:` role labels, "निर्देशों को अनदेखा करें",
  "पूरे अंक दें", …). They also catch text that imitates delimiters or chat roles:
  lines starting with `<<<`, `>>>` or `END_`, `</answer>`-style tags, ```` ```system ````
  fences, `[INST]` / `<<SYS>>`, `### Instruction` headings, and `assistant:` / `user:` /
  `Grader note:` lines. The sentence containing a hit is replaced with
  `[instruction-like text removed]` before grading, so it cannot move the mark.
- **Integrity flags.** The grading JSON schema has an `integrity_flags` list,
  whitelisted to `prompt_injection`, `blank`, `off_topic`, `restates_question`,
  `gibberish`, `language_mismatch`, `illegible`; anything else is dropped. Guard flags
  always count. A model-reported flag counts when at least half of the gradings
  that ran (primary samples + second grader + referee) report it, so a single noisy
  sample at temperature > 0 cannot force a review.
- **Language is not an integrity problem.** Answers in Hindi, Tamil, Hinglish or any
  other language are graded normally. The detected language is returned as
  `confidence.answer_language` (from the script; Devanagari is reported as Hindi,
  romanised Hindi as Hinglish, two scripts as e.g. `Hindi + English`) and is
  informational only. `language_mismatch` is raised only when the rubric explicitly
  requires a language (e.g. "1 mark for answering in English") and the answer is in
  another one; graders cannot raise it.
- **Effect of a flag.** Any flag sets `flagged = true` and caps combined confidence at
  0.5. The reasons are returned in `confidence.integrity_reasons` and shown on the
  question card in the UI. Flags never raise marks. The only change to what gets
  graded is the redaction above.

### Edge cases

Nothing in this list crashes the request. Each case returns a clear warning, or a 400
with a specific message when there is nothing to grade.

| Input | Behaviour |
|---|---|
| Blank answer (empty, whitespace, `N/A`, "not attempted", only `[illegible]`) | 0 marks with no LLM call, flagged `blank` |
| Answer longer than 6000 characters | the first 6000 characters are graded, with a warning |
| Answer for a question that is not in the paper | not graded, with a warning |
| Question in the paper with no answer | graded as blank, with a warning |
| Rubric / model answers for questions not in the paper | ignored, with a warning |
| Rubric marks don't add up to the paper's marks | **the question paper wins**: criteria are scaled proportionally to the paper's marks, with a warning |
| Question paper gives no marks | the rubric total becomes the maximum; with no rubric either, the question is worth 1 mark. Both cases warn |
| Duplicate question numbers | paper / rubric / model answers keep the first occurrence; in the answer sheet the parts are joined (the student continued the answer); always warns |
| Unsupported file type (`.docx`, `.xlsx`, binary renamed `.txt`, …) | 400 for the question paper or answer sheet; a warning (and the file is ignored) for the rubric or model answers |
| Corrupt or password-protected PDF, corrupt image | same as unsupported |
| Image or scanned PDF with no text | answer sheet is treated as empty (every question is blank and flagged); question paper is a 400 |

File types are detected from magic bytes as well as the extension, so a PDF without
a `.pdf` name is still read as a PDF.

### Referee and fallback providers

- **Referee.** `REFEREE_BASE_URL` / `REFEREE_API_KEY` / `REFEREE_MODEL` /
  `REFEREE_EXTRA_BODY` configure the adjudication call. Each defaults to the primary
  LLM. The model that refereed is returned as `confidence.referee_model`.
- **Fallback chain.** If the primary fails after retries for a reason other than
  auth: primary → `FALLBACK_*` provider (per-sample loop, never native `n`) → second
  grader. Auth errors are never masked. The grader that produced the marks is
  returned as `confidence.graded_by` / `confidence.grader_role` (`primary`,
  `fallback`, `judge2`, `mock`, or `none` for blank answers).
- **Request flags stay with their provider.** The primary's Goedel/DeepSeek
  `chat_template_kwargs` (from `LLM_DISABLE_THINKING`) are only sent to the primary,
  or to a referee that uses the same endpoint and model. The referee and fallback send
  only their own `*_EXTRA_BODY`, and the second grader sends none.
- **Reasoning budget.** If any call (primary, fallback, second grader, referee, OCR) returns
  an empty reply with `finish_reason="length"`, a reasoning model spent the whole output
  budget thinking. That is raised at once as "reasoning used the whole output budget for
  `<model>`" with no retries; for grading it then moves down the fallback chain. Fix it with
  that provider's `*_EXTRA_BODY`, e.g. `REFEREE_EXTRA_BODY='{"reasoning":{"enabled":false}}'`
  on OpenRouter.
- `/api/health` reports `referee_model` and `fallback_model` (never keys).

### Adversarial eval

```bash
python -m eval.adversarial              # --samples N to override CONFIDENCE_SAMPLES
```

The script runs 3 baselines and 17 attack/edge cases through the full pipeline.
Examples: an injection appended to a wrong answer (English, Hindi, `SYSTEM:`,
"you are now", fake delimiters, chat-template markup, an authority claim with no regex
hit), which must score ≤ the clean wrong answer and be flagged; blank / `N/A` → 0 with no LLM call; a
restated question → ≤ 20% and flagged; gibberish → 0 and flagged; off-topic → ≤ 20%
and flagged; a partial answer padded past 8000 characters → no more than the
unpadded answer; `500 cm` for `5 m` → full marks; a correct answer in Hindi → 5/5,
not flagged, `answer_language` Hindi; the same Hindi answer against a rubric that
requires English → flagged `language_mismatch`; OCR `[illegible]` gaps → flagged. It prints a pass/fail table and
writes `eval/results/adversarial.json`.

## Project layout

```
app/
  main.py            FastAPI app + /api/grade, /api/health, static frontend
  config.py          env-driven settings
  models.py          pydantic schemas
  llm.py             OpenAI-compatible client, retry/backoff, JSON extraction
  embeddings.py      semantic embeddings (sentence-transformers, n-gram fallback)
  ingestion/parser.py  text/PDF/OCR extraction + question/rubric segmentation
  grading/engine.py  grading prompts, JSON validation, fallback chain, mock grader
  grading/guards.py  cheap pre-checks + integrity flags (injection, blank, gibberish, ...)
  confidence/scorer.py  sampling-based confidence (variance + semantic + EigV)
frontend/            vanilla HTML/JS/CSS single page served by FastAPI
eval/
  dataset.py         labeled items incl. borderline cases
  harness.py         metrics: MAE, Pearson, AUROC, calibration curve
  run.py             CLI entry point
  adversarial.py     prompt-injection / edge-case eval (pass/fail table)
  ocr_dataset.py     OCR benchmark definitions + manifest / real-photo loaders
  make_ocr_images.py generates clean / handwriting / phone-photo benchmark images
  ocr_run.py         OCR benchmark CLI (CER/WER/label recall/latency + grading impact)
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
| `REFEREE_BASE_URL` / `REFEREE_API_KEY` / `REFEREE_MODEL` | = primary LLM | adjudication (referee) provider |
| `REFEREE_EXTRA_BODY` | *(empty)* | JSON merged into referee requests (the primary's flags are reused only when the referee is the primary's endpoint + model) |
| `FALLBACK_MODEL` | *(empty)* | enables the fallback grader used when the primary fails after retries |
| `FALLBACK_BASE_URL` / `FALLBACK_API_KEY` | = primary LLM | fallback endpoint / key |
| `FALLBACK_EXTRA_BODY` | *(empty)* | JSON merged into fallback requests |
| `OCR_PROVIDER` | `auto` | `auto` / `gemini` / `tesseract` / `bodhan` |
| `OCR_BASE_URL` | = `OPENAI_BASE_URL` | OCR/vision endpoint |
| `OCR_API_KEY` | = `OPENAI_API_KEY` | OCR/vision key — the text-only DeepSeek endpoint cannot OCR, so this must point at a vision model (e.g. Gemini) |
| `OCR_MODEL` | = `LLM_MODEL` | vision model used for OCR |
| `OCR_EXTRA_BODY` | *(empty)* | JSON merged into OCR requests (overrides `OCR_DISABLE_THINKING`) |

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
| `REFEREE_API_KEY` | referee (optional; defaults to `OPENAI_API_KEY`) |
| `FALLBACK_API_KEY` | fallback grader (optional; defaults to `OPENAI_API_KEY`) |

**Non-secret configuration** (copy the values from `.env.example`):

`LLM_PROVIDER`, `OPENAI_BASE_URL`, `LLM_MODEL`, `LLM_DISABLE_THINKING`,
`LLM_NATIVE_N`, `LLM_REQUEST_PACING`, `GRADING_TEMPERATURE`, `LLM_MAX_RETRIES`,
`GRADE_WORKERS`, `ADJUDICATE`, `CONFIDENCE_SAMPLES`, `CONFIDENCE_THRESHOLD`,
`W_AGREE`, `W_VARIANCE`, `W_SEMANTIC`, `W_CROSS`, `CLUSTER_THRESHOLD`,
`ERROR_MARGIN`, `EMBEDDING_MODEL`, `OCR_PROVIDER`, `OCR_BASE_URL`, `OCR_MODEL`,
`JUDGE2_BASE_URL`, `JUDGE2_MODEL`, `REFEREE_BASE_URL`, `REFEREE_MODEL`,
`REFEREE_EXTRA_BODY`, `FALLBACK_BASE_URL`, `FALLBACK_MODEL`, `FALLBACK_EXTRA_BODY`.

Notes for serverless: `requirements.txt` intentionally has **no `torch` /
`sentence-transformers`** (heavy native wheels blow the bundle limit) — the embedding
layer falls back to a dependency-free character n-gram vector, and background warm-up
threads are disabled when `VERCEL` is set. Vercel also caps the request body at ~4.5 MB,
so the frontend downscales any uploaded image to a max of 2000 px on the long side
(re-encoded as JPEG q0.85) and refuses to send if the request still exceeds 4 MB; the
server enforces the same 4 MB per-file limit (and a 10-page cap for OCR'd PDFs).

## API

- `GET /api/health` — status, mock flag, provider + models, sampling config,
  second-grader on/off, adjudication on/off, referee and fallback model names (no keys).
- `GET /api/sample` — the bundled `sample_data/` inputs for the "Try with sample data" button.
- `POST /api/grade` — multipart form with optional files `question_paper`,
  `answer_sheet`, `rubric`, `model_answers` (or matching `*_text` form fields),
  optional `structured_json`, and `n_samples` / `threshold`. Returns a
  `PipelineResult` JSON document. Files are limited to 4 MB each and OCR'd PDFs to
  10 pages; images are downscaled in the browser before upload. If the primary LLM fails
  after retries it falls back to the `FALLBACK_*` provider, then to the second grader
  (with a warning). Each question reports `integrity_flags` / `integrity_reasons`,
  `answer_language`, `graded_by`, `grader_role` and `referee_model` in its
  `confidence` object.

## Agentic flow

Each question is graded by a small agentic loop that decides whether a human needs to
look at the result:

0. **Guards** — blank answers get 0 with no LLM call; instruction-like text is
   removed; over-long answers are truncated (see [Robustness](#robustness)).
1. **Sample N** gradings from the primary model (temperature > 0). If it fails after
   retries, the fallback provider and then the second grader take over.
2. **Second grader** — an independent model (e.g. Gemini) grades the same answer once
   (temperature 0) and its agreement with the representative total becomes the
   cross-model confidence component.
3. **Referee on disagreement** — when the two totals differ by `0.25 × max_marks`
   or more (and `ADJUDICATE=true`), the referee (`REFEREE_*`, default: the primary
   model) is called once (temperature 0) with both gradings, shown anonymously as
   Grader A / Grader B in random order. It judges each rubric criterion independently
   against the rubric and model answer (partial credit allowed) before totalling; that
   result replaces the representative. Confidence is *not* raised
   by adjudication.
4. **Flag** — questions whose combined confidence is below `CONFIDENCE_THRESHOLD`, or
   that carry any integrity flag, are flagged for human review.

## References

- Lin, Z. et al. (2023). **Generating with Confidence: Uncertainty Quantification
  for Black-Box Large Language Models.** (EigV / NumSet / Deg / Ecc methods.)
- Related black-box UQ literature on semantic consistency and NLI-based
  self-consistency for LLM outputs.
