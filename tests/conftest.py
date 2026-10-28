"""Shared test setup: mock mode only, no network, no local .env.

`app.config` loads `.env` with override=True at import time, so it is disabled here
before any `app` module is imported, and every provider variable is removed from the
environment. Tests that need other settings change the shared `settings` object with
`monkeypatch`, which restores it afterwards.
"""
import os
import sys

import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

_PREFIXES = (
    "OPENAI_", "LLM_", "JUDGE2_", "REFEREE_", "FALLBACK_", "OCR_", "W_", "CONFIDENCE_",
    "GRADING_", "ADJUDICATE", "CLUSTER_", "ERROR_MARGIN", "EMBEDDING_", "IDLE_", "GRADE_WORKERS",
    "VERCEL",
)
for _key in list(os.environ):
    if _key.startswith(_PREFIXES):
        del os.environ[_key]
os.environ["LLM_PROVIDER"] = "mock"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "sample_data")


def read_sample(name):
    with open(os.path.join(SAMPLE_DIR, name), encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sample_text():
    return read_sample


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retries and pacing never wait in tests."""
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
