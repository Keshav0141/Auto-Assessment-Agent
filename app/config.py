"""Application configuration, loaded from environment variables.

A `.env` file in the working directory is loaded first and overrides the process
environment. Every setting has a default, so the app starts with no configuration
at all (and then grades in mock mode).
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_json(name: str, default: Optional[dict]) -> Optional[dict]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible endpoint and model.

    Attributes:
        role: primary, referee, fallback or judge2.
        base_url: Endpoint URL.
        api_key: API key for the endpoint.
        model: Model name.
        extra_body: Provider-specific JSON merged into every request; never copied
            across providers.
    """
    role: str
    base_url: str
    api_key: str
    model: str
    extra_body: Optional[dict] = None


class Settings:
    """All runtime settings, read once from the environment at import time.

    The module-level `settings` instance is shared by the whole app; tests and the
    eval scripts change its attributes directly.
    """
    def __init__(self):
        # True on Vercel/serverless: background threads and idle warm-ups are useless
        # (each invocation is isolated), so they are disabled.
        self.serverless = bool(os.getenv("VERCEL"))

        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_disable_thinking = os.getenv("LLM_DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
        self.llm_native_n = os.getenv("LLM_NATIVE_N", "false").lower() in ("1", "true", "yes")

        # Optional second grader (cross-model check). Only used when JUDGE2_API_KEY is set.
        self.judge2_base_url = os.getenv("JUDGE2_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
        self.judge2_api_key = os.getenv("JUDGE2_API_KEY", "").strip()
        self.judge2_model = os.getenv("JUDGE2_MODEL", "gemini-3.5-flash-lite")

        # Optional referee for adjudication; every field defaults to the primary LLM.
        self.referee_base_url = os.getenv("REFEREE_BASE_URL", "").strip().rstrip("/")
        self.referee_api_key = os.getenv("REFEREE_API_KEY", "").strip()
        self.referee_model = os.getenv("REFEREE_MODEL", "").strip()
        self.referee_extra_body = _env_json("REFEREE_EXTRA_BODY", None)

        # Optional fallback grader used when the primary fails (non-auth, after
        # retries). Enabled by FALLBACK_MODEL; endpoint/key default to the primary's.
        self.fallback_base_url = os.getenv("FALLBACK_BASE_URL", "").strip().rstrip("/")
        self.fallback_api_key = os.getenv("FALLBACK_API_KEY", "").strip()
        self.fallback_model = os.getenv("FALLBACK_MODEL", "").strip()
        self.fallback_extra_body = _env_json("FALLBACK_EXTRA_BODY", None)

        self.grading_temperature = _env_float("GRADING_TEMPERATURE", 0.7)
        self.max_retries = _env_int("LLM_MAX_RETRIES", 3)
        self.request_pacing = _env_float("LLM_REQUEST_PACING", 0.0)
        self.idle_warmup_seconds = _env_float("IDLE_WARMUP_SECONDS", 90.0)

        self.n_samples = _env_int("CONFIDENCE_SAMPLES", 5)
        self.confidence_threshold = _env_float("CONFIDENCE_THRESHOLD", 0.85)
        self.w_agree = _env_float("W_AGREE", 0.4)
        self.w_variance = _env_float("W_VARIANCE", 0.2)
        self.w_semantic = _env_float("W_SEMANTIC", 0.0)
        self.w_cross = _env_float("W_CROSS", 0.3)
        self.adjudicate = os.getenv("ADJUDICATE", "true").lower() in ("1", "true", "yes")
        self.cluster_threshold = _env_float("CLUSTER_THRESHOLD", 0.72)
        self.error_margin = _env_float("ERROR_MARGIN", 0.5)

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # OCR: auto | vision (alias: gemini) | tesseract | bodhan
        self.ocr_provider = os.getenv("OCR_PROVIDER", "auto").strip().lower()
        self.ocr_model = os.getenv("OCR_MODEL", "").strip() or self.llm_model
        self.ocr_base_url = (os.getenv("OCR_BASE_URL", "").strip() or self.openai_base_url).rstrip("/")
        self.ocr_api_key = os.getenv("OCR_API_KEY", "").strip() or self.openai_api_key
        self.ocr_disable_thinking = os.getenv("OCR_DISABLE_THINKING", "false").lower() in ("1", "true", "yes")
        # Optional JSON merged into the OCR request body (provider-specific flags, e.g.
        # {"reasoning": {"effort": "none"}} for OpenRouter). Takes precedence over the
        # OCR_DISABLE_THINKING shortcut when set.
        self.ocr_extra_body = _env_json("OCR_EXTRA_BODY", {})

        # Bodhan IndicOCR (optional backend; currently a stub, see app/ingestion/ocr.py).
        self.bodhan_base_url = os.getenv("BODHAN_BASE_URL", "").strip().rstrip("/")
        self.bodhan_ocr_key = os.getenv("BODHAN_OCR_KEY", "").strip()
        self.bodhan_ocr_model = os.getenv("BODHAN_OCR_MODEL", "").strip()

        # Language of feedback and justifications: auto (the student's detected answer
        # language), en, or a language name/code such as Hindi or ta.
        self.feedback_language = os.getenv("FEEDBACK_LANGUAGE", "auto").strip() or "auto"

    @property
    def using_mock(self) -> bool:
        """True when grading uses the offline mock grader (no API key, or LLM_PROVIDER=mock)."""
        return self.llm_provider == "mock" or not self.openai_api_key

    @property
    def uses_embeddings(self) -> bool:
        """True when the app needs feedback embeddings.

        Only the semantic-consistency component uses them in the app (EigV is only
        computed when a caller asks for it), so with W_SEMANTIC=0 the embedding model
        is never loaded.
        """
        return self.w_semantic > 0

    def primary_extra_body(self) -> Optional[dict]:
        """Return the primary-only request flags.

        Returns:
            DeepSeek-style `chat_template_kwargs` that turn thinking off when
            LLM_DISABLE_THINKING is set, else None.
        """
        if self.llm_disable_thinking:
            return {"chat_template_kwargs": {"thinking": False}}
        return None

    def primary_provider(self) -> Provider:
        """Return the primary grading provider."""
        return Provider("primary", self.openai_base_url, self.openai_api_key, self.llm_model,
                        self.primary_extra_body())

    def referee_provider(self) -> Provider:
        """Return the referee provider.

        Each REFEREE_* field defaults to the primary's. The primary's request flags are
        reused only when the referee is the same endpoint and model.
        """
        base_url = self.referee_base_url or self.openai_base_url
        model = self.referee_model or self.llm_model
        extra = self.referee_extra_body
        if extra is None and base_url == self.openai_base_url and model == self.llm_model:
            # Same endpoint + model as the primary: its flags apply unchanged.
            extra = self.primary_extra_body()
        return Provider("referee", base_url, self.referee_api_key or self.openai_api_key, model, extra)

    def fallback_provider(self) -> Optional[Provider]:
        """Return the fallback grader, or None when FALLBACK_MODEL is unset."""
        if not self.fallback_model:
            return None
        return Provider("fallback", self.fallback_base_url or self.openai_base_url,
                        self.fallback_api_key or self.openai_api_key, self.fallback_model,
                        self.fallback_extra_body)

    def judge2_provider(self) -> Optional[Provider]:
        """Return the second grader, or None when JUDGE2_API_KEY is unset."""
        if not self.judge2_api_key:
            return None
        return Provider("judge2", self.judge2_base_url, self.judge2_api_key, self.judge2_model, None)


def _hf_model_cached(model_name: str) -> bool:
    """True if the HuggingFace cache already holds `model_name`."""
    hub = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.path.join(hub, "hub")
    names = [model_name]
    if "/" not in model_name:
        # sentence-transformers prepends its org when the name has no slash
        names.insert(0, "sentence-transformers/" + model_name)
    return any(os.path.isdir(os.path.join(hub, "models--" + n.replace("/", "--"))) for n in names)


settings = Settings()

# If the embedding model is already cached locally, never contact the HuggingFace
# Hub (removes the unauthenticated-request warning, speeds up load, works offline).
if _hf_model_cached(settings.embedding_model):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
