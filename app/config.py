"""Application configuration, loaded from environment variables (.env supported)."""
import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_int(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
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

        # OCR: auto | gemini | tesseract | bodhan
        self.ocr_provider = os.getenv("OCR_PROVIDER", "auto").strip().lower()
        self.ocr_model = os.getenv("OCR_MODEL", "").strip() or self.llm_model
        self.ocr_base_url = (os.getenv("OCR_BASE_URL", "").strip() or self.openai_base_url).rstrip("/")
        self.ocr_api_key = os.getenv("OCR_API_KEY", "").strip() or self.openai_api_key
        self.ocr_disable_thinking = os.getenv("OCR_DISABLE_THINKING", "false").lower() in ("1", "true", "yes")

    @property
    def using_mock(self) -> bool:
        return self.llm_provider == "mock" or not self.openai_api_key


def _hf_model_cached(model_name: str) -> bool:
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
