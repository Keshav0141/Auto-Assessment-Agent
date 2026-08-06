"""Application configuration, loaded from environment variables (.env supported)."""
import os

from dotenv import load_dotenv

load_dotenv()


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
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        self.grading_temperature = _env_float("GRADING_TEMPERATURE", 0.7)
        self.max_retries = _env_int("LLM_MAX_RETRIES", 3)
        self.request_pacing = _env_float("LLM_REQUEST_PACING", 0.0)
        self.idle_warmup_seconds = _env_float("IDLE_WARMUP_SECONDS", 90.0)

        self.n_samples = _env_int("CONFIDENCE_SAMPLES", 5)
        self.confidence_threshold = _env_float("CONFIDENCE_THRESHOLD", 0.6)
        self.w_variance = _env_float("W_VARIANCE", 0.5)
        self.w_semantic = _env_float("W_SEMANTIC", 0.5)
        self.cluster_threshold = _env_float("CLUSTER_THRESHOLD", 0.72)
        self.error_margin = _env_float("ERROR_MARGIN", 0.5)

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

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
