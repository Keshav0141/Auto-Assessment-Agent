"""OpenAI-compatible chat client with retry/backoff and robust JSON extraction.

Handles rate-limiting (HTTP 429) with exponential backoff — critical for free-tier
APIs like Google Gemini Flash which enforce tight RPM limits.
"""
import json
import re
import time

from app.config import settings


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    """Raised when the API returns a 429 (rate-limit/quota) error."""
    pass


def extract_json(content: str):
    """Best-effort extraction of a JSON object from an LLM response."""
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        snippet = content[:200] + ("..." if len(content) > 200 else "")
        raise LLMError(f"Model returned no JSON object. Raw output: '{snippet}'")
    
    json_str = content[start:end + 1]
    
    # Common DeepSeek error: missing comma between objects in a list: `} {` or `}, \n {` (without comma)
    json_str = re.sub(r'\}\s*\{', '}, {', json_str)
    json_str = re.sub(r'\}\s*\n\s*\{', '},\n{', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Malformed JSON from model: {exc}") from exc


def _is_rate_limit(exc):
    """Check if an exception is a rate-limit (429) error."""
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate" in exc_str or "quota" in exc_str or "resource_exhausted" in exc_str:
        return True
    # OpenAI SDK raises specific error types
    cls_name = type(exc).__name__.lower()
    if "ratelimit" in cls_name:
        return True
    return False


def _is_auth_error(exc):
    """Check if an exception is an authentication/permission (401/403) error."""
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    exc_str = str(exc).lower()
    if "401" in exc_str or "403" in exc_str:
        return True
    if "invalid api key" in exc_str or "invalid_api_key" in exc_str:
        return True
    if "authentication" in exc_str or "permission denied" in exc_str or "permission_denied" in exc_str:
        return True
    return False


class LLMClient:
    """Minimal OpenAI-compatible client with retry/backoff and forced-JSON parsing.

    For rate-limited APIs (especially Gemini free tier at 15 RPM), uses aggressive
    exponential backoff: 5s → 15s → 45s on 429 errors.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        return self._client

    @staticmethod
    def _extra_body():
        """Disable reasoning/thinking tokens when the provider supports it
        (e.g. DeepSeek V4.1 via Goedel Machines)."""
        if settings.llm_disable_thinking:
            return {"chat_template_kwargs": {"thinking": False}}
        return None

    def warmup(self):
        """Fire a tiny request to keep the provider instance warm (free tiers have
        slow cold starts). Best-effort; never raises."""
        if settings.using_mock:
            return
        try:
            client = self._get_client()
            kwargs = dict(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0.0,
                max_tokens=10,
            )
            extra_body = self._extra_body()
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            client.chat.completions.create(**kwargs)
        except Exception as exc:
            if _is_auth_error(exc):
                raise LLMError(f"invalid API key for {settings.openai_base_url}") from exc
            pass

    def chat_json(self, system, user, temperature=None, max_tokens=1500):
        temperature = settings.grading_temperature if temperature is None else temperature
        max_attempts = settings.max_retries + 3  # extra retries for rate limits
        last_error = None
        for attempt in range(max_attempts):
            try:
                client = self._get_client()
                kwargs = dict(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                extra_body = self._extra_body()
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                if not content.strip():
                    # Free-tier gateways sometimes return 200 with empty content
                    # right before a rate limit. Treat as retryable.
                    raise LLMError("Empty response from model")
                return extract_json(content)
            except Exception as exc:
                if _is_auth_error(exc):
                    raise LLMError(f"invalid API key for {settings.openai_base_url}") from exc
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                if isinstance(exc, LLMError):
                    # Empty / malformed JSON output: short backoff, then retry.
                    time.sleep(1.5 * (attempt + 1))
                elif _is_rate_limit(exc):
                    # Exponential backoff for rate limits: 5s, 15s, 45s, ...
                    wait = min(5 * (3 ** attempt), 120)
                    time.sleep(wait)
                else:
                    # Shorter backoff for other transient errors
                    time.sleep(2.0 * (attempt + 1))
        raise LLMError(f"LLM call failed after {max_attempts} attempts: {last_error}")

    def chat_json_n(self, system, user, n, temperature=None, max_tokens=1500):
        """One request asking for `n` choices; returns a list of parsed JSON dicts
        (unparseable choices are skipped). Raises LLMError if none parse."""
        temperature = settings.grading_temperature if temperature is None else temperature
        max_attempts = settings.max_retries + 3  # extra retries for rate limits
        last_error = None
        for attempt in range(max_attempts):
            try:
                client = self._get_client()
                kwargs = dict(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    response_format={"type": "json_object"},
                )
                extra_body = self._extra_body()
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                resp = client.chat.completions.create(**kwargs)
                parsed = []
                for choice in resp.choices:
                    content = (choice.message.content or "") if choice.message else ""
                    if not content.strip():
                        continue
                    try:
                        parsed.append(extract_json(content))
                    except LLMError:
                        continue
                if not parsed:
                    raise LLMError("No parseable JSON choices in response")
                return parsed
            except Exception as exc:
                if _is_auth_error(exc):
                    raise LLMError(f"invalid API key for {settings.openai_base_url}") from exc
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                if isinstance(exc, LLMError):
                    time.sleep(1.5 * (attempt + 1))
                elif _is_rate_limit(exc):
                    wait = min(5 * (3 ** attempt), 120)
                    time.sleep(wait)
                else:
                    time.sleep(2.0 * (attempt + 1))
        raise LLMError(f"LLM call failed after {max_attempts} attempts: {last_error}")
