"""OpenAI-compatible chat client with retry/backoff and robust JSON extraction.

Rate limits (HTTP 429) get exponential backoff, which free-tier APIs with tight
per-minute limits need. Auth errors and exhausted output budgets are raised at once,
since retrying cannot fix them.
"""
import json
import re
import time
from typing import Any, Callable, List, Optional

from app.config import Provider, settings


class LLMError(RuntimeError):
    """An LLM call failed or returned unusable output."""


class RateLimitError(LLMError):
    """Raised when the API returns a 429 (rate-limit/quota) error."""


class OutputBudgetError(LLMError):
    """The reply hit max_tokens before any content was written.

    A reasoning model spent the whole budget thinking. Retrying the same request
    cannot help.
    """


def check_output_budget(choice: Any, model: str) -> None:
    """Raise OutputBudgetError when a choice is empty and finish_reason is "length".

    Args:
        choice: One `choices[i]` entry of a chat completion.
        model: Model name, for the error message.

    Raises:
        OutputBudgetError: If the reply is empty because the budget ran out.
    """
    content = (choice.message.content or "") if getattr(choice, "message", None) else ""
    if not content.strip() and getattr(choice, "finish_reason", None) == "length":
        raise OutputBudgetError(
            f"reasoning used the whole output budget for {model} (empty reply, "
            "finish_reason=length); disable reasoning via its *_EXTRA_BODY or raise max_tokens"
        )


def extract_json(content: str) -> Any:
    """Extract a JSON object from an LLM reply.

    Strips code fences, takes the outermost `{...}` and repairs a common missing-comma
    error between list objects.

    Args:
        content: The raw reply text.

    Returns:
        The parsed JSON value.

    Raises:
        LLMError: If there is no JSON object or it does not parse.
    """
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


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error."""
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate" in exc_str or "quota" in exc_str or "resource_exhausted" in exc_str:
        return True
    # OpenAI SDK raises specific error types
    cls_name = type(exc).__name__.lower()
    if "ratelimit" in cls_name:
        return True
    return False


def _is_auth_error(exc: Exception) -> bool:
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

    Rate limits get exponential backoff (5s, 15s, 45s, ... capped at 120s); malformed
    or empty JSON gets a short backoff and a retry.

    Args:
        provider: Endpoint, key, model and provider-specific extra body to use.
            Defaults to the primary grading LLM, re-read on every call.
    """

    def __init__(self, provider: Optional[Provider] = None):
        self._provider = provider
        self._client = None
        self._client_key = None

    @property
    def provider(self) -> Provider:
        """The bound provider (the primary when none was given)."""
        return self._provider or settings.primary_provider()

    def _get_client(self):
        p = self.provider
        key = (p.base_url, p.api_key)
        if self._client is None or self._client_key != key:
            from openai import OpenAI
            self._client = OpenAI(base_url=p.base_url, api_key=p.api_key)
            self._client_key = key
        return self._client

    def _request(self, **kwargs):
        p = self.provider
        kwargs["model"] = p.model
        if p.extra_body:
            kwargs["extra_body"] = dict(p.extra_body)
        return self._get_client().chat.completions.create(**kwargs)

    def _auth_error(self, exc):
        return LLMError(f"invalid API key for {self.provider.base_url}")

    def warmup(self) -> None:
        """Send a tiny request to wake the provider.

        Free tiers have slow cold starts. Best-effort: only an invalid key raises.

        Raises:
            LLMError: If the API key is rejected.
        """
        if settings.using_mock:
            return
        try:
            self._request(
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0.0,
                max_tokens=10,
            )
        except Exception as exc:
            if _is_auth_error(exc):
                raise self._auth_error(exc) from exc

    def _with_retries(self, call: Callable[[], Any], max_attempts: Optional[int] = None) -> Any:
        max_attempts = max_attempts or settings.max_retries + 3  # extra retries for rate limits
        last_error = None
        for attempt in range(max_attempts):
            try:
                return call()
            except Exception as exc:
                if _is_auth_error(exc):
                    raise self._auth_error(exc) from exc
                if isinstance(exc, OutputBudgetError):
                    raise  # deterministic for this config: retrying cannot help
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                if isinstance(exc, LLMError):
                    # Empty / malformed JSON output: short backoff, then retry.
                    time.sleep(1.5 * (attempt + 1))
                elif _is_rate_limit(exc):
                    # Exponential backoff for rate limits: 5s, 15s, 45s, ...
                    time.sleep(min(5 * (3 ** attempt), 120))
                else:
                    # Shorter backoff for other transient errors
                    time.sleep(2.0 * (attempt + 1))
        raise LLMError(
            f"{self.provider.model} call failed after {max_attempts} attempts: {last_error}"
        )

    def chat_json(self, system: str, user: str, temperature: Optional[float] = None, max_tokens: int = 1500,
                  max_attempts: Optional[int] = None) -> Any:
        """Send one chat request in JSON mode and parse the reply.

        Args:
            system: System message.
            user: User message.
            temperature: Sampling temperature (default: GRADING_TEMPERATURE).
            max_tokens: Output token budget.
            max_attempts: Attempts before giving up (default: LLM_MAX_RETRIES + 3).

        Returns:
            The parsed JSON reply.

        Raises:
            LLMError: On an auth error, an exhausted output budget, or when every
                attempt fails.
        """
        temperature = settings.grading_temperature if temperature is None else temperature

        def call():
            resp = self._request(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            check_output_budget(resp.choices[0], self.provider.model)
            content = resp.choices[0].message.content or ""
            if not content.strip():
                # Free-tier gateways sometimes return 200 with empty content
                # right before a rate limit. Treat as retryable.
                raise LLMError("Empty response from model")
            return extract_json(content)

        return self._with_retries(call, max_attempts)

    def chat_json_n(self, system: str, user: str, n: int, temperature: Optional[float] = None,
                    max_tokens: int = 1500) -> List[Any]:
        """Send one request for `n` choices and parse each.

        Args:
            system: System message.
            user: User message.
            n: Number of choices to request.
            temperature: Sampling temperature (default: GRADING_TEMPERATURE).
            max_tokens: Output token budget per choice.

        Returns:
            The parsed JSON dicts; unparseable choices are skipped.

        Raises:
            LLMError: If no choice parses after retries.
        """
        temperature = settings.grading_temperature if temperature is None else temperature

        def call():
            resp = self._request(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                response_format={"type": "json_object"},
            )
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
                for choice in resp.choices:
                    check_output_budget(choice, self.provider.model)
                raise LLMError("No parseable JSON choices in response")
            return parsed

        return self._with_retries(call)
