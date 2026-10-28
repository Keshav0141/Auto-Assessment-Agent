"""Provider routing with a fake OpenAI client: nothing leaves the process."""
import json
from types import SimpleNamespace

import pytest

from app.config import settings
from app.grading.engine import GradingEngine
from app.llm import LLMClient, LLMError, OutputBudgetError
from app.models import QuestionSpec, StudentAnswer

PRIMARY_FLAGS = {"chat_template_kwargs": {"thinking": False}}
FALLBACK_FLAGS = {"reasoning": {"enabled": False}}


def grade_json(total, max_marks=5):
    return json.dumps({
        "criterion_scores": [{"criterion": "law", "awarded": total, "max": max_marks, "justification": "ok"}],
        "total_awarded": total,
        "feedback": "fine",
        "integrity_flags": [],
    })


def reply(content, finish_reason="stop"):
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class FakeBackend:
    """Records every request and answers per provider role."""

    def __init__(self):
        self.calls = []
        self.behaviour = {}

    def on(self, role, fn):
        self.behaviour[role] = fn

    def client_for(self, provider):
        backend = self

        class Completions:
            def create(self, **kwargs):
                backend.calls.append((provider.role, kwargs))
                fn = backend.behaviour.get(provider.role)
                if fn is None:
                    raise AssertionError(f"unexpected call to {provider.role}")
                return fn(kwargs)

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    def roles(self):
        return [role for role, _ in self.calls]

    def requests(self, role):
        return [kwargs for r, kwargs in self.calls if r == role]


def fail(message):
    def fn(kwargs):
        raise RuntimeError(message)
    return fn


def ok(total):
    return lambda kwargs: reply(grade_json(total))


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(LLMClient, "_get_client", lambda self: fake.client_for(self.provider))
    for name, value in {
        "llm_provider": "openai",
        "openai_api_key": "test-primary-key",
        "openai_base_url": "http://primary.test/v1",
        "llm_model": "primary-m",
        "llm_native_n": False,
        "llm_disable_thinking": False,
        "max_retries": 0,
        "idle_warmup_seconds": 1e12,
        "judge2_api_key": "",
        "judge2_base_url": "http://judge2.test/v1",
        "judge2_model": "judge2-m",
        "fallback_model": "",
        "fallback_base_url": "",
        "fallback_api_key": "",
        "fallback_extra_body": None,
        "referee_base_url": "",
        "referee_api_key": "",
        "referee_model": "",
        "referee_extra_body": None,
        "adjudicate": True,
    }.items():
        monkeypatch.setattr(settings, name, value)
    return fake


QUESTION = QuestionSpec(question_id="1", question_text="State Newton's second law.", max_marks=5)
ANSWER = StudentAnswer(question_id="1", student_answer_text="Force equals mass times acceleration, F = ma.")


def grade(n=3):
    engine = GradingEngine()
    assert not engine.mock
    return engine.grade_with_confidence(QUESTION, ANSWER, n_samples=n)


# --- fallback chain -----------------------------------------------------------


def test_primary_grades_when_healthy(backend):
    backend.on("primary", ok(4))
    rep, samples, conf = grade()
    assert rep.total_awarded == 4 and len(samples) == 3
    assert conf.grader_role == "primary" and conf.graded_by == "primary-m"
    assert backend.roles() == ["primary"] * 3


def test_fallback_after_primary_fails(backend, monkeypatch):
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    backend.on("primary", fail("503 Service Unavailable"))
    backend.on("fallback", ok(3))
    rep, samples, conf = grade()
    assert conf.grader_role == "fallback" and conf.graded_by == "fallback-m"
    assert rep.total_awarded == 3 and len(samples) == 3
    assert any("fallback model (fallback-m)" in w for w in conf.warnings)
    assert backend.roles().count("primary") == 3  # LLM_MAX_RETRIES=0 -> 3 attempts
    assert all(r["model"] == "fallback-m" for r in backend.requests("fallback"))


def test_second_grader_after_fallback_fails(backend, monkeypatch):
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    monkeypatch.setattr(settings, "judge2_api_key", "test-judge2-key")
    backend.on("primary", fail("503 Service Unavailable"))
    backend.on("fallback", fail("502 Bad Gateway"))
    backend.on("judge2", ok(2))
    rep, samples, conf = grade()
    assert conf.grader_role == "judge2" and conf.graded_by == "judge2-m"
    assert len(samples) == 1 and rep.total_awarded == 2
    assert conf.cross_model_confidence is None  # no independent check left
    assert any("and so did the fallback" in w for w in conf.warnings)


def test_every_grader_failing_raises(backend, monkeypatch):
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    monkeypatch.setattr(settings, "judge2_api_key", "test-judge2-key")
    for role in ("primary", "fallback", "judge2"):
        backend.on(role, fail("503 Service Unavailable"))
    with pytest.raises(LLMError, match="all graders failed"):
        grade()


def test_no_fallback_configured_reraises(backend):
    backend.on("primary", fail("503 Service Unavailable"))
    with pytest.raises(LLMError, match="primary-m call failed after 3 attempts"):
        grade()


def test_auth_error_never_falls_back(backend, monkeypatch):
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    backend.on("primary", fail("Error code: 401 - invalid api key"))
    backend.on("fallback", ok(5))
    with pytest.raises(LLMError, match="invalid API key"):
        grade()
    assert backend.roles() == ["primary"]


# --- extra_body scoping ---------------------------------------------------------


def test_extra_body_stays_with_its_provider(backend, monkeypatch):
    monkeypatch.setattr(settings, "llm_disable_thinking", True)
    monkeypatch.setattr(settings, "judge2_api_key", "test-judge2-key")
    backend.on("primary", ok(1))
    backend.on("judge2", ok(5))  # disagrees by 4/5 -> referee
    backend.on("referee", ok(3))
    rep, _, conf = grade()
    assert conf.adjudicated and rep.total_awarded == 3 and conf.pre_adjudication_total == 1
    assert all(r["extra_body"] == PRIMARY_FLAGS for r in backend.requests("primary"))
    assert all("extra_body" not in r for r in backend.requests("judge2"))
    # The referee defaults to the primary's endpoint + model, so it shares its flags.
    assert [r["extra_body"] for r in backend.requests("referee")] == [PRIMARY_FLAGS]


def test_fallback_and_other_referee_get_only_their_own_flags(backend, monkeypatch):
    monkeypatch.setattr(settings, "llm_disable_thinking", True)
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    monkeypatch.setattr(settings, "fallback_extra_body", FALLBACK_FLAGS)
    monkeypatch.setattr(settings, "referee_model", "referee-m")
    assert settings.fallback_provider().extra_body == FALLBACK_FLAGS
    assert settings.referee_provider().extra_body is None
    assert settings.judge2_provider() is None

    backend.on("fallback", lambda kwargs: reply("{}"))
    backend.on("referee", lambda kwargs: reply("{}"))
    LLMClient(settings.fallback_provider()).chat_json("s", "u")
    LLMClient(settings.referee_provider()).chat_json("s", "u")
    assert backend.requests("fallback")[0]["extra_body"] == FALLBACK_FLAGS
    assert "extra_body" not in backend.requests("referee")[0]


def test_extra_body_is_copied_per_request(backend, monkeypatch):
    monkeypatch.setattr(settings, "llm_disable_thinking", True)
    backend.on("primary", lambda kwargs: kwargs["extra_body"].clear() or reply("{}"))
    client = LLMClient()
    client.chat_json("s", "u")
    client.chat_json("s", "u")
    assert settings.primary_extra_body() == PRIMARY_FLAGS


# --- output budget --------------------------------------------------------------


def test_output_budget_error_is_not_retried(backend):
    backend.on("primary", lambda kwargs: reply("", finish_reason="length"))
    with pytest.raises(OutputBudgetError, match="primary-m"):
        LLMClient().chat_json("s", "u")
    assert backend.roles() == ["primary"]


def test_output_budget_error_native_n(backend):
    backend.on("primary", lambda kwargs: reply("", finish_reason="length"))
    with pytest.raises(OutputBudgetError):
        LLMClient().chat_json_n("s", "u", n=3)
    assert len(backend.calls) == 1 and backend.calls[0][1]["n"] == 3


def test_output_budget_error_moves_to_fallback(backend, monkeypatch):
    monkeypatch.setattr(settings, "fallback_model", "fallback-m")
    backend.on("primary", lambda kwargs: reply("", finish_reason="length"))
    backend.on("fallback", ok(4))
    _, _, conf = grade()
    assert conf.grader_role == "fallback"
    assert backend.roles().count("primary") == 1


def test_empty_reply_without_length_is_retried(backend):
    replies = iter([reply(""), reply(grade_json(2))])
    backend.on("primary", lambda kwargs: next(replies))
    assert LLMClient().chat_json("s", "u")["total_awarded"] == 2
    assert backend.roles() == ["primary", "primary"]


# --- feedback language ------------------------------------------------------------


HINDI_ANSWER = StudentAnswer(question_id="1", student_answer_text="बल द्रव्यमान और त्वरण का गुणनफल है, F = ma।")


def _user_message(kwargs):
    return next(m["content"] for m in kwargs["messages"] if m["role"] == "user")


def test_feedback_language_reaches_every_grader(backend, monkeypatch):
    monkeypatch.setattr(settings, "judge2_api_key", "test-judge2-key")
    backend.on("primary", ok(1))
    backend.on("judge2", ok(5))
    backend.on("referee", ok(3))
    _, _, conf = GradingEngine().grade_with_confidence(QUESTION, HINDI_ANSWER, n_samples=2)
    assert conf.feedback_language == "Hindi" and conf.adjudicated
    for role in ("primary", "judge2", "referee"):
        for request in backend.requests(role):
            assert "in Hindi (Devanagari script)" in _user_message(request), role


def test_english_answer_gets_no_language_line(backend):
    backend.on("primary", ok(4))
    grade()
    assert all("LANGUAGE:" not in _user_message(r) for r in backend.requests("primary"))


def test_explicit_feedback_language_overrides_auto(backend):
    backend.on("primary", ok(4))
    _, _, conf = GradingEngine().grade_with_confidence(QUESTION, ANSWER, n_samples=1, feedback_language="ta")
    assert conf.feedback_language == "Tamil"
    assert "in Tamil (Tamil script)" in _user_message(backend.requests("primary")[0])


# --- feedback script check ---------------------------------------------------------


TELUGU_ANSWER = StudentAnswer(question_id="1", student_answer_text="బలం ద్రవ్యరాశి మరియు త్వరణం యొక్క లబ్ధం, F = ma.")
CLEAN_TELUGU = "సరైన సమాధానం."
DRIFTED_TELUGU = "conversion stepあげて లెక్కించాలి."


def graded(total, feedback):
    return reply(json.dumps({
        "criterion_scores": [{"criterion": "law", "awarded": total, "max": 5, "justification": feedback}],
        "total_awarded": total, "feedback": feedback, "integrity_flags": [],
    }))


def test_drifted_feedback_is_regenerated_once(backend):
    replies = iter([graded(4, DRIFTED_TELUGU), graded(4, CLEAN_TELUGU)])
    backend.on("primary", lambda kwargs: next(replies))
    rep, samples, conf = GradingEngine().grade_with_confidence(QUESTION, TELUGU_ANSWER, n_samples=1)
    assert rep.feedback == CLEAN_TELUGU
    assert len(backend.requests("primary")) == 2
    assert not any("even after a retry" in w for w in conf.warnings)


def test_persistent_drift_is_kept_with_a_warning(backend):
    backend.on("primary", lambda kwargs: graded(4, DRIFTED_TELUGU))
    rep, samples, conf = GradingEngine().grade_with_confidence(QUESTION, TELUGU_ANSWER, n_samples=2)
    assert len(samples) == 2 and rep.feedback == DRIFTED_TELUGU
    assert len(backend.requests("primary")) == 4  # one retry per sample, no more
    warning = next(w for w in conf.warnings if "even after a retry" in w)
    assert "2 marking(s)" in warning and "Japanese kana" in warning and "Telugu" in warning
    assert "_foreign_scripts" not in rep.model_dump()


def test_clean_feedback_is_not_regenerated(backend):
    backend.on("primary", lambda kwargs: graded(4, CLEAN_TELUGU))
    GradingEngine().grade_with_confidence(QUESTION, TELUGU_ANSWER, n_samples=3)
    assert len(backend.requests("primary")) == 3


def test_native_n_regenerates_only_the_drifted_choice(backend, monkeypatch):
    monkeypatch.setattr(settings, "llm_native_n", True)

    def fn(kwargs):
        if kwargs.get("n"):
            choices = graded(4, CLEAN_TELUGU).choices + graded(3, DRIFTED_TELUGU).choices
            return SimpleNamespace(choices=choices)
        return graded(3, CLEAN_TELUGU)

    backend.on("primary", fn)
    _, samples, conf = GradingEngine().grade_with_confidence(QUESTION, TELUGU_ANSWER, n_samples=2)
    assert [s.feedback for s in samples] == [CLEAN_TELUGU, CLEAN_TELUGU]
    assert [bool(r.get("n")) for r in backend.requests("primary")] == [True, False]


def test_referee_feedback_is_checked(backend, monkeypatch):
    monkeypatch.setattr(settings, "judge2_api_key", "test-judge2-key")
    backend.on("primary", lambda kwargs: graded(1, CLEAN_TELUGU))
    backend.on("judge2", lambda kwargs: graded(5, CLEAN_TELUGU))
    backend.on("referee", lambda kwargs: graded(3, DRIFTED_TELUGU))
    rep, _, conf = GradingEngine().grade_with_confidence(QUESTION, TELUGU_ANSWER, n_samples=1)
    assert conf.adjudicated and rep.total_awarded == 3
    assert len(backend.requests("referee")) == 2
    assert any("1 marking(s)" in w for w in conf.warnings)
