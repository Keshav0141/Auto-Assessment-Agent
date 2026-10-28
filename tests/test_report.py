"""The README evaluation tables must match eval/results/*.json exactly."""
import pytest

from eval.report import README, apply, load_results, render


def test_readme_tables_match_the_results():
    with open(README, encoding="utf-8") as f:
        text = f.read()
    assert apply(text, render(load_results())) == text, "run: python -m eval.report"


def test_every_block_has_markers():
    with open(README, encoding="utf-8") as f:
        text = f.read()
    for name in render(load_results()):
        assert f"<!-- report:{name} -->" in text and f"<!-- /report:{name} -->" in text


def test_apply_requires_markers():
    with pytest.raises(KeyError):
        apply("no markers here", {"grading": "x"})
    assert apply("a\n<!-- report:x -->\nold\n<!-- /report:x -->\nb", {"x": "new"}) == (
        "a\n<!-- report:x -->\nnew\n<!-- /report:x -->\nb"
    )


def test_apply_fills_an_empty_block():
    assert apply("<!-- report:x -->\n<!-- /report:x -->", {"x": "new"}) == "<!-- report:x -->\nnew\n<!-- /report:x -->"
