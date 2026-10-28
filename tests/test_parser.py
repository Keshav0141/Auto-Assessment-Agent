import pytest

from app.ingestion.parser import (
    NoTextError,
    ParseError,
    apply_rubric,
    default_criterion,
    extract_text_meta,
    parse_answer_sheet,
    parse_model_answers,
    parse_question_paper,
    parse_rubric,
)
from app.models import QuestionSpec, RubricCriterion

# --- question anchors -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Question 1 (5 marks)\nFirst?\n\nQuestion 2 (3 marks)\nSecond?",
        "Q1 (5 marks)\nFirst?\n\nQ2 (3 marks)\nSecond?",
        "Q.1 (5 marks)\nFirst?\n\nQ.2 (3 marks)\nSecond?",
        "Problem 1 (5 marks)\nFirst?\n\nProblem 2 (3 marks)\nSecond?",
        "question 1 (5 marks)\nFirst?\n\nQUESTION 2 (3 marks)\nSecond?",
        "1. First? (5 marks)\n2) Second? (3 marks)",
    ],
)
def test_english_anchors(text):
    qs = parse_question_paper(text)
    assert [q.question_id for q in qs] == ["1", "2"]
    assert [q.max_marks for q in qs] == [5.0, 3.0]
    assert "First?" in qs[0].question_text
    assert all(q.marks_specified for q in qs)


@pytest.mark.parametrize("unit", ["marks", "mark", "points", "pts"])
def test_marks_units(unit):
    qs = parse_question_paper(f"Question 1 (2.5 {unit})\nExplain.")
    assert qs[0].max_marks == 2.5


def test_marks_in_body_when_header_has_none():
    qs = parse_question_paper("Question 1\nExplain gravity. [4 marks]")
    assert qs[0].max_marks == 4.0


def test_question_without_marks_defaults_to_one():
    qs = parse_question_paper("Question 1\nExplain gravity.")
    assert qs[0].max_marks == 1.0
    assert qs[0].marks_specified is False


def test_no_questions_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_question_paper("Just some prose with no question labels at all.")


def test_duplicate_questions_keep_first_and_warn():
    warnings = []
    qs = parse_question_paper("Question 1 (2 marks)\nA\nQuestion 1 (9 marks)\nB", warnings)
    assert len(qs) == 1 and qs[0].max_marks == 2.0
    assert any("repeats" in w for w in warnings)


def test_structured_json_paper():
    text = (
        '[{"question_id": "7", "question_text": "Why?", "max_marks": 4,'
        ' "rubric_criteria": [{"criterion": "reason", "marks": 4}], "model_answer": "Because."}]'
    )
    qs = parse_question_paper(text)
    assert qs[0].question_id == "7"
    assert qs[0].max_marks == 4.0
    assert qs[0].rubric_criteria[0].criterion == "reason"
    assert qs[0].model_answer == "Because."


def test_sample_question_paper(sample_text):
    qs = parse_question_paper(sample_text("question_paper.txt"))
    assert [(q.question_id, q.max_marks) for q in qs] == [("1", 5.0), ("2", 3.0), ("3", 4.0), ("4", 6.0)]


# --- rubric -------------------------------------------------------------------


def test_rubric_criteria_and_marks():
    rubric = parse_rubric(
        "Question 1\n2 marks for stating the law, 1 mark for F=ma\n\n"
        "Question 2\n3 points: correct diagram"
    )
    assert [(c.criterion, c.marks) for c in rubric["1"]] == [("stating the law", 2.0), ("F=ma", 1.0)]
    assert [(c.criterion, c.marks) for c in rubric["2"]] == [("correct diagram", 3.0)]


def test_rubric_without_anchors_belongs_to_question_one():
    rubric = parse_rubric("2 marks for the method, 1 mark for the answer")
    assert list(rubric) == ["1"]
    assert sum(c.marks for c in rubric["1"]) == 3.0


def test_sample_rubric_sums_match_the_paper(sample_text):
    qs = parse_question_paper(sample_text("question_paper.txt"))
    rubric = parse_rubric(sample_text("rubric.txt"))
    for q in qs:
        assert sum(c.marks for c in rubric[q.question_id]) == q.max_marks
        assert apply_rubric(q, rubric[q.question_id]) == []


def test_rubric_scaled_to_paper_marks():
    q = QuestionSpec(question_id="1", question_text="?", max_marks=4)
    warnings = apply_rubric(q, [RubricCriterion(criterion="a", marks=6), RubricCriterion(criterion="b", marks=2)])
    assert [c.marks for c in q.rubric_criteria] == [3.0, 1.0]
    assert q.max_marks == 4
    assert "precedence" in warnings[0]


def test_rubric_total_used_when_paper_has_no_marks():
    q = QuestionSpec(question_id="1", question_text="?", max_marks=1, marks_specified=False)
    warnings = apply_rubric(q, [RubricCriterion(criterion="a", marks=3), RubricCriterion(criterion="b", marks=2)])
    assert q.max_marks == 5
    assert "rubric total" in warnings[0]


def test_no_rubric_and_no_marks_warns():
    q = QuestionSpec(question_id="1", question_text="?", max_marks=1, marks_specified=False)
    warnings = apply_rubric(q, [])
    assert q.rubric_criteria == [default_criterion(1)]
    assert "no marks and there is no rubric" in warnings[0]


def test_no_rubric_with_marks_is_silent():
    q = QuestionSpec(question_id="1", question_text="?", max_marks=5)
    assert apply_rubric(q, None) == []
    assert q.rubric_criteria[0].marks == 5.0


# --- model answers and answer sheets --------------------------------------------


def test_model_answers(sample_text):
    answers = parse_model_answers(sample_text("model_answers.txt"))
    assert set(answers) == {"1", "2", "3", "4"}
    assert "F = ma" in answers["1"]
    assert parse_model_answers("just one answer") == {"1": "just one answer"}


def test_answer_sheet_segments(sample_text):
    answers, warnings = parse_answer_sheet(sample_text("answer_sheet_good.txt"), ["1", "2", "3", "4"])
    assert set(answers) == {"1", "2", "3", "4"}
    assert "photosynthesis" in answers["2"].student_answer_text.lower()
    assert warnings == []


def test_answer_sheet_joins_repeated_questions():
    answers, warnings = parse_answer_sheet("Question 1\nfirst part\nQuestion 1\nsecond part", ["1"])
    assert answers["1"].student_answer_text == "first part\n\nsecond part"
    assert any("joined" in w for w in warnings)


def test_answer_sheet_drops_unknown_questions():
    answers, warnings = parse_answer_sheet("Question 1\nyes\nQuestion 9\nextra", ["1"])
    assert set(answers) == {"1"}
    assert any("Q9" in w for w in warnings)


def test_answer_sheet_without_anchors():
    answers, warnings = parse_answer_sheet("an answer with no label", ["1"])
    assert answers is None
    assert warnings


# --- file extraction -----------------------------------------------------------


def test_extract_plain_text():
    text, ocr = extract_text_meta("paper.txt", "Question 1 (2 marks)\nWhy?".encode("utf-8"))
    assert text.startswith("Question 1") and ocr is None


def test_extract_rejects_unsupported_extension():
    with pytest.raises(ParseError, match="Unsupported file type"):
        extract_text_meta("paper.docx", b"PK\x03\x04 not really a docx")


def test_extract_rejects_binary_text_file():
    with pytest.raises(ParseError, match="does not look like a text file"):
        extract_text_meta("paper.txt", b"\x00\x01\x02binary")


def test_extract_rejects_corrupt_pdf():
    with pytest.raises(ParseError, match="corrupt"):
        extract_text_meta("paper.pdf", b"not a pdf at all")


def test_extract_empty_text_file():
    with pytest.raises(NoTextError):
        extract_text_meta("paper.txt", b"   \n")
