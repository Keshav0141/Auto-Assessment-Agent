import pytest

from app.grading.scripts import foreign_scripts, scripts_in


def test_scripts_in_ignores_digits_punctuation_and_symbols():
    assert scripts_in("F = ma, 5 m/s^2 × 2 → 10 ।") == {"Latin"}
    assert scripts_in("ఆక్సిజన్ あげて 中文 привет") == {"Telugu", "Japanese kana", "Han", "Cyrillic"}


@pytest.mark.parametrize(
    "text,language,expected",
    [
        ("సరైన సమాధానం, F = ma అని రాశారు.", "Telugu", []),
        ("conversion step showingあげて లెక్కించాలి", "Telugu", ["Japanese kana"]),
        ("সঠিক উত্তর। ভালো।", "Bengali", []),  # the danda is punctuation, not Devanagari
        ("Correct, θ = 30° and Δv = 5 m/s.", "English", []),  # Greek symbols are fine
        ("सही उत्तर, लेकिन 単位 गलत है।", "Hindi", ["Han"]),
        ("Sahi jawab hai, F = ma.", "Hinglish", []),
        ("सही जवाब", "Hinglish", ["Devanagari"]),
        ("Tamil text வணக்கம் in Telugu feedback", "Telugu", ["Tamil"]),
        ("Réponse correcte, mais 単位", "French", []),  # unknown script for the language: no check
    ],
)
def test_foreign_scripts(text, language, expected):
    assert foreign_scripts(text, language) == expected


def test_quoting_the_student_is_allowed():
    answer = "बल = द्रव्यमान × त्वरण"
    assert foreign_scripts("You wrote 'बल = द्रव्यमान × त्वरण', which is correct.", "English", [answer]) == []
    assert foreign_scripts("You wrote 'बल', which is correct.", "English") == ["Devanagari"]
