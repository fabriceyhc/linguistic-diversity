"""Tests for syllable stress and weight analysis.

RhythmicAnalyzer backs the Rhythmic metric. It uses pyphen for syllabification
and the CMU pronouncing dictionary for stress, falling back to heuristics for
words it does not know, so both paths are covered here.
"""

import pytest

pyphen = pytest.importorskip("pyphen", reason="phonological extras not installed")
pytest.importorskip("pronouncing", reason="phonological extras not installed")

from linguistic_diversity.diversities.rhythmic_analyzer import (
    RhythmicAnalyzer,
)


@pytest.fixture(scope="module")
def analyzer():
    return RhythmicAnalyzer()


class TestAnalyzeWord:
    """Per-word syllable analysis."""

    def test_returns_one_entry_per_syllable(self, analyzer):
        syllables = analyzer.analyze_word("water")

        assert len(syllables) >= 1
        assert all({"stress", "weight"} <= set(s) for s in syllables)

    def test_multisyllabic_word_yields_multiple_syllables(self, analyzer):
        assert len(analyzer.analyze_word("communication")) > len(analyzer.analyze_word("cat"))

    def test_empty_and_whitespace_input(self, analyzer):
        assert analyzer.analyze_word("") == []
        assert analyzer.analyze_word("   ") == []

    def test_is_case_insensitive(self, analyzer):
        assert analyzer.analyze_word("Water") == analyzer.analyze_word("water")

    def test_stress_and_weight_are_binary(self, analyzer):
        for word in ("computer", "diversity", "run", "linguistics"):
            for syllable in analyzer.analyze_word(word):
                assert syllable["stress"] in (0, 1), word
                assert syllable["weight"] in (0, 1), word

    def test_unknown_word_falls_back_to_heuristics(self, analyzer):
        """Words absent from CMUdict must still produce a pattern, not crash."""
        syllables = analyzer.analyze_word("zorblattian")

        assert syllables
        assert all(s["stress"] in (0, 1) for s in syllables)


class TestSyllableWeight:
    """Weight follows the documented heavy/light rules."""

    def test_closed_syllable_is_heavy(self, analyzer):
        assert analyzer._calculate_syllable_weight("cat") == 1

    def test_long_vowel_is_heavy(self, analyzer):
        assert analyzer._calculate_syllable_weight("bee") == 1
        assert analyzer._calculate_syllable_weight("boat") == 1

    def test_diphthong_is_heavy(self, analyzer):
        assert analyzer._calculate_syllable_weight("cow") == 1

    def test_open_short_vowel_is_light(self, analyzer):
        assert analyzer._calculate_syllable_weight("ba") == 0

    def test_empty_syllable_is_light(self, analyzer):
        assert analyzer._calculate_syllable_weight("") == 0


class TestAnalyzeText:
    """Whole-text analysis."""

    def test_returns_one_record_per_word(self, analyzer):
        analyses = analyzer.analyze_text("the quick brown fox")

        assert len(analyses) == 4
        assert all("syllables" in a for a in analyses)

    def test_empty_text_yields_nothing(self, analyzer):
        assert analyzer.analyze_text("") == []

    def test_punctuation_does_not_create_words(self, analyzer):
        analyses = analyzer.analyze_text("hello, world!")

        assert len(analyses) == 2


class TestRhythmPattern:
    """The stress+weight codes consumed by the Rhythmic metric."""

    def test_one_code_per_word(self, analyzer):
        pattern = analyzer.extract_rhythm_pattern("the quick brown fox")

        assert len(pattern) == 4

    def test_codes_are_two_binary_digits(self, analyzer):
        for code in analyzer.extract_rhythm_pattern("diversity measures linguistic variation"):
            assert code in {"00", "01", "10", "11"}, code

    def test_is_deterministic(self, analyzer):
        text = "the quick brown fox jumps"

        assert analyzer.extract_rhythm_pattern(text) == analyzer.extract_rhythm_pattern(text)

    def test_different_text_gives_different_pattern(self, analyzer):
        """A flat metric would return the same codes regardless of input."""
        a = analyzer.extract_rhythm_pattern("cat dog run")
        b = analyzer.extract_rhythm_pattern("beautiful communication happening")

        assert a != b

    def test_empty_text_yields_empty_pattern(self, analyzer):
        assert analyzer.extract_rhythm_pattern("") == []
