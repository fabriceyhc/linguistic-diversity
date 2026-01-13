"""Custom rhythmic analysis module - replacement for broken cadences library.

This module provides syllable stress and weight analysis using pure Python libraries:
- pyphen for syllabification
- pronouncing for stress patterns from CMU dictionary
- Custom logic for syllable weight
"""

from __future__ import annotations

import re
from typing import Any

import pyphen
import pronouncing


class RhythmicAnalyzer:
    """Analyzes rhythmic patterns in text using syllable stress and weight.

    This is a pure Python replacement for the cadences library, which is broken on PyPI.
    """

    def __init__(self, lang: str = "en_US") -> None:
        """Initialize the rhythmic analyzer.

        Args:
            lang: Language code for pyphen (default: en_US).
        """
        self.dic = pyphen.Pyphen(lang=lang)
        # Preload pronouncing data
        try:
            # Force download of CMU dictionary if needed
            _ = pronouncing.phones_for_word("test")
        except Exception:
            pass

    def analyze_word(self, word: str) -> list[dict[str, Any]]:
        """Analyze a single word to extract syllable information.

        Args:
            word: The word to analyze.

        Returns:
            List of syllable dictionaries with 'stress' and 'weight' keys.
        """
        word = word.lower().strip()
        if not word:
            return []

        # Get syllables using pyphen
        syllables = self.dic.inserted(word).split("-")

        # Get stress pattern from pronouncing (CMU dict)
        stress_pattern = self._get_stress_pattern(word)

        # Analyze each syllable
        results = []
        for i, syll in enumerate(syllables):
            # Determine stress (0=unstressed, 1=primary, 2=secondary)
            stress = stress_pattern[i] if i < len(stress_pattern) else 0

            # Determine weight (heavy=1, light=0)
            # Heavy syllables: contain long vowel, diphthong, or end in consonant
            weight = self._calculate_syllable_weight(syll)

            results.append({
                "syllable": syll,
                "stress": stress,
                "weight": weight,
            })

        return results

    def _get_stress_pattern(self, word: str) -> list[int]:
        """Get stress pattern for a word from CMU pronouncing dictionary.

        Args:
            word: The word to look up.

        Returns:
            List of stress levels (0, 1, or 2) for each syllable.
        """
        # Get all pronunciations for this word
        phones_list = pronouncing.phones_for_word(word)

        if not phones_list:
            # Word not in dictionary - use heuristic
            return self._heuristic_stress(word)

        # Use first pronunciation
        phones = phones_list[0]

        # Extract stress markers (0, 1, 2) from ARPAbet notation
        # Example: "HH AH0 L OW1" -> [0, 1]
        stress_pattern = []
        for phone in phones.split():
            if phone[-1] in "012":
                stress_pattern.append(int(phone[-1]))

        return stress_pattern

    def _heuristic_stress(self, word: str) -> list[int]:
        """Fallback heuristic for stress when word not in dictionary.

        Simple rules:
        - Single syllable: stressed (1)
        - Two syllables: first syllable stressed (1, 0)
        - Multi-syllable: penultimate syllable stressed

        Args:
            word: The word to analyze.

        Returns:
            List of stress levels.
        """
        syllables = self.dic.inserted(word).split("-")
        num_sylls = len(syllables)

        if num_sylls == 0:
            return []
        elif num_sylls == 1:
            return [1]  # Single syllable is stressed
        elif num_sylls == 2:
            return [1, 0]  # First syllable stressed in bisyllabic words
        else:
            # Penultimate stress is common in English
            stress = [0] * num_sylls
            stress[-2] = 1
            return stress

    def _calculate_syllable_weight(self, syllable: str) -> int:
        """Calculate syllable weight (heavy=1, light=0).

        Syllable weight rules:
        - Heavy: contains long vowel/diphthong OR ends in consonant
        - Light: short vowel in open syllable (ends in vowel)

        Args:
            syllable: The syllable to analyze.

        Returns:
            1 for heavy, 0 for light.
        """
        syllable = syllable.lower()

        # Vowel patterns
        vowels = "aeiou"
        long_vowels = ["ee", "oo", "ea", "ai", "ay", "ey", "ie", "oa", "ue"]
        diphthongs = ["ou", "oi", "oy", "ow", "au", "aw"]

        # Check for long vowels or diphthongs
        for pattern in long_vowels + diphthongs:
            if pattern in syllable:
                return 1  # Heavy

        # Check if syllable ends in consonant (closed syllable)
        if syllable and syllable[-1] not in vowels:
            return 1  # Heavy (closed syllable)

        # Check for multiple consonants (complex coda)
        consonant_cluster = re.search(r"[^aeiou]{2,}$", syllable)
        if consonant_cluster:
            return 1  # Heavy

        # Default: light syllable (open syllable with short vowel)
        return 0

    def analyze_text(self, text: str) -> list[dict[str, Any]]:
        """Analyze all words in a text.

        Args:
            text: The text to analyze.

        Returns:
            List of word analysis results, each containing syllable info.
        """
        # Split into words, remove punctuation
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        results = []
        for word in words:
            syllables = self.analyze_word(word)
            if syllables:
                results.append({
                    "word": word,
                    "syllables": syllables,
                })

        return results

    def extract_rhythm_pattern(self, text: str) -> list[str]:
        """Extract rhythm pattern as sequence of stress+weight codes.

        This matches the cadences format: each element is "stress+weight"
        for the first syllable of each word.

        Args:
            text: The text to analyze.

        Returns:
            List of rhythm codes (e.g., ["10", "01", "11"]).
        """
        word_analyses = self.analyze_text(text)

        rhythm_pattern = []
        for word_data in word_analyses:
            syllables = word_data["syllables"]
            if syllables:
                # Get first syllable's stress and weight
                first_syll = syllables[0]
                stress = first_syll["stress"]
                weight = first_syll["weight"]

                # Binary stress (0 or 1)
                stress_binary = 1 if stress > 0 else 0

                # Combine into pattern code
                code = f"{stress_binary}{weight}"
                rhythm_pattern.append(code)

        return rhythm_pattern
