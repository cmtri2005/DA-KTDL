"""Unit tests for Phase 7 triple quality scoring."""

from __future__ import annotations

import unittest
from collections import Counter

from phase7_quality.scoring import (
    dependency_rule_score,
    phrase_length_score,
    relation_frequency_score,
    score_triple,
    source_sentence_score,
)


class Phase7QualityScoringTest(unittest.TestCase):
    def test_dependency_rule_score(self):
        self.assertEqual(dependency_rule_score("dobj"), 1.0)
        self.assertEqual(dependency_rule_score("attr"), 0.9)
        self.assertEqual(dependency_rule_score("prep->pobj"), 0.65)
        self.assertEqual(dependency_rule_score("prep->pcomp"), 0.6)
        self.assertEqual(dependency_rule_score("unknown"), 0.55)

    def test_phrase_length_score_penalizes_short_and_long_phrases(self):
        good = phrase_length_score("neural language models", "contextual representations")
        one_token = phrase_length_score("we", "representations")
        long_phrase = phrase_length_score(
            " ".join(["very"] * 14),
            "contextual representations",
        )
        self.assertGreater(good, one_token)
        self.assertGreater(good, long_phrase)
        self.assertLessEqual(one_token, 0.70)

    def test_relation_frequency_score_penalizes_common_and_generic_relations(self):
        counts = Counter({"use": 100, "calibrate": 2, "discover": 1})
        self.assertLess(relation_frequency_score("use", counts), relation_frequency_score("calibrate", counts))
        self.assertLessEqual(relation_frequency_score("use", counts), 0.45)

    def test_source_sentence_score_penalizes_long_and_noisy_sentences(self):
        short = source_sentence_score("we propose a method for clustering documents.")
        long = source_sentence_score(" ".join(["token"] * 120))
        noisy = source_sentence_score("x = y ^ 2 + z_{i} \\alpha \\beta \\gamma")
        self.assertGreater(short, long)
        self.assertLess(noisy, short)

    def test_quality_score_is_bounded(self):
        triple = {
            "subj": "we",
            "rel": "study",
            "obj": "the system",
            "source_sentence": "we study the system.",
            "rule_tag": "dobj",
        }
        scored = score_triple(triple, Counter({"study": 10}))
        self.assertGreaterEqual(scored["quality_score"], 0.0)
        self.assertLessEqual(scored["quality_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
