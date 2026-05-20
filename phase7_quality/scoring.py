"""Rule-based triple quality scoring for Phase 7."""

from __future__ import annotations

import math
import re
from collections import Counter


RULE_SCORES = {
    "dobj": 1.00,
    "attr": 0.90,
    "prep->pobj": 0.65,
    "prep->pcomp": 0.60,
}

GENERIC_PRONOUNS = {
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "which",
    "we",
    "our",
    "ours",
    "they",
    "their",
}

GENERIC_RELATIONS = {
    "be",
    "have",
    "use",
    "do",
    "make",
    "show",
    "present",
    "study",
    "consider",
}


def dependency_rule_score(rule_tag: str | None) -> float:
    return float(RULE_SCORES.get(str(rule_tag or "").strip().lower(), 0.55))


def token_count(text: str | None) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def _single_phrase_length_score(text: str | None) -> float:
    count = token_count(text)
    if 2 <= count <= 8:
        return 1.00
    if count == 1:
        return 0.55
    if 9 <= count <= 12:
        return 0.80
    if 13 <= count <= 20:
        return 0.45
    if count > 20:
        return 0.25
    return 0.25


def _has_generic_pronoun(text: str | None) -> bool:
    if not text:
        return False
    tokens = {token.lower() for token in re.findall(r"\b\w+\b", str(text))}
    return bool(tokens & GENERIC_PRONOUNS)


def phrase_length_score(subject: str | None, object_: str | None) -> float:
    score = (
        _single_phrase_length_score(subject) + _single_phrase_length_score(object_)
    ) / 2.0
    if _has_generic_pronoun(subject) or _has_generic_pronoun(object_):
        score = min(score, 0.70)
    return float(score)


def relation_frequency_score(
    relation: str | None,
    relation_counts: Counter[str] | dict[str, int],
) -> float:
    rel = str(relation or "").strip().lower()
    if not rel:
        return 0.25

    max_freq = max(relation_counts.values()) if relation_counts else 0
    if max_freq <= 0:
        score = 1.0
    else:
        freq = int(relation_counts.get(rel, 0))
        freq_norm = math.log1p(freq) / math.log1p(max_freq)
        score = max(0.25, min(1.0, 1.0 - freq_norm))

    if rel in GENERIC_RELATIONS:
        score = min(score, 0.45)
    return float(score)


def source_sentence_score(source_sentence: str | None) -> float:
    sentence = str(source_sentence or "")
    count = token_count(sentence)
    if count <= 40:
        score = 1.00
    elif count <= 70:
        score = 0.75
    elif count <= 100:
        score = 0.50
    else:
        score = 0.25

    if _looks_formula_noisy(sentence):
        score = max(0.10, score - 0.15)
    return float(score)


def _looks_formula_noisy(text: str) -> bool:
    if not text:
        return False
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    symbolic = sum(1 for char in chars if not char.isalnum() and char not in ".,;:-()[]")
    symbol_ratio = symbolic / len(chars)
    repeated_math_tokens = len(re.findall(r"[=<>^_{}\\]|\\[a-zA-Z]+", text))
    return symbol_ratio > 0.10 or repeated_math_tokens >= 4


def score_triple(
    triple: dict,
    relation_counts: Counter[str] | dict[str, int],
) -> dict:
    dep_score = dependency_rule_score(triple.get("rule_tag"))
    phrase_score = phrase_length_score(triple.get("subj"), triple.get("obj"))
    rel_score = relation_frequency_score(triple.get("rel"), relation_counts)
    sentence_score = source_sentence_score(triple.get("source_sentence"))
    quality_score = (
        0.35 * dep_score
        + 0.25 * phrase_score
        + 0.25 * rel_score
        + 0.15 * sentence_score
    )

    enriched = dict(triple)
    enriched.update(
        {
            "dependency_rule_score": round(dep_score, 6),
            "phrase_length_score": round(phrase_score, 6),
            "relation_frequency_score": round(rel_score, 6),
            "source_sentence_score": round(sentence_score, 6),
            "quality_score": round(float(quality_score), 6),
        }
    )
    return enriched


def linearize_triples(triples: list[dict]) -> str:
    sentences: list[str] = []
    for triple in triples:
        subject = str(triple.get("subj", "")).strip()
        relation = str(triple.get("rel", "")).strip()
        object_ = str(triple.get("obj", "")).strip()
        if not subject or not relation or not object_:
            continue
        sentences.append(f"{subject.capitalize()} {relation} {object_}.")
    return " ".join(sentences)
