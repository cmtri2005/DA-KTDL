"""Loading and filtering helpers for the arXiv Phase 1 pipeline."""

import json
import logging
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)


def get_primary_category(categories: str) -> Optional[str]:
    """Return the first arXiv category token."""
    tokens = categories.split()
    if not tokens:
        return None
    return tokens[0].strip()


def get_top_level_label(primary_category: str) -> Optional[str]:
    """Map an arXiv category like 'cs.AI' to its top-level archive label."""
    if not primary_category:
        return None
    return primary_category.split(".")[0].lower()


def extract_update_year(update_date: str) -> Optional[int]:
    """Extract the year from an arXiv update_date string."""
    match = re.match(r"^(\d{4})-", update_date or "")
    if not match:
        return None
    return int(match.group(1))


def clean_abstract(text: str) -> str:
    """Lowercase, flatten line breaks, strip simple LaTeX noise, normalize whitespace."""
    text = text.replace("\n", " ")
    text = re.sub(r"\$\$?.+?\$\$?", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_arxiv(
    path: str,
    n_cluster: int,
    n_classify: int,
    seed: int,
    year_from: Optional[int],
    year_to: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    """Stream arXiv JSONL, filter eligible records, deduplicate, and split."""
    rng = random.Random(seed)
    needed = n_cluster + n_classify
    candidates: List[Dict] = []

    log.info("Streaming %s ...", path)
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Reading JSONL", unit=" lines"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            abstract = record.get("abstract", "") or ""
            categories = record.get("categories", "") or ""
            update_date = record.get("update_date", "") or ""
            if not abstract.strip() or not categories.strip() or not update_date.strip():
                continue

            primary_category = get_primary_category(categories)
            label = get_top_level_label(primary_category or "")
            if primary_category is None or label is None:
                continue

            update_year = extract_update_year(update_date)
            if year_from is not None and (update_year is None or update_year < year_from):
                continue
            if year_to is not None and (update_year is None or update_year > year_to):
                continue

            cleaned_abstract = clean_abstract(abstract)
            if not cleaned_abstract:
                continue

            candidates.append(
                {
                    "id": record.get("id"),
                    "title": record.get("title", "").replace("\n", " ").strip(),
                    "abstract": cleaned_abstract,
                    "categories": categories,
                    "primary_category": primary_category,
                    "label": label,
                    "update_date": update_date,
                }
            )

    log.info("Loaded %d eligible docs before deduplication", len(candidates))

    seen = set()
    deduped: List[Dict] = []
    for doc in candidates:
        if doc["id"] in seen:
            continue
        seen.add(doc["id"])
        deduped.append(doc)

    if len(deduped) < needed:
        raise ValueError(
            f"Not enough eligible documents after filtering: need {needed}, found {len(deduped)}"
        )

    rng.shuffle(deduped)
    cluster_set = deduped[:n_cluster]
    classify_set = deduped[n_cluster : n_cluster + n_classify]

    label_counts = Counter(doc["label"] for doc in deduped)
    log.info("Deduplicated to %d docs across %d labels", len(deduped), len(label_counts))
    log.info("  Clustering split     : %d docs", len(cluster_set))
    log.info("  Classification split : %d docs", len(classify_set))
    return cluster_set, classify_set

