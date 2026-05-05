"""I/O helpers for the Phase 2 embedding pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


REPRESENTATION_KEYS = {
    "abstract": "fmt_abstract",
    "triples": "fmt_triples",
    "concatenate": "fmt_concatenate",
    "hybrid": "fmt_hybrid",
}


def slugify_model_name(name: str) -> str:
    """Convert a model name into a filesystem-safe directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    slug = slug.strip("-._")
    return slug or "model"


def read_jsonl(path: Path) -> Iterable[dict]:
    """Yield decoded JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_representation_source(phase1_output: Path, split: str, representation: str) -> tuple[Path, str]:
    """Return the source JSONL path and whether it is combined or slim."""
    combined_path = phase1_output / f"{split}_combined.jsonl"
    slim_path = phase1_output / f"{split}_{representation}.jsonl"
    if combined_path.exists():
        return combined_path, "combined"
    if slim_path.exists():
        return slim_path, "slim"
    raise FileNotFoundError(
        "Could not find Phase 1 source for "
        f"split={split}, representation={representation}. Checked: {combined_path} and {slim_path}"
    )


def load_phase1_records(
    phase1_output: Path,
    split: str,
    representation: str,
    limit: int | None = None,
) -> tuple[list[dict], Path, str]:
    """Load records for one split/representation from Phase 1 artifacts."""
    source_path, source_kind = resolve_representation_source(phase1_output, split, representation)
    rep_key = REPRESENTATION_KEYS[representation]

    records: list[dict] = []
    for idx, raw in enumerate(read_jsonl(source_path)):
        if source_kind == "combined":
            text = raw.get(rep_key, "") or ""
            record = {
                "id": raw.get("id"),
                "label": raw.get("label"),
                "primary_category": raw.get("primary_category"),
                "n_triples": raw.get("n_triples", 0),
                "text": text,
                "text_num_chars": len(text),
            }
        else:
            text = raw.get("text", "") or ""
            record = {
                "id": raw.get("id"),
                "label": raw.get("label"),
                "primary_category": raw.get("primary_category"),
                "n_triples": raw.get("n_triples", 0),
                "text": text,
                "text_num_chars": len(text),
            }

        if not record["id"] or not record["label"]:
            continue
        records.append(record)

        if limit is not None and (idx + 1) >= limit:
            break

    if not records:
        raise ValueError(
            f"No usable records found in {source_path} for split={split}, representation={representation}"
        )

    return records, source_path, source_kind


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

