"""I/O helpers for Phase 3 clustering pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Phase2Job:
    split: str
    representation: str
    model_slug: str
    job_dir: Path
    embeddings_path: Path
    metadata_path: Path
    run_config_path: Path


def detect_phase2_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested

    candidates = (
        Path("outputs/phase2_embeddings"),
        Path("outputs/phase2_embeddings_eval"),
        Path("embeddings/outputs/phase2_embeddings_eval"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect Phase 2 root. Please pass --phase2_root.\n"
        "Tried: outputs/phase2_embeddings, outputs/phase2_embeddings_eval, "
        "embeddings/outputs/phase2_embeddings_eval"
    )


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_metadata_rows(path: Path) -> list[dict]:
    """Return metadata rows sorted by embedding row_index."""
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty metadata file: {path}")

    rows.sort(key=lambda item: int(item["row_index"]))
    return rows


def load_metadata_labels(path: Path) -> tuple[np.ndarray, list[str], list[str], list[dict]]:
    """Return (y_true, doc_ids, labels_as_str, rows) in row_index order."""
    rows = load_metadata_rows(path)
    doc_ids = [row["id"] for row in rows]
    labels = [row["label"] for row in rows]
    y_true = np.asarray(labels, dtype=object)
    return y_true, doc_ids, labels, rows


def iter_phase2_jobs(
    phase2_root: Path,
    split: str,
    representations: list[str],
    model_slugs: list[str] | None,
) -> list[Phase2Job]:
    jobs: list[Phase2Job] = []
    for representation in representations:
        rep_dir = phase2_root / split / representation
        if not rep_dir.exists():
            continue

        for model_dir in sorted(rep_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_slugs is not None and model_dir.name not in model_slugs:
                continue

            embeddings_path = model_dir / "embeddings.npy"
            metadata_path = model_dir / "metadata.jsonl"
            run_config_path = model_dir / "run_config.json"
            if not (embeddings_path.exists() and metadata_path.exists()):
                continue

            jobs.append(
                Phase2Job(
                    split=split,
                    representation=representation,
                    model_slug=model_dir.name,
                    job_dir=model_dir,
                    embeddings_path=embeddings_path,
                    metadata_path=metadata_path,
                    run_config_path=run_config_path,
                )
            )
    if not jobs:
        suffix = f"models={model_slugs}" if model_slugs else "models=all"
        raise FileNotFoundError(
            f"No Phase 2 jobs found under {phase2_root} for split={split}, reps={representations}, {suffix}."
        )
    return jobs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_dataclass(obj) -> dict:
    return _jsonable(asdict(obj))


def _jsonable(value):
    """Convert common non-JSON types (e.g., Path) recursively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
