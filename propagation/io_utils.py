"""I/O helpers for Phase 4 cluster propagation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EmbeddingJob:
    split: str
    representation: str
    model_slug: str
    job_dir: Path
    embeddings_path: Path
    metadata_path: Path
    run_config_path: Path


@dataclass(frozen=True)
class ClusterLabelJob:
    split: str
    representation: str
    model_slug: str
    algorithm: str
    param_name: str
    param_value: int
    labels_path: Path
    score: float | None


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
            if line:
                yield json.loads(line)


def load_metadata_rows(path: Path) -> list[dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty metadata file: {path}")
    rows.sort(key=lambda item: int(item["row_index"]))
    return rows


def load_embedding_job(
    phase2_root: Path,
    split: str,
    representation: str,
    model_slug: str,
) -> EmbeddingJob:
    job_dir = phase2_root / split / representation / model_slug
    embeddings_path = job_dir / "embeddings.npy"
    metadata_path = job_dir / "metadata.jsonl"
    run_config_path = job_dir / "run_config.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    return EmbeddingJob(
        split=split,
        representation=representation,
        model_slug=model_slug,
        job_dir=job_dir,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        run_config_path=run_config_path,
    )


def load_embeddings_and_metadata(job: EmbeddingJob) -> tuple[np.ndarray, list[dict], list[str]]:
    embeddings = np.load(job.embeddings_path).astype(np.float32)
    metadata_rows = load_metadata_rows(job.metadata_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for {job.embeddings_path}, got {embeddings.shape}")
    if embeddings.shape[0] != len(metadata_rows):
        raise ValueError(
            "Embedding/metadata row count mismatch for "
            f"{job.job_dir}: embeddings={embeddings.shape[0]}, metadata={len(metadata_rows)}"
        )
    expected_indices = list(range(len(metadata_rows)))
    actual_indices = [int(row["row_index"]) for row in metadata_rows]
    if actual_indices != expected_indices:
        raise ValueError(f"metadata row_index is not contiguous/aligned: {job.metadata_path}")
    if np.isnan(embeddings).any():
        raise ValueError(f"Embeddings contain NaN values: {job.embeddings_path}")
    if np.isinf(embeddings).any():
        raise ValueError(f"Embeddings contain Inf values: {job.embeddings_path}")
    doc_ids = [str(row["id"]) for row in metadata_rows]
    return embeddings, metadata_rows, doc_ids


def read_doc_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def read_best_cluster_jobs(
    phase3_root: Path,
    representations: list[str],
    model_slugs: list[str] | None,
    algorithms: list[str],
) -> list[ClusterLabelJob]:
    path = phase3_root / "results_table_best_by_algorithm.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 best-by-algorithm table: {path}. Run Phase 3 first."
        )

    jobs: list[ClusterLabelJob] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "cluster":
                continue
            if row["representation"] not in representations:
                continue
            if model_slugs is not None and row["model_slug"] not in model_slugs:
                continue
            if row["algorithm"] not in algorithms:
                continue

            labels_path = resolve_phase3_path(phase3_root, row["labels_path"])
            jobs.append(
                ClusterLabelJob(
                    split=row["split"],
                    representation=row["representation"],
                    model_slug=row["model_slug"],
                    algorithm=row["algorithm"],
                    param_name=row["param_name"],
                    param_value=int(row["param_value"]),
                    labels_path=labels_path,
                    score=float(row["score"]) if row.get("score") else None,
                )
            )

    if not jobs:
        suffix = f"models={model_slugs}" if model_slugs else "models=all"
        raise FileNotFoundError(
            "No Phase 3 cluster label jobs matched "
            f"representations={representations}, {suffix}, algorithms={algorithms}"
        )
    return jobs


def resolve_phase3_path(phase3_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if path.is_absolute():
        return path

    # Old outputs may store labels_path relative to the repo root. If that file no
    # longer exists, rebuild it under the requested Phase 3 root.
    parts = path.parts
    if "cluster" in parts:
        cluster_index = parts.index("cluster")
        rebuilt = phase3_root.joinpath(*parts[cluster_index:])
        if rebuilt.exists():
            return rebuilt
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_dataclass(obj) -> dict:
    return _jsonable(asdict(obj))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
