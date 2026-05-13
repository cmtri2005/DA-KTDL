"""I/O helpers for Phase 5 supervised classification."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PropagationJob:
    clustering_representation: str
    model_slug: str
    algorithm: str
    param_name: str
    param_value: int
    job_dir: Path
    propagated_clusters_path: Path
    config_path: Path
    summary_path: Path
    phase3_score: float | None


def detect_phase1_root(requested: Path) -> Path:
    if requested.exists():
        return requested
    raise FileNotFoundError(f"Missing Phase 1 root: {requested}")


def detect_phase3_root(requested: Path) -> Path:
    if requested.exists():
        return requested
    raise FileNotFoundError(f"Missing Phase 3 root: {requested}")


def detect_phase4_root(requested: Path) -> Path:
    if requested.exists():
        return requested
    raise FileNotFoundError(f"Missing Phase 4 root: {requested}")


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


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


def load_phase1_records(phase1_root: Path) -> list[dict]:
    combined_path = phase1_root / "classify_combined.jsonl"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 1 combined classification file: {combined_path}"
        )

    records: list[dict] = []
    for row_index, row in enumerate(read_jsonl(combined_path)):
        records.append(
            {
                "row_index": int(row.get("row_index", row_index)),
                "id": str(row["id"]),
                "label": str(row["label"]),
                "primary_category": str(row.get("primary_category", "")),
                "n_triples": int(row.get("n_triples", 0)),
                "texts": {
                    "abstract": str(row.get("fmt_abstract", "")),
                    "triples": str(row.get("fmt_triples", "")),
                    "concatenate": str(row.get("fmt_concatenate", "")),
                    "hybrid": str(row.get("fmt_hybrid", "")),
                },
            }
        )
    if not records:
        raise ValueError(f"Empty Phase 1 classification file: {combined_path}")

    records.sort(key=lambda item: int(item["row_index"]))
    expected = list(range(len(records)))
    actual = [int(record["row_index"]) for record in records]
    if actual != expected:
        raise ValueError(f"Phase 1 row_index is not contiguous in {combined_path}")
    return records


def _parse_job_name(name: str) -> tuple[str, str, int]:
    if "_" not in name:
        raise ValueError(f"Unrecognized Phase 4 job directory name: {name}")
    algorithm, raw_param = name.split("_", 1)
    if raw_param.startswith("k"):
        return algorithm, "k", int(raw_param[1:])
    if raw_param.startswith("mcs"):
        return algorithm, "min_cluster_size", int(raw_param[3:])
    raise ValueError(f"Unrecognized Phase 4 job parameter in directory name: {name}")


def _load_phase3_score_lookup(phase3_root: Path) -> dict[tuple[str, str, str, str, int], float]:
    results_table_path = phase3_root / "results_table.csv"
    if not results_table_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 table: {results_table_path}")

    lookup: dict[tuple[str, str, str, str, int], float] = {}
    with open(results_table_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "cluster":
                continue
            key = (
                row["representation"],
                row["model_slug"],
                row["algorithm"],
                row["param_name"],
                int(row["param_value"]),
            )
            lookup[key] = float(row["score"])
    return lookup


def iter_propagation_jobs(
    *,
    phase4_root: Path,
    phase3_root: Path,
    clustering_representations: list[str],
    model_slugs: list[str],
) -> list[PropagationJob]:
    score_lookup = _load_phase3_score_lookup(phase3_root)
    jobs: list[PropagationJob] = []

    for records_path in sorted(phase4_root.rglob("propagated_clusters.jsonl")):
        job_dir = records_path.parent
        rel_parts = records_path.relative_to(phase4_root).parts
        if len(rel_parts) < 4:
            continue

        clustering_representation = rel_parts[0]
        model_slug = rel_parts[1]
        if clustering_representation not in clustering_representations:
            continue
        if model_slug not in model_slugs:
            continue

        algorithm, param_name, param_value = _parse_job_name(job_dir.name)
        phase3_score = score_lookup.get(
            (clustering_representation, model_slug, algorithm, param_name, param_value)
        )
        jobs.append(
            PropagationJob(
                clustering_representation=clustering_representation,
                model_slug=model_slug,
                algorithm=algorithm,
                param_name=param_name,
                param_value=param_value,
                job_dir=job_dir,
                propagated_clusters_path=records_path,
                config_path=job_dir / "propagation_config.json",
                summary_path=job_dir / "propagation_summary.json",
                phase3_score=phase3_score,
            )
        )

    if not jobs:
        raise FileNotFoundError(
            "No Phase 4 propagation jobs found for the requested representations/models."
        )
    return jobs


def select_best_propagation_jobs(
    jobs: list[PropagationJob],
) -> dict[tuple[str, str], PropagationJob]:
    selected: dict[tuple[str, str], PropagationJob] = {}
    for job in jobs:
        key = (job.clustering_representation, job.model_slug)
        current = selected.get(key)
        if current is None:
            selected[key] = job
            continue

        current_score = float("-inf") if current.phase3_score is None else current.phase3_score
        new_score = float("-inf") if job.phase3_score is None else job.phase3_score
        if new_score > current_score:
            selected[key] = job
            continue
        if new_score == current_score and str(job.job_dir) < str(current.job_dir):
            selected[key] = job
    return selected


def load_propagation_records(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty propagation file: {path}")
    return {str(row["id"]): row for row in rows}


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

