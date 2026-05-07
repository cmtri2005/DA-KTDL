"""Phase 4 - propagate clustering signals to the classification split."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .cli import CLUSTER_ALGORITHMS, REPRESENTATIONS, expand_requested, parse_args
from .io_utils import (
    ClusterLabelJob,
    detect_phase2_root,
    load_embedding_job,
    load_embeddings_and_metadata,
    read_best_cluster_jobs,
    read_doc_ids,
    serialize_dataclass,
    write_csv,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class PropagationJob:
    representation: str
    model_slug: str
    algorithm: str
    param_name: str
    param_value: int
    cluster_labels_path: Path
    score: float | None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _cluster_config_slug(job: PropagationJob) -> str:
    return f"{job.algorithm}_{job.param_name}{job.param_value}"


def propagation_output_dir(output_root: Path, job: PropagationJob) -> Path:
    return output_root / job.representation / job.model_slug / _cluster_config_slug(job)


def validate_phase3_alignment(
    phase3_root: Path,
    job: PropagationJob,
    cluster_doc_ids: list[str],
    cluster_labels: np.ndarray,
) -> None:
    if cluster_labels.shape[0] != len(cluster_doc_ids):
        raise ValueError(
            "Cluster label count does not match cluster metadata: "
            f"labels={cluster_labels.shape[0]}, metadata={len(cluster_doc_ids)}, "
            f"path={job.cluster_labels_path}"
        )

    doc_ids_path = phase3_root / "cluster" / job.representation / job.model_slug / "doc_ids.txt"
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 doc_ids.txt for alignment check: {doc_ids_path}")
    phase3_doc_ids = read_doc_ids(doc_ids_path)
    if phase3_doc_ids != cluster_doc_ids:
        raise ValueError(
            "Phase 3 doc_ids.txt does not match Phase 2 cluster metadata order for "
            f"{job.representation}/{job.model_slug}."
        )


def _filtered_votes(
    neighbor_cluster_ids: np.ndarray,
    neighbor_distances: np.ndarray,
    handle_noise: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = neighbor_cluster_ids.astype(np.int32)
    distances = neighbor_distances.astype(np.float64)
    if handle_noise == "keep":
        return labels, distances

    keep = labels != -1
    if keep.any():
        return labels[keep], distances[keep]
    return labels, distances


def majority_vote(
    neighbor_cluster_ids: np.ndarray,
    neighbor_distances: np.ndarray,
    handle_noise: str,
) -> tuple[int, float]:
    labels, _distances = _filtered_votes(
        neighbor_cluster_ids=neighbor_cluster_ids,
        neighbor_distances=neighbor_distances,
        handle_noise=handle_noise,
    )
    counts = Counter(int(label) for label in labels)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}

    # Tie-break by the closest original neighbor among tied labels.
    chosen = None
    for label in neighbor_cluster_ids.astype(np.int32):
        if int(label) in tied:
            chosen = int(label)
            break
    if chosen is None:
        chosen = int(labels[0])

    confidence = float(max_count / len(labels)) if len(labels) else 0.0
    return chosen, confidence


def distance_weighted_vote(
    neighbor_cluster_ids: np.ndarray,
    neighbor_distances: np.ndarray,
    handle_noise: str,
) -> tuple[int, float]:
    labels, distances = _filtered_votes(
        neighbor_cluster_ids=neighbor_cluster_ids,
        neighbor_distances=neighbor_distances,
        handle_noise=handle_noise,
    )
    weights_by_label: dict[int, float] = defaultdict(float)
    for label, distance in zip(labels, distances):
        weights_by_label[int(label)] += 1.0 / (float(distance) + 1e-12)

    chosen, chosen_weight = max(weights_by_label.items(), key=lambda item: item[1])
    total_weight = sum(weights_by_label.values())
    confidence = float(chosen_weight / total_weight) if total_weight else 0.0
    return int(chosen), confidence


def assign_cluster(
    neighbor_cluster_ids: np.ndarray,
    neighbor_distances: np.ndarray,
    assignment_rule: str,
    handle_noise: str,
) -> tuple[int, float]:
    if assignment_rule == "majority_vote":
        return majority_vote(neighbor_cluster_ids, neighbor_distances, handle_noise)
    if assignment_rule == "distance_weighted_vote":
        return distance_weighted_vote(neighbor_cluster_ids, neighbor_distances, handle_noise)
    raise ValueError(f"Unsupported assignment_rule: {assignment_rule}")


def build_summary(rows: list[dict], cluster_labels: np.ndarray) -> dict:
    propagated_counts = Counter(row["propagated_cluster_id"] for row in rows)
    n_docs = len(rows)
    noise_count = int(propagated_counts.get(-1, 0))
    confidences = [float(row["propagation_confidence"]) for row in rows]
    nearest_distances = [float(row["nearest_neighbor_distance"]) for row in rows]
    topk_mean_distances = [float(row["mean_neighbor_distance"]) for row in rows]
    largest_cluster_count = max(propagated_counts.values()) if propagated_counts else 0
    source_counts = Counter(int(label) for label in cluster_labels)
    return {
        "n_classify_docs": n_docs,
        "n_source_cluster_docs": int(cluster_labels.shape[0]),
        "n_unique_source_clusters": len(source_counts),
        "n_unique_propagated_clusters": len(propagated_counts),
        "cluster_distribution": dict(sorted(propagated_counts.items())),
        "source_cluster_distribution": dict(sorted(source_counts.items())),
        "largest_cluster_fraction": float(largest_cluster_count / n_docs) if n_docs else 0.0,
        "noise_fraction": float(noise_count / n_docs) if n_docs else 0.0,
        "mean_neighbor_distance": float(np.mean(topk_mean_distances)) if topk_mean_distances else 0.0,
        "mean_nearest_neighbor_distance": float(np.mean(nearest_distances)) if nearest_distances else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def run_propagation_job(
    phase2_root: Path,
    phase3_root: Path,
    output_root: Path,
    job: PropagationJob,
    neighbor_k: int,
    metric: str,
    assignment_rule: str,
    handle_noise: str,
    seed: int,
    overwrite: bool,
) -> None:
    out_dir = propagation_output_dir(output_root, job)
    output_jsonl = out_dir / "propagated_clusters.jsonl"
    if output_jsonl.exists() and not overwrite:
        logging.getLogger(__name__).info("Skipping existing propagation output: %s", out_dir)
        return

    cluster_embedding_job = load_embedding_job(
        phase2_root=phase2_root,
        split="cluster",
        representation=job.representation,
        model_slug=job.model_slug,
    )
    classify_embedding_job = load_embedding_job(
        phase2_root=phase2_root,
        split="classify",
        representation=job.representation,
        model_slug=job.model_slug,
    )
    cluster_X, cluster_metadata, cluster_doc_ids = load_embeddings_and_metadata(cluster_embedding_job)
    classify_X, classify_metadata, _classify_doc_ids = load_embeddings_and_metadata(classify_embedding_job)

    cluster_labels = np.load(job.cluster_labels_path).astype(np.int32)
    validate_phase3_alignment(
        phase3_root=phase3_root,
        job=job,
        cluster_doc_ids=cluster_doc_ids,
        cluster_labels=cluster_labels,
    )

    n_neighbors = min(int(neighbor_k), int(cluster_X.shape[0]))
    if n_neighbors <= 0:
        raise ValueError("--neighbor_k must be positive.")

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(cluster_X)
    distances, indices = nn.kneighbors(classify_X)

    records: list[dict] = []
    csv_rows: list[dict] = []
    for row_idx, (metadata, neighbor_indices, neighbor_distances) in enumerate(
        zip(classify_metadata, indices, distances)
    ):
        neighbor_cluster_ids = cluster_labels[neighbor_indices]
        propagated_cluster_id, confidence = assign_cluster(
            neighbor_cluster_ids=neighbor_cluster_ids,
            neighbor_distances=neighbor_distances,
            assignment_rule=assignment_rule,
            handle_noise=handle_noise,
        )
        neighbor_ids = [cluster_doc_ids[int(index)] for index in neighbor_indices]
        neighbor_labels = [
            str(cluster_metadata[int(index)]["label"]) for index in neighbor_indices
        ]
        record = {
            "row_index": row_idx,
            "id": metadata["id"],
            "label": metadata["label"],
            "primary_category": metadata.get("primary_category", ""),
            "n_triples": metadata.get("n_triples", ""),
            "representation": job.representation,
            "embedding_model": job.model_slug,
            "source_cluster_algorithm": job.algorithm,
            "source_cluster_param_name": job.param_name,
            "source_cluster_param_value": job.param_value,
            "propagated_cluster_id": int(propagated_cluster_id),
            "propagation_confidence": float(confidence),
            "neighbor_ids": neighbor_ids,
            "neighbor_distances": [float(value) for value in neighbor_distances],
            "neighbor_cluster_ids": [int(value) for value in neighbor_cluster_ids],
            "neighbor_labels": neighbor_labels,
            "nearest_neighbor_distance": float(neighbor_distances[0]),
            "mean_neighbor_distance": float(np.mean(neighbor_distances)),
        }
        records.append(record)
        csv_rows.append(
            {
                **{key: record[key] for key in record if not isinstance(record[key], list)},
                "neighbor_ids": "|".join(neighbor_ids),
                "neighbor_distances": "|".join(f"{float(value):.8f}" for value in neighbor_distances),
                "neighbor_cluster_ids": "|".join(str(int(value)) for value in neighbor_cluster_ids),
                "neighbor_labels": "|".join(neighbor_labels),
            }
        )

    summary = {
        **build_summary(records, cluster_labels),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_root": str(phase2_root),
        "phase3_root": str(phase3_root),
        "output_root": str(output_root),
        "representation": job.representation,
        "model_slug": job.model_slug,
        "cluster_algorithm": job.algorithm,
        "cluster_param_name": job.param_name,
        "cluster_param_value": job.param_value,
        "cluster_labels_path": str(job.cluster_labels_path),
        "neighbor_k": n_neighbors,
        "metric": metric,
        "assignment_rule": assignment_rule,
        "handle_noise": handle_noise,
        "seed": seed,
    }

    config = {
        "created_at_utc": summary["created_at_utc"],
        "phase2_root": str(phase2_root),
        "phase3_root": str(phase3_root),
        "cluster_embedding_job": serialize_dataclass(cluster_embedding_job),
        "classify_embedding_job": serialize_dataclass(classify_embedding_job),
        "propagation_job": serialize_dataclass(job),
        "neighbor_k": n_neighbors,
        "metric": metric,
        "assignment_rule": assignment_rule,
        "handle_noise": handle_noise,
        "seed": seed,
    }

    write_jsonl(output_jsonl, records)
    write_csv(
        out_dir / "propagated_clusters.csv",
        csv_rows,
        fieldnames=[
            "row_index",
            "id",
            "label",
            "primary_category",
            "n_triples",
            "representation",
            "embedding_model",
            "source_cluster_algorithm",
            "source_cluster_param_name",
            "source_cluster_param_value",
            "propagated_cluster_id",
            "propagation_confidence",
            "nearest_neighbor_distance",
            "mean_neighbor_distance",
            "neighbor_ids",
            "neighbor_distances",
            "neighbor_cluster_ids",
            "neighbor_labels",
        ],
    )
    write_json(out_dir / "propagation_config.json", config)
    write_json(out_dir / "propagation_summary.json", summary)


def explicit_label_job(args, representations: list[str], model_slugs: list[str] | None, algorithms: list[str]) -> PropagationJob:
    if len(representations) != 1:
        raise ValueError("--cluster_label_file requires exactly one representation.")
    if model_slugs is None or len(model_slugs) != 1:
        raise ValueError("--cluster_label_file requires exactly one model slug.")
    if len(algorithms) != 1:
        raise ValueError("--cluster_label_file requires exactly one cluster algorithm.")

    label_path = args.cluster_label_file
    if label_path is None or not label_path.exists():
        raise FileNotFoundError(f"Missing explicit cluster label file: {label_path}")

    stem = label_path.stem
    param_name = "explicit"
    param_value = 0
    for prefix, name in (("kmeans_k", "k"), ("gmm_k", "k"), ("hdbscan_mcs", "min_cluster_size")):
        if stem.startswith(prefix):
            param_name = name
            param_value = int(stem[len(prefix):].replace("_labels", ""))
            break

    return PropagationJob(
        representation=representations[0],
        model_slug=model_slugs[0],
        algorithm=algorithms[0],
        param_name=param_name,
        param_value=param_value,
        cluster_labels_path=label_path,
        score=None,
    )


def main() -> None:
    configure_logging()
    log = logging.getLogger(__name__)
    args = parse_args()

    phase2_root = detect_phase2_root(args.phase2_root)
    representations = expand_requested(args.representations, REPRESENTATIONS, "representations")
    algorithms = expand_requested(args.cluster_algorithms, CLUSTER_ALGORITHMS, "cluster algorithms")
    model_slugs = None if any(item.lower() == "all" for item in args.models) else args.models

    if args.cluster_label_file is not None:
        jobs = [explicit_label_job(args, representations, model_slugs, algorithms)]
    else:
        cluster_jobs: list[ClusterLabelJob] = read_best_cluster_jobs(
            phase3_root=args.phase3_root,
            representations=representations,
            model_slugs=model_slugs,
            algorithms=algorithms,
        )
        jobs = [
            PropagationJob(
                representation=item.representation,
                model_slug=item.model_slug,
                algorithm=item.algorithm,
                param_name=item.param_name,
                param_value=item.param_value,
                cluster_labels_path=item.labels_path,
                score=item.score,
            )
            for item in cluster_jobs
        ]

    log.info("=" * 72)
    log.info("Phase 4 - Cluster Propagation")
    log.info("=" * 72)
    log.info("Phase 2 root      : %s", phase2_root)
    log.info("Phase 3 root      : %s", args.phase3_root)
    log.info("Output root       : %s", args.output_root)
    log.info("Representations   : %s", ", ".join(representations))
    log.info("Models            : %s", "all" if model_slugs is None else ", ".join(model_slugs))
    log.info("Algorithms        : %s", ", ".join(algorithms))
    log.info("neighbor_k        : %d", args.neighbor_k)
    log.info("metric            : %s", args.metric)
    log.info("assignment_rule   : %s", args.assignment_rule)
    log.info("handle_noise      : %s", args.handle_noise)
    log.info("Jobs              : %d", len(jobs))

    for job in jobs:
        log.info(
            "[%s/%s/%s %s=%s] propagating ...",
            job.representation,
            job.model_slug,
            job.algorithm,
            job.param_name,
            job.param_value,
        )
        run_propagation_job(
            phase2_root=phase2_root,
            phase3_root=args.phase3_root,
            output_root=args.output_root,
            job=job,
            neighbor_k=args.neighbor_k,
            metric=args.metric,
            assignment_rule=args.assignment_rule,
            handle_noise=args.handle_noise,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    log.info("Saved propagation artifacts -> %s", args.output_root)


if __name__ == "__main__":
    main()
