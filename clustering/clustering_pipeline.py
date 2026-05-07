"""Phase 3 - Unsupervised clustering on Phase 2 embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .cli import expand_requested, parse_args, parse_int_list
from .io_utils import (
    Phase2Job,
    detect_phase2_root,
    iter_phase2_jobs,
    load_metadata_labels,
    serialize_dataclass,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class ClusteringResult:
    split: str
    representation: str
    model_slug: str
    algorithm: str
    param_name: str
    param_value: int
    ari: float
    nmi: float
    silhouette: float
    noise_fraction: float | None
    score: float
    labels_path: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ("matplotlib", "numba", "hdbscan", "umap"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> float:
    unique = set(int(x) for x in np.unique(labels))
    if len(unique) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, labels, metric=metric))
    except Exception:
        return 0.0


def _score_kmeans_gmm(ari: float, nmi: float) -> float:
    return 0.5 * float(ari) + 0.5 * float(nmi)


def _score_hdbscan(ari: float, nmi: float, noise_fraction: float) -> float:
    return float(nmi) + 0.5 * float(ari) - 0.5 * float(noise_fraction)


def run_kmeans(
    X: np.ndarray,
    y_true: np.ndarray,
    split: str,
    representation: str,
    model_slug: str,
    out_dir: Path,
    k_min: int,
    k_max: int,
    seed: int,
    metric: str,
    overwrite: bool,
) -> list[ClusteringResult]:
    results: list[ClusteringResult] = []
    for k in range(k_min, k_max + 1):
        labels_path = out_dir / f"kmeans_k{k}_labels.npy"
        if labels_path.exists() and not overwrite:
            labels = np.load(labels_path)
        else:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=seed,
                algorithm="lloyd",
            )
            labels = model.fit_predict(X).astype(np.int32)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels)

        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels, average_method="arithmetic")
        sil = _safe_silhouette(X, labels, metric=metric)
        score = _score_kmeans_gmm(ari, nmi)
        results.append(
            ClusteringResult(
                split=split,
                representation=representation,
                model_slug=model_slug,
                algorithm="kmeans",
                param_name="k",
                param_value=k,
                ari=float(ari),
                nmi=float(nmi),
                silhouette=float(sil),
                noise_fraction=None,
                score=float(score),
                labels_path=str(labels_path),
            )
        )
    return results


def run_gmm(
    X: np.ndarray,
    y_true: np.ndarray,
    split: str,
    representation: str,
    model_slug: str,
    out_dir: Path,
    k_min: int,
    k_max: int,
    seed: int,
    metric: str,
    overwrite: bool,
) -> list[ClusteringResult]:
    results: list[ClusteringResult] = []
    for k in range(k_min, k_max + 1):
        labels_path = out_dir / f"gmm_k{k}_labels.npy"
        if labels_path.exists() and not overwrite:
            labels = np.load(labels_path)
        else:
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                tol=1e-3,
                reg_covar=1e-6,
                max_iter=200,
                n_init=5,
                init_params="kmeans",
                random_state=seed,
            )
            labels = model.fit_predict(X).astype(np.int32)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels)

        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels, average_method="arithmetic")
        sil = _safe_silhouette(X, labels, metric=metric)
        score = _score_kmeans_gmm(ari, nmi)
        results.append(
            ClusteringResult(
                split=split,
                representation=representation,
                model_slug=model_slug,
                algorithm="gmm",
                param_name="k",
                param_value=k,
                ari=float(ari),
                nmi=float(nmi),
                silhouette=float(sil),
                noise_fraction=None,
                score=float(score),
                labels_path=str(labels_path),
            )
        )
    return results


def run_hdbscan(
    X: np.ndarray,
    y_true: np.ndarray,
    split: str,
    representation: str,
    model_slug: str,
    out_dir: Path,
    min_cluster_sizes: list[int],
    metric: str,
    overwrite: bool,
) -> list[ClusteringResult]:
    try:
        import hdbscan
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'hdbscan'. Install via requirements.txt or pip install hdbscan."
        ) from exc

    results: list[ClusteringResult] = []
    for mcs in min_cluster_sizes:
        labels_path = out_dir / f"hdbscan_mcs{mcs}_labels.npy"
        if labels_path.exists() and not overwrite:
            labels = np.load(labels_path)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=None,
                metric=metric,
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(X).astype(np.int32)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(labels_path, labels)

        mask = labels != -1
        noise_fraction = float((~mask).mean())
        if mask.sum() >= 2 and len(set(int(x) for x in np.unique(labels[mask]))) >= 2:
            ari = adjusted_rand_score(y_true[mask], labels[mask])
            nmi = normalized_mutual_info_score(y_true[mask], labels[mask], average_method="arithmetic")
            sil = _safe_silhouette(X[mask], labels[mask], metric=metric)
        else:
            ari, nmi, sil = 0.0, 0.0, 0.0

        score = _score_hdbscan(ari, nmi, noise_fraction)
        results.append(
            ClusteringResult(
                split=split,
                representation=representation,
                model_slug=model_slug,
                algorithm="hdbscan",
                param_name="min_cluster_size",
                param_value=int(mcs),
                ari=float(ari),
                nmi=float(nmi),
                silhouette=float(sil),
                noise_fraction=float(noise_fraction),
                score=float(score),
                labels_path=str(labels_path),
            )
        )
    return results


def job_output_dir(output_root: Path, split: str, representation: str, model_slug: str) -> Path:
    return output_root / split / representation / model_slug


def best_by_group(results: list[ClusteringResult], key_fields: tuple[str, ...]) -> list[ClusteringResult]:
    groups: dict[tuple, ClusteringResult] = {}
    for item in results:
        key = tuple(getattr(item, field) for field in key_fields)
        current = groups.get(key)
        if current is None or item.score > current.score:
            groups[key] = item
    return list(groups.values())


def main() -> None:
    configure_logging()
    log = logging.getLogger(__name__)
    args = parse_args()

    phase2_root = detect_phase2_root(args.phase2_root)
    representations = expand_requested(args.representations, allowed=("abstract", "triples", "concatenate", "hybrid"))
    model_slugs = None if ("all" in [item.lower() for item in args.models]) else args.models
    min_cluster_sizes = parse_int_list(args.hdbscan_min_cluster_sizes)

    jobs = iter_phase2_jobs(
        phase2_root=phase2_root,
        split=args.split,
        representations=representations,
        model_slugs=model_slugs,
    )

    log.info("=" * 72)
    log.info("Phase 3 - Unsupervised Clustering")
    log.info("=" * 72)
    log.info("Phase 2 root   : %s", phase2_root)
    log.info("Output root    : %s", args.output_root)
    log.info("Split          : %s", args.split)
    log.info("Representations: %s", ", ".join(representations))
    log.info("Models         : %s", "all" if model_slugs is None else ", ".join(model_slugs))
    log.info("k sweep        : %d..%d", args.k_min, args.k_max)
    log.info("HDBSCAN mcs    : %s", ", ".join(str(x) for x in min_cluster_sizes))
    log.info("Metric         : %s", args.metric)
    log.info("Seed           : %d", args.seed)

    all_results: list[ClusteringResult] = []
    for job in jobs:
        X = np.load(job.embeddings_path).astype(np.float32)
        y_true, doc_ids, _labels = load_metadata_labels(job.metadata_path)

        out_dir = job_output_dir(args.output_root, job.split, job.representation, job.model_slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "phase2_job.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "phase2_root": str(phase2_root),
                "phase2_job": serialize_dataclass(job),
                "num_documents": int(X.shape[0]),
                "embedding_dim": int(X.shape[1]),
                "doc_ids_path": str(out_dir / "doc_ids.txt"),
            },
        )
        (out_dir / "doc_ids.txt").write_text("\n".join(doc_ids) + "\n", encoding="utf-8")

        log.info("[%s/%s/%s] X=%s", job.split, job.representation, job.model_slug, tuple(X.shape))
        all_results.extend(
            run_kmeans(
                X=X,
                y_true=y_true,
                split=job.split,
                representation=job.representation,
                model_slug=job.model_slug,
                out_dir=out_dir,
                k_min=args.k_min,
                k_max=args.k_max,
                seed=args.seed,
                metric=args.metric,
                overwrite=args.overwrite,
            )
        )
        all_results.extend(
            run_gmm(
                X=X,
                y_true=y_true,
                split=job.split,
                representation=job.representation,
                model_slug=job.model_slug,
                out_dir=out_dir,
                k_min=args.k_min,
                k_max=args.k_max,
                seed=args.seed,
                metric=args.metric,
                overwrite=args.overwrite,
            )
        )
        all_results.extend(
            run_hdbscan(
                X=X,
                y_true=y_true,
                split=job.split,
                representation=job.representation,
                model_slug=job.model_slug,
                out_dir=out_dir,
                min_cluster_sizes=min_cluster_sizes,
                metric=args.metric,
                overwrite=args.overwrite,
            )
        )

    # Persist full table.
    results_rows = [serialize_dataclass(item) for item in all_results]
    fieldnames = [
        "split",
        "representation",
        "model_slug",
        "algorithm",
        "param_name",
        "param_value",
        "ari",
        "nmi",
        "silhouette",
        "noise_fraction",
        "score",
        "labels_path",
    ]
    write_csv(args.output_root / "results_table.csv", results_rows, fieldnames=fieldnames)

    # Best per (split, representation, model_slug, algorithm).
    best_algo = best_by_group(all_results, ("split", "representation", "model_slug", "algorithm"))
    best_algo_rows = [serialize_dataclass(item) for item in best_algo]
    write_csv(args.output_root / "results_table_best_by_algorithm.csv", best_algo_rows, fieldnames=fieldnames)

    # Table-1-like: best per representation across models/algorithms (by score).
    best_repr = best_by_group(all_results, ("split", "representation"))
    best_repr_rows = [serialize_dataclass(item) for item in best_repr]
    write_csv(args.output_root / "results_table_best_by_representation.csv", best_repr_rows, fieldnames=fieldnames)

    log.info("Saved results -> %s", args.output_root)


if __name__ == "__main__":
    main()

