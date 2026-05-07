"""CLI helpers for the Phase 4 cluster propagation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


REPRESENTATIONS = ("abstract", "triples", "concatenate", "hybrid")
CLUSTER_ALGORITHMS = ("kmeans", "gmm", "hdbscan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: propagate Phase 3 cluster labels to classification documents."
    )
    parser.add_argument(
        "--phase2_root",
        type=Path,
        default=None,
        help=(
            "Root directory containing Phase 2 artifacts. If omitted, the pipeline "
            "tries outputs/phase2_embeddings, outputs/phase2_embeddings_eval, then "
            "embeddings/outputs/phase2_embeddings_eval."
        ),
    )
    parser.add_argument(
        "--phase3_root",
        type=Path,
        default=Path("outputs/phase3_clustering"),
        help="Root directory containing Phase 3 clustering artifacts.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/phase4_cluster_propagation"),
        help="Root directory where Phase 4 artifacts will be written.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["all"],
        help="One or more of: abstract, triples, concatenate, hybrid, all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model slugs (directory names) or 'all'.",
    )
    parser.add_argument(
        "--cluster_algorithms",
        nargs="+",
        default=["kmeans", "gmm"],
        help="One or more of: kmeans, gmm, hdbscan, all. Default: kmeans gmm.",
    )
    parser.add_argument(
        "--cluster_label_file",
        type=Path,
        default=None,
        help=(
            "Optional explicit Phase 3 labels .npy file. Use with a single representation, "
            "single model, and single cluster algorithm."
        ),
    )
    parser.add_argument(
        "--neighbor_k",
        type=int,
        default=5,
        help="Number of nearest cluster documents used for propagation.",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Nearest-neighbor distance metric.",
    )
    parser.add_argument(
        "--assignment_rule",
        choices=("majority_vote", "distance_weighted_vote"),
        default="majority_vote",
        help="How top-k neighbor cluster ids are converted into one propagated cluster id.",
    )
    parser.add_argument(
        "--handle_noise",
        choices=("ignore", "keep"),
        default="ignore",
        help="How to handle HDBSCAN noise label -1 during voting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed recorded in propagation_config.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing propagation artifacts.",
    )
    return parser.parse_args()


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_requested(raw_items: list[str], allowed: tuple[str, ...], label: str) -> list[str]:
    items: list[str] = []
    for item in raw_items:
        if "," in item:
            items.extend(_csv_list(item))
        else:
            items.append(item)

    normalized = [item.strip().lower() for item in items if item.strip()]
    if not normalized or "all" in normalized:
        return list(allowed)

    invalid = [item for item in normalized if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported {label}: {', '.join(invalid)}. Valid: {', '.join(allowed)}"
        )

    ordered_unique: list[str] = []
    for item in normalized:
        if item not in ordered_unique:
            ordered_unique.append(item)
    return ordered_unique
