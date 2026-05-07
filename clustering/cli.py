"""CLI helpers for the Phase 3 clustering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

REPRESENTATIONS = ("abstract", "triples", "concatenate", "hybrid")
SPLITS = ("cluster",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: clustering on Phase 2 embeddings."
    )
    parser.add_argument(
        "--phase2_root",
        type=Path,
        default=None,
        help=(
            "Root directory containing Phase 2 artifacts. "
            "If omitted, the pipeline tries outputs/phase2_embeddings, "
            "outputs/phase2_embeddings_eval, then embeddings/outputs/phase2_embeddings_eval."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/phase3_clustering"),
        help="Root directory where Phase 3 artifacts will be written.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="cluster",
        help="Which Phase 2 split to cluster (default: cluster).",
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
        help="Model slugs (directory names) or 'all'. Example: sentence-transformers-all-mpnet-base-v2",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=3,
        help="Minimum k for KMeans/GMM sweep (inclusive).",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=12,
        help="Maximum k for KMeans/GMM sweep (inclusive).",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_sizes",
        type=str,
        default="5,10,20,30,50,100",
        help="Comma-separated values for HDBSCAN min_cluster_size sweep.",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "cosine"),
        default="euclidean",
        help="Distance metric used for silhouette and HDBSCAN (default: euclidean).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite per-job labels if they already exist.",
    )
    return parser.parse_args()


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_requested(raw_items: list[str], allowed: tuple[str, ...]) -> list[str]:
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
            f"Unsupported representations: {', '.join(invalid)}. Valid: {', '.join(allowed)}"
        )

    ordered_unique: list[str] = []
    for item in normalized:
        if item not in ordered_unique:
            ordered_unique.append(item)
    return ordered_unique


def parse_int_list(value: str) -> list[int]:
    items = _csv_list(value)
    parsed: list[int] = []
    for item in items:
        parsed.append(int(item))
    if not parsed:
        raise ValueError("Expected at least one integer.")
    return parsed
