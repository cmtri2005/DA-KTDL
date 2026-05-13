"""CLI helpers for the Phase 5 classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


REPRESENTATIONS = ("abstract", "triples", "concatenate", "hybrid")
MODEL_ALIASES = ("scibert", "specter", "minilm")
EXPERIMENT_PLANS = ("table2", "full_grid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5: supervised classification with propagated cluster signals "
            "from Phase 4."
        )
    )
    parser.add_argument(
        "--phase1_root",
        type=Path,
        default=Path("outputs/phase1_data"),
        help="Root directory containing Phase 1 classification texts.",
    )
    parser.add_argument(
        "--phase3_root",
        type=Path,
        default=Path("outputs/phase3_clustering"),
        help="Root directory containing Phase 3 clustering result tables.",
    )
    parser.add_argument(
        "--phase4_root",
        type=Path,
        default=Path("outputs/phase4_cluster_propagation"),
        help="Root directory containing Phase 4 propagation artifacts.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/phase5_classification"),
        help="Root directory where Phase 5 artifacts will be written.",
    )
    parser.add_argument(
        "--experiment_plan",
        choices=EXPERIMENT_PLANS,
        default="table2",
        help=(
            "Experiment plan to run. 'table2' follows the 16 paper-reported rows; "
            "'full_grid' runs the full cartesian product over requested modes/models."
        ),
    )
    parser.add_argument(
        "--clustering_representations",
        nargs="+",
        default=["all"],
        help="One or more of: abstract, triples, concatenate, hybrid, all.",
    )
    parser.add_argument(
        "--classifier_representations",
        nargs="+",
        default=["all"],
        help="One or more of: abstract, triples, concatenate, hybrid, all.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "One or more of: scibert, specter, minilm, all. Under experiment_plan=table2, "
            "this acts as an optional filter over the paper rows."
        ),
    )
    parser.add_argument(
        "--cluster_signal_mode",
        choices=("prefix", "suffix", "none"),
        default="prefix",
        help="How propagated cluster signals are injected into classifier text.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Tokenizer truncation/padding length.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Train split fraction for stratified train/validation split.",
    )
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=3,
        help="Number of Optuna trials per experiment.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Lower bound for log-uniform learning rate search.",
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=1e-4,
        help="Upper bound for log-uniform learning rate search.",
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Candidate batch sizes for Optuna search.",
    )
    parser.add_argument(
        "--epochs_min",
        type=int,
        default=2,
        help="Minimum number of epochs sampled by Optuna.",
    )
    parser.add_argument(
        "--epochs_max",
        type=int,
        default=7,
        help="Maximum number of epochs sampled by Optuna.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Stop when validation loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="PyTorch DataLoader num_workers.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment outputs.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Raise immediately on the first failed experiment.",
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
