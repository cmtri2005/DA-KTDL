"""CLI helpers for the Phase 2 embedding pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


REPRESENTATIONS = ("abstract", "triples", "concatenate", "hybrid")
SPLITS = ("cluster", "classify")


def _csv_or_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: generate normalized embeddings from Phase 1 outputs."
    )
    parser.add_argument(
        "--phase1_output",
        type=Path,
        default=Path("output_triples"),
        help="Directory containing Phase 1 JSONL artifacts.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/phase2_embeddings"),
        help="Root directory where Phase 2 artifacts will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        help="One or more of: cluster, classify, all.",
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
        help=(
            "Model aliases or HuggingFace names. Supported aliases: "
            "minilm, mpnet, specter, scibert, all."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used during embedding generation.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Tokenizer max_length for HuggingFace backbones and sentence encoders.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "cls"),
        default="mean",
        help="Pooling strategy for SciBERT and SPECTER transformer outputs.",
    )
    parser.add_argument("--device", default="auto", help="Computation device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing embeddings job directory.",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="L2-normalize the final embeddings (default: enabled).",
    )
    parser.add_argument(
        "--no_normalize",
        dest="normalize",
        action="store_false",
        help="Disable final L2 normalization.",
    )
    return parser.parse_args()


def expand_requested(raw_items: list[str], allowed: tuple[str, ...], label: str) -> list[str]:
    items: list[str] = []
    for item in raw_items:
        pieces = _csv_or_list(item)
        items.extend(pieces or [item])

    normalized = [item.strip().lower() for item in items if item.strip()]
    if not normalized or "all" in normalized:
        return list(allowed)

    invalid = [item for item in normalized if item not in allowed]
    if invalid:
        valid_str = ", ".join(allowed)
        bad_str = ", ".join(invalid)
        raise ValueError(f"Unsupported {label}: {bad_str}. Valid values: {valid_str}")

    ordered_unique: list[str] = []
    for item in normalized:
        if item not in ordered_unique:
            ordered_unique.append(item)
    return ordered_unique
