"""Command-line interface for Phase 7."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


QUALITY_CLASSIFIER_REPRESENTATIONS = (
    "abstract",
    "triples",
    "concatenate",
    "hybrid",
    "quality_triples_top5",
    "quality_hybrid_top5",
    "quality_triples_top50",
    "quality_hybrid_top50",
    "quality_hybrid_banded",
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: triple-quality-aware hybrid classification extension."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Score triples and build quality representations.")
    build.add_argument("--phase1_root", type=Path, default=Path("outputs/phase1_data"))
    build.add_argument("--output_root", type=Path, default=Path("outputs/phase7_quality"))
    build.add_argument("--splits", nargs="+", default=["cluster", "classify"])
    build.add_argument("--top_k", type=int, default=5)
    build.add_argument("--top_fraction", type=float, default=0.5)
    build.add_argument("--high_threshold", type=float, default=0.65)
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split record limit for smoke tests.",
    )

    classify = subparsers.add_parser(
        "classify",
        help="Run Phase 7 classification ablations. This trains models and can take a long time.",
    )
    classify.add_argument("--quality_root", type=Path, default=Path("outputs/phase7_quality"))
    classify.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/phase7_classification"),
    )
    classify.add_argument(
        "--representations",
        nargs="+",
        default=[
            "abstract",
            "triples",
            "concatenate",
            "hybrid",
            "quality_hybrid_top5",
            "quality_hybrid_top50",
            "quality_hybrid_banded",
        ],
        help="Representations to train. Use 'all' for every Phase 7 option.",
    )
    classify.add_argument(
        "--models",
        nargs="+",
        default=["scibert"],
        help="One or more of: scibert, specter, minilm, all.",
    )
    classify.add_argument("--max_length", type=int, default=128)
    classify.add_argument("--train_size", type=float, default=0.8)
    classify.add_argument("--optuna_trials", type=int, default=3)
    classify.add_argument("--lr_min", type=float, default=1e-6)
    classify.add_argument("--lr_max", type=float, default=1e-4)
    classify.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32])
    classify.add_argument("--epochs_min", type=int, default=2)
    classify.add_argument("--epochs_max", type=int, default=7)
    classify.add_argument("--early_stopping_patience", type=int, default=2)
    classify.add_argument("--weight_decay", type=float, default=0.01)
    classify.add_argument("--num_workers", type=int, default=0)
    classify.add_argument("--device", default="auto")
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional example limit for tiny smoke runs.",
    )
    classify.add_argument("--overwrite", action="store_true")
    classify.add_argument("--fail_fast", action="store_true")

    report = subparsers.add_parser("report", help="Create Phase 7 markdown/CSV/figure report.")
    report.add_argument(
        "--classification_root",
        type=Path,
        default=Path("outputs/phase7_classification"),
    )
    report.add_argument("--quality_root", type=Path, default=Path("outputs/phase7_quality"))
    report.add_argument("--output_dir", type=Path, default=Path("reports/phase7"))

    return parser.parse_args()


def expand_requested(raw_items: list[str], allowed: tuple[str, ...], label: str) -> list[str]:
    items: list[str] = []
    for item in raw_items:
        if "," in item:
            items.extend(part.strip() for part in item.split(","))
        else:
            items.append(item.strip())
    normalized = [item.lower() for item in items if item]
    if not normalized or "all" in normalized:
        return list(allowed)
    invalid = [item for item in normalized if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported {label}: {', '.join(invalid)}. "
            f"Valid: {', '.join(allowed)}"
        )
    ordered: list[str] = []
    for item in normalized:
        if item not in ordered:
            ordered.append(item)
    return ordered


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.command == "build":
        from .build import build_quality_artifacts

        summary = build_quality_artifacts(
            phase1_root=args.phase1_root,
            output_root=args.output_root,
            splits=args.splits,
            top_k=args.top_k,
            top_fraction=args.top_fraction,
            high_threshold=args.high_threshold,
            limit=args.limit,
        )
        logging.info("Phase 7 build summary: %s", summary)
        return

    if args.command == "classify":
        from .classify import run_phase7_classification

        run_phase7_classification(args)
        return

    if args.command == "report":
        from .report import create_phase7_report

        create_phase7_report(
            classification_root=args.classification_root,
            quality_root=args.quality_root,
            output_dir=args.output_dir,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")
