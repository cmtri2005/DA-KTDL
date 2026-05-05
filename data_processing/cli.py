"""Command-line interface for the arXiv triples Phase 1 pipeline."""

import argparse
from pathlib import Path


DEFAULT_INPUT = Path("dataset/arxiv-metadata-oai-snapshot.json")
DEFAULT_OUTPUT = Path("output_triples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="arXiv -> triples extraction and 4-format data pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the arXiv metadata JSONL file (default: ./arxiv-metadata-oai-snapshot.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for processed artifacts (default: ./output_triples)",
    )
    parser.add_argument("--n_cluster", type=int, default=5000)
    parser.add_argument("--n_classify", type=int, default=10000)
    parser.add_argument(
        "--spacy_model",
        default="en_core_sci_md",
        help="Scientific spaCy model used for triple extraction",
    )
    parser.add_argument(
        "--year_from",
        type=int,
        default=None,
        help="Optional lower bound on update year",
    )
    parser.add_argument(
        "--year_to",
        type=int,
        default=None,
        help="Optional upper bound on update year",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
