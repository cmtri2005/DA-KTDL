"""Output writers for the arXiv Phase 1 pipeline."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from .constants import FORMATS, FMT_KEYS

log = logging.getLogger(__name__)


def save_outputs(records: List[Dict], split_name: str, out_dir: Path):
    """Save combined JSONL plus per-format JSONL and CSV views."""
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / f"{split_name}_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("  Saved %d records -> %s", len(records), combined_path)

    for fmt in FORMATS:
        key = FMT_KEYS[fmt]
        fmt_path_jsonl = out_dir / f"{split_name}_{fmt}.jsonl"
        fmt_path_csv = out_dir / f"{split_name}_{fmt}.csv"

        slim_records = [
            {
                "id": record["id"],
                "label": record["label"],
                "primary_category": record["primary_category"],
                "n_triples": record["n_triples"],
                "text": record[key],
            }
            for record in records
        ]

        with open(fmt_path_jsonl, "w", encoding="utf-8") as fh:
            for record in slim_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(fmt_path_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["id", "label", "primary_category", "n_triples", "text"],
            )
            writer.writeheader()
            writer.writerows(slim_records)

        log.info("    [%s] %s & %s", fmt, fmt_path_jsonl.name, fmt_path_csv.name)

