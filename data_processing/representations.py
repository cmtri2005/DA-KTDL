"""Representation builders for the arXiv Phase 1 pipeline."""

from collections import defaultdict
from typing import Dict, List

from .constants import FMT_KEYS, SEP_TOKEN


def build_representations(doc: Dict) -> Dict:
    """Build the four text representations described in the paper."""
    abstract = doc["abstract"]
    triples_text = doc["triples_text"]
    fmt_hybrid = f"{abstract} {SEP_TOKEN} {triples_text}".strip()

    return {
        "id": doc["id"],
        "title": doc["title"],
        "label": doc["label"],
        "primary_category": doc["primary_category"],
        "categories": doc["categories"],
        "update_date": doc["update_date"],
        "n_triples": len(doc["triples"]),
        "triples": doc["triples"],
        "kg_edges": doc["kg_edges"],
        "triples_text": triples_text,
        "fmt_abstract": abstract,
        "fmt_triples": triples_text,
        "fmt_concatenate": f"{abstract} {triples_text}".strip(),
        "fmt_hybrid": fmt_hybrid,
    }


def print_sample(records: List[Dict], n: int = 3):
    """Print example records showing the four text representations."""
    print("\n" + "=" * 78)
    print("SAMPLE OUTPUTS")
    print("=" * 78)
    for i, record in enumerate(records[:n]):
        print(
            f"\n-- Record {i + 1} -- id={record['id']}  "
            f"label={record['label']}  n_triples={record['n_triples']}"
        )
        for fmt, key in FMT_KEYS.items():
            preview = record[key][:220].replace("\n", " ")
            print(f"\n  [{fmt.upper()}]\n  {preview}")
        print()


def summarize_records(records: List[Dict]) -> Dict[str, object]:
    """Return aggregate stats for one split."""
    labels = defaultdict(int)
    for record in records:
        labels[record["label"]] += 1
    total_triples = sum(record["n_triples"] for record in records)
    avg_triples = total_triples / len(records) if records else 0
    return {
        "num_documents": len(records),
        "label_distribution": dict(labels),
        "total_triples": total_triples,
        "avg_triples": avg_triples,
    }
