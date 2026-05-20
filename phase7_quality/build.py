"""Build Phase 7 triple-quality artifacts from Phase 1 outputs."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import read_jsonl, write_csv, write_json, write_jsonl
from .scoring import linearize_triples, score_triple


log = logging.getLogger(__name__)

QUALITY_REPRESENTATIONS = (
    "quality_triples_top5",
    "quality_hybrid_top5",
    "quality_triples_top50",
    "quality_hybrid_top50",
    "quality_hybrid_banded",
)


def build_quality_artifacts(
    *,
    phase1_root: Path,
    output_root: Path,
    splits: list[str],
    top_k: int,
    top_fraction: float,
    high_threshold: float,
    limit: int | None = None,
) -> dict:
    records_by_split = {
        split: _load_split_records(phase1_root=phase1_root, split=split, limit=limit)
        for split in splits
    }
    relation_counts = _count_relations(records_by_split)
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "quality_config.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_root": str(phase1_root),
        "splits": splits,
        "top_k": top_k,
        "top_fraction": top_fraction,
        "high_threshold": high_threshold,
        "limit": limit,
    })
    write_json(output_root / "relation_frequency.json", dict(sorted(relation_counts.items())))

    triple_quality_rows: list[dict] = []
    split_summaries: list[dict] = []
    total_documents = 0
    total_triples = 0

    for split, records in records_by_split.items():
        quality_records = []
        for row_index, record in enumerate(records):
            quality_record, triple_rows = build_quality_record(
                record=record,
                row_index=row_index,
                relation_counts=relation_counts,
                top_k=top_k,
                top_fraction=top_fraction,
                high_threshold=high_threshold,
            )
            quality_records.append(quality_record)
            triple_quality_rows.extend(triple_rows)

        write_jsonl(output_root / f"{split}_quality_combined.jsonl", quality_records)
        for representation in QUALITY_REPRESENTATIONS:
            _write_representation_view(
                output_root=output_root,
                split=split,
                representation=representation,
                records=quality_records,
            )

        summary = _summarize_split(
            split=split,
            records=quality_records,
            high_threshold=high_threshold,
        )
        split_summaries.append(summary)
        total_documents += int(summary["n_documents"])
        total_triples += int(summary["n_triples"])
        log.info("Built %s quality artifacts: %d records", split, len(quality_records))

    write_jsonl(output_root / "triple_quality_rows.jsonl", triple_quality_rows)
    summary_rows = split_summaries + [_summarize_all(split_summaries, total_documents, total_triples)]
    write_csv(
        output_root / "triple_quality_summary.csv",
        summary_rows,
        fieldnames=[
            "split",
            "n_documents",
            "n_triples",
            "avg_triples_per_doc",
            "avg_quality_score",
            "max_quality_score",
            "avg_high_triples_per_doc",
            "avg_low_triples_per_doc",
            "zero_triple_documents",
            "high_threshold",
        ],
    )

    return {
        "output_root": str(output_root),
        "splits": splits,
        "n_documents": total_documents,
        "n_triples": total_triples,
        "n_relation_types": len(relation_counts),
    }


def _load_split_records(*, phase1_root: Path, split: str, limit: int | None) -> list[dict]:
    path = phase1_root / f"{split}_combined.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1 combined file: {path}")
    records: list[dict] = []
    for row_index, record in enumerate(read_jsonl(path)):
        if limit is not None and row_index >= limit:
            break
        records.append(record)
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def _count_relations(records_by_split: dict[str, list[dict]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for records in records_by_split.values():
        for record in records:
            for triple in record.get("triples", []) or []:
                relation = str(triple.get("rel", "")).strip().lower()
                if relation:
                    counts[relation] += 1
    return counts


def build_quality_record(
    *,
    record: dict,
    row_index: int,
    relation_counts: Counter[str],
    top_k: int,
    top_fraction: float,
    high_threshold: float,
) -> tuple[dict, list[dict]]:
    triples = list(record.get("triples", []) or [])
    scored_triples = [
        score_triple(triple, relation_counts)
        for triple in triples
    ]
    scored_triples.sort(
        key=lambda item: (
            -float(item.get("quality_score", 0.0)),
            str(item.get("subj", "")),
            str(item.get("rel", "")),
            str(item.get("obj", "")),
        )
    )

    top5_triples = scored_triples[:max(0, top_k)]
    top50_count = _top_fraction_count(len(scored_triples), top_fraction)
    top50_triples = scored_triples[:top50_count]
    high_triples = [
        triple for triple in scored_triples
        if float(triple.get("quality_score", 0.0)) >= high_threshold
    ]
    low_triples = [
        triple for triple in scored_triples
        if float(triple.get("quality_score", 0.0)) < high_threshold
    ]

    abstract = str(record.get("fmt_abstract") or "")
    top5_text = linearize_triples(top5_triples)
    top50_text = linearize_triples(top50_triples)
    high_text = linearize_triples(high_triples)
    low_text = linearize_triples(low_triples)

    quality_scores = [float(triple["quality_score"]) for triple in scored_triples]
    quality_record = dict(record)
    quality_record.update(
        {
            "row_index": int(record.get("row_index", row_index)),
            "quality_triples": scored_triples,
            "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 6)
            if quality_scores else 0.0,
            "max_quality_score": round(max(quality_scores), 6) if quality_scores else 0.0,
            "n_quality_high": len(high_triples),
            "n_quality_low": len(low_triples),
            "n_quality_top5": len(top5_triples),
            "n_quality_top50": len(top50_triples),
            "fmt_quality_triples_top5": top5_text,
            "fmt_quality_hybrid_top5": _hybrid_text(abstract, top5_text),
            "fmt_quality_triples_top50": top50_text,
            "fmt_quality_hybrid_top50": _hybrid_text(abstract, top50_text),
            "fmt_quality_hybrid_banded": _banded_hybrid_text(
                abstract=abstract,
                high_text=high_text,
                low_text=low_text,
            ),
        }
    )

    triple_rows = [
        {
            "row_index": quality_record["row_index"],
            "id": str(record.get("id", "")),
            "label": str(record.get("label", "")),
            "triple_rank": rank,
            "subj": triple.get("subj", ""),
            "rel": triple.get("rel", ""),
            "obj": triple.get("obj", ""),
            "rule_tag": triple.get("rule_tag", ""),
            "quality_score": triple.get("quality_score", 0.0),
            "dependency_rule_score": triple.get("dependency_rule_score", 0.0),
            "phrase_length_score": triple.get("phrase_length_score", 0.0),
            "relation_frequency_score": triple.get("relation_frequency_score", 0.0),
            "source_sentence_score": triple.get("source_sentence_score", 0.0),
            "source_sentence": triple.get("source_sentence", ""),
        }
        for rank, triple in enumerate(scored_triples, start=1)
    ]
    return quality_record, triple_rows


def _top_fraction_count(n_triples: int, top_fraction: float) -> int:
    if n_triples <= 0:
        return 0
    fraction = max(0.0, min(1.0, float(top_fraction)))
    return max(1, int((n_triples * fraction) + 0.999999))


def _hybrid_text(abstract: str, triples_text: str) -> str:
    abstract = str(abstract or "").strip()
    triples_text = str(triples_text or "").strip()
    if abstract and triples_text:
        return f"{abstract} [SEP] {triples_text}"
    return abstract or triples_text


def _banded_hybrid_text(*, abstract: str, high_text: str, low_text: str) -> str:
    parts = [str(abstract or "").strip()]
    if high_text.strip():
        parts.append(high_text.strip())
    if low_text.strip():
        parts.append(low_text.strip())
    parts = [part for part in parts if part]
    return " [SEP] ".join(parts)


def _write_representation_view(
    *,
    output_root: Path,
    split: str,
    representation: str,
    records: list[dict],
) -> None:
    text_key = f"fmt_{representation}"
    slim_rows = [
        {
            "id": str(record.get("id", "")),
            "label": str(record.get("label", "")),
            "primary_category": str(record.get("primary_category", "")),
            "n_triples": int(record.get("n_triples", 0)),
            "avg_quality_score": record.get("avg_quality_score", 0.0),
            "max_quality_score": record.get("max_quality_score", 0.0),
            "n_quality_high": record.get("n_quality_high", 0),
            "n_quality_low": record.get("n_quality_low", 0),
            "text": str(record.get(text_key, "")),
        }
        for record in records
    ]
    write_jsonl(output_root / f"{split}_{representation}.jsonl", slim_rows)
    write_csv(
        output_root / f"{split}_{representation}.csv",
        slim_rows,
        fieldnames=[
            "id",
            "label",
            "primary_category",
            "n_triples",
            "avg_quality_score",
            "max_quality_score",
            "n_quality_high",
            "n_quality_low",
            "text",
        ],
    )


def _summarize_split(*, split: str, records: list[dict], high_threshold: float) -> dict:
    n_documents = len(records)
    n_triples = sum(int(record.get("n_triples", 0)) for record in records)
    quality_scores = [
        float(triple.get("quality_score", 0.0))
        for record in records
        for triple in record.get("quality_triples", [])
    ]
    high_counts = [int(record.get("n_quality_high", 0)) for record in records]
    low_counts = [int(record.get("n_quality_low", 0)) for record in records]
    thresholds = [
        1 if int(record.get("n_quality_high", 0)) or int(record.get("n_quality_low", 0)) else 0
        for record in records
    ]
    return {
        "split": split,
        "n_documents": n_documents,
        "n_triples": n_triples,
        "avg_triples_per_doc": round(n_triples / n_documents, 6) if n_documents else 0.0,
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 6)
        if quality_scores else 0.0,
        "max_quality_score": round(max(quality_scores), 6) if quality_scores else 0.0,
        "avg_high_triples_per_doc": round(sum(high_counts) / n_documents, 6)
        if n_documents else 0.0,
        "avg_low_triples_per_doc": round(sum(low_counts) / n_documents, 6)
        if n_documents else 0.0,
        "zero_triple_documents": thresholds.count(0),
        "high_threshold": high_threshold,
    }


def _summarize_all(
    split_summaries: list[dict],
    total_documents: int,
    total_triples: int,
) -> dict:
    if not split_summaries:
        return {
            "split": "all",
            "n_documents": 0,
            "n_triples": 0,
            "avg_triples_per_doc": 0.0,
            "avg_quality_score": 0.0,
            "max_quality_score": 0.0,
            "avg_high_triples_per_doc": 0.0,
            "avg_low_triples_per_doc": 0.0,
            "zero_triple_documents": 0,
            "high_threshold": "",
        }
    return {
        "split": "all",
        "n_documents": total_documents,
        "n_triples": total_triples,
        "avg_triples_per_doc": round(total_triples / total_documents, 6)
        if total_documents else 0.0,
        "avg_quality_score": _weighted_average(split_summaries, "avg_quality_score", "n_triples"),
        "max_quality_score": max(float(row["max_quality_score"]) for row in split_summaries),
        "avg_high_triples_per_doc": _weighted_average(
            split_summaries,
            "avg_high_triples_per_doc",
            "n_documents",
        ),
        "avg_low_triples_per_doc": _weighted_average(
            split_summaries,
            "avg_low_triples_per_doc",
            "n_documents",
        ),
        "zero_triple_documents": sum(int(row["zero_triple_documents"]) for row in split_summaries),
        "high_threshold": split_summaries[0].get("high_threshold", ""),
    }


def _weighted_average(rows: list[dict], value_key: str, weight_key: str) -> float:
    numerator = sum(float(row[value_key]) * float(row[weight_key]) for row in rows)
    denominator = sum(float(row[weight_key]) for row in rows)
    return round(numerator / denominator, 6) if denominator else 0.0
