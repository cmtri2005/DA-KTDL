"""Phase 7 classification ablation runner."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from classification.cli import MODEL_ALIASES
from classification.training import MODEL_REGISTRY, resolve_device, run_experiment

from .cli import QUALITY_CLASSIFIER_REPRESENTATIONS, expand_requested
from .io_utils import read_jsonl, write_csv, write_json


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase7ExperimentSpec:
    representation: str
    model_alias: str


TEXT_KEYS = {
    "abstract": "fmt_abstract",
    "triples": "fmt_triples",
    "concatenate": "fmt_concatenate",
    "hybrid": "fmt_hybrid",
    "quality_triples_top5": "fmt_quality_triples_top5",
    "quality_hybrid_top5": "fmt_quality_hybrid_top5",
    "quality_triples_top50": "fmt_quality_triples_top50",
    "quality_hybrid_top50": "fmt_quality_hybrid_top50",
    "quality_hybrid_banded": "fmt_quality_hybrid_banded",
}


def run_phase7_classification(args) -> None:
    quality_root = Path(args.quality_root)
    output_root = Path(args.output_root)
    quality_records = load_quality_records(quality_root, limit=args.limit)
    representations = expand_requested(
        args.representations,
        QUALITY_CLASSIFIER_REPRESENTATIONS,
        "Phase 7 classifier representations",
    )
    model_aliases = expand_requested(args.models, MODEL_ALIASES, "models")
    specs = [
        Phase7ExperimentSpec(representation=representation, model_alias=model_alias)
        for model_alias in model_aliases
        for representation in representations
    ]
    if not specs:
        raise ValueError("No Phase 7 classification experiments were requested.")

    device = resolve_device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "planned_experiments.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "quality_root": str(quality_root),
            "num_experiments": len(specs),
            "experiments": [asdict(spec) for spec in specs],
            "note": "Phase 7 does not use Phase 4 propagation; cluster fields are neutral.",
        },
    )

    rows: list[dict] = []
    errors: list[dict] = []
    per_label_rows: list[dict] = []
    total = len(specs)

    for experiment_index, spec in enumerate(specs, start=1):
        model_slug = MODEL_REGISTRY[spec.model_alias]["model_slug"]
        experiment_dir = output_root / spec.representation / model_slug
        log.info(
            "Phase 7 experiment %d/%d: model=%s representation=%s",
            experiment_index,
            total,
            spec.model_alias,
            spec.representation,
        )
        try:
            examples = build_examples(
                quality_records=quality_records,
                representation=spec.representation,
            )
            summary = run_experiment(
                examples=examples,
                model_alias=spec.model_alias,
                output_dir=experiment_dir,
                max_length=args.max_length,
                train_size=args.train_size,
                optuna_trials=args.optuna_trials,
                lr_min=args.lr_min,
                lr_max=args.lr_max,
                batch_sizes=args.batch_sizes,
                epochs_min=args.epochs_min,
                epochs_max=args.epochs_max,
                early_stopping_patience=args.early_stopping_patience,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                device=device,
                seed=args.seed,
                overwrite=args.overwrite,
                experiment_metadata={
                    "phase": 7,
                    "experiment_index": experiment_index,
                    "num_experiments": total,
                    "quality_root": str(quality_root),
                    "clustering_representation": "none",
                    "classifier_representation": spec.representation,
                    "phase7_representation": spec.representation,
                    "cluster_signal_mode": "none",
                },
            )
            rows.append(summary)
            per_label_rows.extend(
                collect_per_label_metrics(
                    summary=summary,
                    representation=spec.representation,
                    model_alias=spec.model_alias,
                    experiment_dir=experiment_dir,
                )
            )
            write_json(
                experiment_dir / "run_config.json",
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "quality_root": str(quality_root),
                    "representation": spec.representation,
                    "model_alias": spec.model_alias,
                    "model_slug": model_slug,
                    "max_length": args.max_length,
                    "train_size": args.train_size,
                    "optuna_trials": args.optuna_trials,
                    "device": device,
                    "seed": args.seed,
                },
            )
            write_phase7_tables(output_root, rows, per_label_rows)
        except Exception as exc:
            error_row = {
                "experiment_index": experiment_index,
                "num_experiments": total,
                "model_alias": spec.model_alias,
                "model_slug": model_slug,
                "representation": spec.representation,
                "error": repr(exc),
            }
            errors.append(error_row)
            log.exception(
                "Phase 7 experiment failed: model=%s representation=%s",
                spec.model_alias,
                spec.representation,
            )
            if args.fail_fast:
                raise

    write_phase7_tables(output_root, rows, per_label_rows)
    if errors:
        write_csv(
            output_root / "failed_experiments.csv",
            errors,
            fieldnames=list(errors[0].keys()),
        )
        raise RuntimeError(
            f"{len(errors)} Phase 7 experiment(s) failed. "
            f"See {output_root / 'failed_experiments.csv'}"
        )


def load_quality_records(quality_root: Path, limit: int | None = None) -> list[dict]:
    path = quality_root / "classify_quality_combined.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 7 quality classification file: {path}. "
            "Run `python -m phase7_quality build` first."
        )
    records: list[dict] = []
    for row_index, row in enumerate(read_jsonl(path)):
        if limit is not None and row_index >= limit:
            break
        records.append(row)
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def build_examples(*, quality_records: list[dict], representation: str) -> list[dict]:
    text_key = TEXT_KEYS[representation]
    examples: list[dict] = []
    for row_index, record in enumerate(quality_records):
        base_text = str(record.get(text_key) or "")
        if not base_text and representation.startswith("quality_"):
            base_text = str(record.get("fmt_abstract") or "")
        examples.append(
            {
                "row_index": int(record.get("row_index", row_index)),
                "id": str(record["id"]),
                "label": str(record["label"]),
                "primary_category": str(record.get("primary_category", "")),
                "n_triples": int(record.get("n_triples", 0)),
                "clustering_representation": "none",
                "classifier_representation": representation,
                "propagated_cluster_id": -1,
                "propagation_confidence": 0.0,
                "base_text": base_text,
                "model_text": base_text,
            }
        )
    if not examples:
        raise ValueError(f"No examples built for Phase 7 representation: {representation}")
    return examples


def collect_per_label_metrics(
    *,
    summary: dict,
    representation: str,
    model_alias: str,
    experiment_dir: Path,
) -> list[dict]:
    trial_dir = Path(summary["best_trial_output_dir"])
    predictions_path = trial_dir / "validation_predictions.csv"
    if not predictions_path.exists():
        predictions_path = experiment_dir / f"trial_{int(summary['best_trial_number']):03d}" / "validation_predictions.csv"
    if not predictions_path.exists():
        return []

    with open(predictions_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    label_names = sorted({row["label"] for row in rows} | {row["predicted_label"] for row in rows})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y_true = np.asarray([label_to_id[row["label"]] for row in rows], dtype=np.int64)
    y_pred = np.asarray([label_to_id[row["predicted_label"]] for row in rows], dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    return [
        {
            "representation": representation,
            "model_alias": model_alias,
            "label": label,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(label_names)
    ]


def write_phase7_tables(
    output_root: Path,
    rows: list[dict],
    per_label_rows: list[dict],
) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                -float(row.get("accuracy", 0.0)),
                -float(row.get("f1_macro", 0.0)),
                str(row.get("classifier_representation", "")),
            ),
        )
        write_csv(output_root / "results_table_phase7.csv", sorted_rows, fieldnames=fieldnames)
        write_csv(output_root / "results_table_phase7_best.csv", sorted_rows[:1], fieldnames=fieldnames)
    if per_label_rows:
        per_label_fields = list(per_label_rows[0].keys())
        write_csv(output_root / "per_label_metrics.csv", per_label_rows, fieldnames=per_label_fields)
        delta_rows = build_per_label_delta_rows(per_label_rows)
        if delta_rows:
            write_csv(
                output_root / "per_label_delta_vs_hybrid.csv",
                delta_rows,
                fieldnames=list(delta_rows[0].keys()),
            )


def build_per_label_delta_rows(per_label_rows: list[dict]) -> list[dict]:
    baseline: dict[tuple[str, str], dict] = {}
    for row in per_label_rows:
        if row["representation"] == "hybrid":
            baseline[(row["model_alias"], row["label"])] = row
    if not baseline:
        return []

    delta_rows: list[dict] = []
    for row in per_label_rows:
        base = baseline.get((row["model_alias"], row["label"]))
        if base is None or row["representation"] == "hybrid":
            continue
        delta_rows.append(
            {
                "representation": row["representation"],
                "model_alias": row["model_alias"],
                "label": row["label"],
                "support": row["support"],
                "baseline_f1": float(base["f1"]),
                "phase7_f1": float(row["f1"]),
                "delta_f1_vs_hybrid": float(row["f1"]) - float(base["f1"]),
            }
        )
    delta_rows.sort(key=lambda item: (item["representation"], -item["delta_f1_vs_hybrid"]))
    return delta_rows
