"""Phase 5 - supervised classification with propagated cluster signals."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cli import MODEL_ALIASES, REPRESENTATIONS, expand_requested, parse_args
from .io_utils import (
    detect_phase1_root,
    detect_phase3_root,
    detect_phase4_root,
    iter_propagation_jobs,
    load_phase1_records,
    load_propagation_records,
    select_best_propagation_jobs,
    write_csv,
    write_json,
)
from .training import MODEL_REGISTRY, resolve_device, run_experiment


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSpec:
    clustering_representation: str
    classifier_representation: str
    model_alias: str


PAPER_TABLE2_EXPERIMENTS = (
    ExperimentSpec("abstract", "abstract", "specter"),
    ExperimentSpec("abstract", "triples", "scibert"),
    ExperimentSpec("abstract", "concatenate", "specter"),
    ExperimentSpec("abstract", "hybrid", "scibert"),
    ExperimentSpec("triples", "abstract", "scibert"),
    ExperimentSpec("triples", "triples", "scibert"),
    ExperimentSpec("triples", "concatenate", "scibert"),
    ExperimentSpec("triples", "hybrid", "scibert"),
    ExperimentSpec("concatenate", "abstract", "scibert"),
    ExperimentSpec("concatenate", "triples", "scibert"),
    ExperimentSpec("concatenate", "concatenate", "scibert"),
    ExperimentSpec("concatenate", "hybrid", "scibert"),
    ExperimentSpec("hybrid", "abstract", "scibert"),
    ExperimentSpec("hybrid", "triples", "scibert"),
    ExperimentSpec("hybrid", "concatenate", "specter"),
    ExperimentSpec("hybrid", "hybrid", "scibert"),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def format_cluster_signal(
    *,
    clustering_representation: str,
    propagated_cluster_id: int,
    propagation_confidence: float,
    base_text: str,
    mode: str,
) -> str:
    if mode == "none":
        return base_text

    cluster_text = "noise" if int(propagated_cluster_id) == -1 else str(int(propagated_cluster_id))
    signal = (
        f"clustering mode {clustering_representation}. "
        f"propagated cluster {cluster_text}. "
        f"confidence {float(propagation_confidence):.3f}."
    )
    if not base_text:
        return signal
    if mode == "prefix":
        return f"{signal} [SEP] {base_text}"
    if mode == "suffix":
        return f"{base_text} [SEP] {signal}"
    raise ValueError(f"Unsupported cluster signal mode: {mode}")


def build_examples(
    *,
    phase1_records: list[dict],
    propagation_records: dict[str, dict],
    clustering_representation: str,
    classifier_representation: str,
    cluster_signal_mode: str,
) -> list[dict]:
    examples: list[dict] = []
    missing_ids: list[str] = []
    for record in phase1_records:
        propagation_row = propagation_records.get(record["id"])
        if propagation_row is None:
            missing_ids.append(record["id"])
            continue

        base_text = str(record["texts"][classifier_representation] or "")
        propagated_cluster_id = int(propagation_row["propagated_cluster_id"])
        propagation_confidence = float(propagation_row["propagation_confidence"])
        model_text = format_cluster_signal(
            clustering_representation=clustering_representation,
            propagated_cluster_id=propagated_cluster_id,
            propagation_confidence=propagation_confidence,
            base_text=base_text,
            mode=cluster_signal_mode,
        )
        examples.append(
            {
                "row_index": record["row_index"],
                "id": record["id"],
                "label": record["label"],
                "primary_category": record["primary_category"],
                "n_triples": record["n_triples"],
                "clustering_representation": clustering_representation,
                "classifier_representation": classifier_representation,
                "propagated_cluster_id": propagated_cluster_id,
                "propagation_confidence": propagation_confidence,
                "base_text": base_text,
                "model_text": model_text,
            }
        )

    if missing_ids:
        raise ValueError(
            f"Propagation file is missing {len(missing_ids)} ids for "
            f"{clustering_representation}/{classifier_representation}. "
            f"First missing ids: {missing_ids[:5]}"
        )
    if not examples:
        raise ValueError(
            f"No examples built for {clustering_representation}/{classifier_representation}."
        )
    return examples


def _write_aggregate_tables(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return

    all_fields = list(rows[0].keys())
    write_csv(output_root / "results_table_all_runs.csv", rows, fieldnames=all_fields)

    best_by_pair: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            row["model_alias"],
            row["clustering_representation"],
            row["classifier_representation"],
        )
        current = best_by_pair.get(key)
        if current is None or float(row["best_val_loss"]) < float(current["best_val_loss"]):
            best_by_pair[key] = row
    best_rows = sorted(
        best_by_pair.values(),
        key=lambda item: (
            item["model_alias"],
            item["clustering_representation"],
            item["classifier_representation"],
        ),
    )
    write_csv(output_root / "results_table_best_by_pair.csv", best_rows, fieldnames=all_fields)


def _ordered_representation_subset(items: set[str]) -> list[str]:
    return [representation for representation in REPRESENTATIONS if representation in items]


def build_experiment_specs(
    *,
    experiment_plan: str,
    clustering_representations: list[str],
    classifier_representations: list[str],
    model_aliases: list[str],
) -> list[ExperimentSpec]:
    allowed_clusters = set(clustering_representations)
    allowed_classifiers = set(classifier_representations)
    allowed_models = set(model_aliases)

    if experiment_plan == "table2":
        specs = [
            spec
            for spec in PAPER_TABLE2_EXPERIMENTS
            if spec.clustering_representation in allowed_clusters
            and spec.classifier_representation in allowed_classifiers
            and spec.model_alias in allowed_models
        ]
    elif experiment_plan == "full_grid":
        specs = [
            ExperimentSpec(
                clustering_representation=clustering_representation,
                classifier_representation=classifier_representation,
                model_alias=model_alias,
            )
            for model_alias in model_aliases
            for clustering_representation in clustering_representations
            for classifier_representation in classifier_representations
        ]
    else:
        raise ValueError(f"Unsupported experiment plan: {experiment_plan}")

    if not specs:
        raise ValueError(
            "No experiments matched the requested filters. "
            f"plan={experiment_plan}, models={model_aliases}, "
            f"clustering={clustering_representations}, classifier={classifier_representations}"
        )
    return specs


def main() -> None:
    configure_logging()
    args = parse_args()

    phase1_root = detect_phase1_root(args.phase1_root)
    phase3_root = detect_phase3_root(args.phase3_root)
    phase4_root = detect_phase4_root(args.phase4_root)

    clustering_representations = expand_requested(
        args.clustering_representations,
        REPRESENTATIONS,
        "clustering representations",
    )
    classifier_representations = expand_requested(
        args.classifier_representations,
        REPRESENTATIONS,
        "classifier representations",
    )
    model_aliases = expand_requested(args.models, MODEL_ALIASES, "models")
    experiment_specs = build_experiment_specs(
        experiment_plan=args.experiment_plan,
        clustering_representations=clustering_representations,
        classifier_representations=classifier_representations,
        model_aliases=model_aliases,
    )
    required_model_aliases = sorted({spec.model_alias for spec in experiment_specs})
    required_model_slugs = [MODEL_REGISTRY[alias]["model_slug"] for alias in required_model_aliases]
    required_clustering_representations = _ordered_representation_subset(
        {spec.clustering_representation for spec in experiment_specs}
    )
    device = resolve_device(args.device)

    log.info("Phase 5 - Supervised Classification")
    log.info("Phase 1 root             : %s", phase1_root)
    log.info("Phase 3 root             : %s", phase3_root)
    log.info("Phase 4 root             : %s", phase4_root)
    log.info("Output root              : %s", args.output_root)
    log.info("Experiment plan          : %s", args.experiment_plan)
    log.info("Device                   : %s", device)
    log.info("Requested models         : %s", ", ".join(model_aliases))
    log.info("Clustering modes         : %s", ", ".join(clustering_representations))
    log.info("Classifier modes         : %s", ", ".join(classifier_representations))
    log.info("Planned experiments      : %d", len(experiment_specs))

    phase1_records = load_phase1_records(phase1_root)
    propagation_jobs = iter_propagation_jobs(
        phase4_root=phase4_root,
        phase3_root=phase3_root,
        clustering_representations=required_clustering_representations,
        model_slugs=required_model_slugs,
    )
    selected_jobs = select_best_propagation_jobs(propagation_jobs)

    selected_payload = {
        f"{representation}/{model_slug}": {
            "job_dir": str(job.job_dir),
            "algorithm": job.algorithm,
            "param_name": job.param_name,
            "param_value": job.param_value,
            "phase3_score": job.phase3_score,
        }
        for (representation, model_slug), job in sorted(selected_jobs.items())
        if representation in required_clustering_representations
        and model_slug in required_model_slugs
    }
    write_json(
        args.output_root / "selected_phase4_jobs.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": args.experiment_plan,
            "selected_jobs": selected_payload,
        },
    )
    write_json(
        args.output_root / "planned_experiments.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_plan": args.experiment_plan,
            "num_experiments": len(experiment_specs),
            "experiments": [asdict(spec) for spec in experiment_specs],
        },
    )

    rows: list[dict] = []
    errors: list[dict] = []
    propagation_cache: dict[Path, dict[str, dict]] = {}
    total_experiments = len(experiment_specs)
    for experiment_index, spec in enumerate(experiment_specs, start=1):
        model_slug = MODEL_REGISTRY[spec.model_alias]["model_slug"]
        job = selected_jobs.get((spec.clustering_representation, model_slug))
        if job is None:
            raise FileNotFoundError(
                f"No Phase 4 propagation job found for {spec.clustering_representation}/{model_slug}"
            )
        propagation_records = propagation_cache.get(job.propagated_clusters_path)
        if propagation_records is None:
            propagation_records = load_propagation_records(job.propagated_clusters_path)
            propagation_cache[job.propagated_clusters_path] = propagation_records

        experiment_dir = (
            args.output_root
            / spec.clustering_representation
            / spec.classifier_representation
            / model_slug
        )
        experiment_config = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase1_root": str(phase1_root),
            "phase3_root": str(phase3_root),
            "phase4_root": str(phase4_root),
            "experiment_plan": args.experiment_plan,
            "experiment_index": experiment_index,
            "num_experiments": total_experiments,
            "clustering_representation": spec.clustering_representation,
            "classifier_representation": spec.classifier_representation,
            "model_alias": spec.model_alias,
            "model_slug": model_slug,
            "cluster_signal_mode": args.cluster_signal_mode,
            "selected_propagation_job": {
                "job_dir": str(job.job_dir),
                "propagated_clusters_path": str(job.propagated_clusters_path),
                "algorithm": job.algorithm,
                "param_name": job.param_name,
                "param_value": job.param_value,
                "phase3_score": job.phase3_score,
            },
            "max_length": args.max_length,
            "train_size": args.train_size,
            "optuna_trials": args.optuna_trials,
            "lr_min": args.lr_min,
            "lr_max": args.lr_max,
            "batch_sizes": args.batch_sizes,
            "epochs_min": args.epochs_min,
            "epochs_max": args.epochs_max,
            "early_stopping_patience": args.early_stopping_patience,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "device": device,
            "seed": args.seed,
        }
        log.info(
            "Running experiment      : %d/%d model=%s cluster=%s classifier=%s",
            experiment_index,
            total_experiments,
            spec.model_alias,
            spec.clustering_representation,
            spec.classifier_representation,
        )
        try:
            examples = build_examples(
                phase1_records=phase1_records,
                propagation_records=propagation_records,
                clustering_representation=spec.clustering_representation,
                classifier_representation=spec.classifier_representation,
                cluster_signal_mode=args.cluster_signal_mode,
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
                    "experiment_plan": args.experiment_plan,
                    "experiment_index": experiment_index,
                    "num_experiments": total_experiments,
                    "clustering_representation": spec.clustering_representation,
                    "classifier_representation": spec.classifier_representation,
                    "selected_phase4_job_dir": str(job.job_dir),
                    "selected_phase4_algorithm": job.algorithm,
                    "selected_phase4_param_name": job.param_name,
                    "selected_phase4_param_value": job.param_value,
                    "selected_phase4_score": job.phase3_score,
                    "cluster_signal_mode": args.cluster_signal_mode,
                },
            )
            rows.append(summary)
            write_json(experiment_dir / "run_config.json", experiment_config)
            _write_aggregate_tables(args.output_root, rows)
        except Exception as exc:
            error_row = {
                "experiment_index": experiment_index,
                "num_experiments": total_experiments,
                "experiment_plan": args.experiment_plan,
                "model_alias": spec.model_alias,
                "model_slug": model_slug,
                "clustering_representation": spec.clustering_representation,
                "classifier_representation": spec.classifier_representation,
                "error": repr(exc),
            }
            errors.append(error_row)
            log.exception(
                "Experiment failed      : %d/%d model=%s cluster=%s classifier=%s",
                experiment_index,
                total_experiments,
                spec.model_alias,
                spec.clustering_representation,
                spec.classifier_representation,
            )
            if args.fail_fast:
                raise

    _write_aggregate_tables(args.output_root, rows)
    if errors:
        write_csv(
            args.output_root / "failed_experiments.csv",
            errors,
            fieldnames=list(errors[0].keys()),
        )
        raise RuntimeError(
            f"{len(errors)} experiment(s) failed. See {args.output_root / 'failed_experiments.csv'}"
        )


if __name__ == "__main__":
    main()
