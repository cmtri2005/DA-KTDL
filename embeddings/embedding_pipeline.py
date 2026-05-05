"""Main orchestration for Phase 2 embedding generation."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from .cli import REPRESENTATIONS, SPLITS, expand_requested, parse_args
from .encoders import choose_device, encode_texts, resolve_models
from .io_utils import load_phase1_records, slugify_model_name, write_json, write_jsonl


def configure_logging() -> None:
    """Configure consistent console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def summarize_records(records: list[dict]) -> dict:
    """Return compact summary stats for loaded Phase 1 records."""
    label_counts = Counter(record["label"] for record in records)
    text_lengths = [record["text_num_chars"] for record in records]
    triple_counts = [record["n_triples"] for record in records]
    return {
        "num_documents": len(records),
        "num_labels": len(label_counts),
        "avg_text_chars": round(sum(text_lengths) / len(text_lengths), 2),
        "avg_n_triples": round(sum(triple_counts) / len(triple_counts), 2),
    }


def build_verification(embeddings: np.ndarray, normalize: bool) -> dict:
    """Compute verification metrics required by Phase 2."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "shape": [int(value) for value in embeddings.shape],
        "dtype": str(embeddings.dtype),
        "has_nan": bool(np.isnan(embeddings).any()),
        "has_inf": bool(np.isinf(embeddings).any()),
        "zero_norm_vectors": int(np.sum(norms <= 1e-12)),
        "norm_min": float(norms.min()) if len(norms) else 0.0,
        "norm_max": float(norms.max()) if len(norms) else 0.0,
        "norm_mean": float(norms.mean()) if len(norms) else 0.0,
        "norm_std": float(norms.std()) if len(norms) else 0.0,
        "normalized_requested": normalize,
        "approx_unit_norm": bool(np.allclose(norms, 1.0, atol=1e-3)) if normalize else None,
    }


def ensure_verification_ok(verification: dict) -> None:
    """Fail fast when embeddings violate required health checks."""
    if verification["has_nan"]:
        raise ValueError("Verification failed: embeddings contain NaN values.")
    if verification["has_inf"]:
        raise ValueError("Verification failed: embeddings contain Inf values.")
    if verification["zero_norm_vectors"] > 0:
        raise ValueError("Verification failed: one or more embeddings have zero norm.")
    if verification["normalized_requested"] and not verification["approx_unit_norm"]:
        raise ValueError("Verification failed: normalized embeddings are not close to unit norm.")


def prepare_metadata(records: list[dict]) -> list[dict]:
    """Keep the id-to-label mapping plus lightweight metadata."""
    metadata = []
    for index, record in enumerate(records):
        metadata.append(
            {
                "row_index": index,
                "id": record["id"],
                "label": record["label"],
                "primary_category": record["primary_category"],
                "n_triples": record["n_triples"],
                "text_num_chars": record["text_num_chars"],
            }
        )
    return metadata


def job_output_dir(output_root, split: str, representation: str, model_name: str):
    """Return the directory where one job should be saved."""
    return output_root / split / representation / slugify_model_name(model_name)


def run() -> None:
    """Execute the Phase 2 pipeline."""
    configure_logging()
    log = logging.getLogger(__name__)
    args = parse_args()

    splits = expand_requested(args.splits, SPLITS, "splits")
    representations = expand_requested(args.representations, REPRESENTATIONS, "representations")
    model_specs = resolve_models(args.models)
    device = choose_device(args.device)

    log.info("=" * 72)
    log.info("Phase 2 - Embedding Generation")
    log.info("=" * 72)
    log.info("Phase 1 input  : %s", args.phase1_output)
    log.info("Output root    : %s", args.output_root)
    log.info("Splits         : %s", ", ".join(splits))
    log.info("Representations: %s", ", ".join(representations))
    log.info("Batch size     : %d", args.batch_size)
    log.info("Max length     : %d", args.max_length)
    log.info("Pooling        : %s", args.pooling)
    log.info("Normalize L2   : %s", args.normalize)
    log.info("Device         : %s", device)

    for split in splits:
        for representation in representations:
            records, source_path, source_kind = load_phase1_records(
                phase1_output=args.phase1_output,
                split=split,
                representation=representation,
            )
            summary = summarize_records(records)
            log.info(
                "[%s/%s] Loaded %d docs from %s (%s)",
                split,
                representation,
                summary["num_documents"],
                source_path,
                source_kind,
            )
            log.info(
                "[%s/%s] avg_text_chars=%s avg_n_triples=%s labels=%d",
                split,
                representation,
                summary["avg_text_chars"],
                summary["avg_n_triples"],
                summary["num_labels"],
            )

            texts = [record["text"] for record in records]
            metadata = prepare_metadata(records)

            for spec in model_specs:
                output_dir = job_output_dir(
                    output_root=args.output_root,
                    split=split,
                    representation=representation,
                    model_name=spec.huggingface_name,
                )
                embeddings_path = output_dir / "embeddings.npy"
                if embeddings_path.exists() and not args.overwrite:
                    log.info(
                        "[%s/%s/%s] Skipping existing output at %s",
                        split,
                        representation,
                        spec.alias,
                        output_dir,
                    )
                    continue

                output_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    "[%s/%s/%s] Encoding %d docs ...",
                    split,
                    representation,
                    spec.alias,
                    len(texts),
                )

                embeddings = encode_texts(
                    texts=texts,
                    spec=spec,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    pooling=args.pooling,
                    normalize=args.normalize,
                )
                verification = build_verification(embeddings, normalize=args.normalize)
                ensure_verification_ok(verification)

                np.save(embeddings_path, embeddings.astype(np.float32))
                write_jsonl(output_dir / "metadata.jsonl", metadata)
                write_json(output_dir / "verification.json", verification)
                write_json(
                    output_dir / "run_config.json",
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "phase1_output": str(args.phase1_output),
                        "source_path": str(source_path),
                        "source_kind": source_kind,
                        "split": split,
                        "representation": representation,
                        "model_alias": spec.alias,
                        "model_name": spec.huggingface_name,
                        "backend": spec.backend,
                        "batch_size": args.batch_size,
                        "max_length": args.max_length,
                        "pooling": args.pooling,
                        "device": device,
                        "normalize": args.normalize,
                        "num_documents": len(records),
                    },
                )

                log.info(
                    "[%s/%s/%s] Saved embeddings shape=%s -> %s",
                    split,
                    representation,
                    spec.alias,
                    verification["shape"],
                    output_dir,
                )


def main() -> None:
    """CLI entrypoint."""
    run()
