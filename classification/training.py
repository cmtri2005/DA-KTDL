"""Training helpers for Phase 5 supervised classification."""

from __future__ import annotations

import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .io_utils import write_csv, write_json
from .metrics import (
    build_confusion_rows,
    build_prediction_rows,
    build_top_error_rows,
    compute_label_counts,
    compute_metrics,
)


log = logging.getLogger(__name__)
_CPU_THREADS_CONFIGURED = False

MODEL_REGISTRY = {
    "scibert": {
        "hf_name": "allenai/scibert_scivocab_uncased",
        "model_slug": "allenai-scibert_scivocab_uncased",
    },
    "specter": {
        "hf_name": "allenai/specter",
        "model_slug": "allenai-specter",
    },
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_slug": "sentence-transformers-all-MiniLM-L6-v2",
    },
}


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    learning_rate: float
    batch_size: int
    epochs: int
    best_epoch: int
    best_val_loss: float
    metrics: dict
    output_dir: Path
    checkpoint_dir: Path


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def configure_torch_cpu_threads(device: str) -> None:
    global _CPU_THREADS_CONFIGURED
    if not device.startswith("cpu"):
        return
    if _CPU_THREADS_CONFIGURED:
        return
    available = os.cpu_count() or 1
    threads = min(available, 8)
    interop_threads = min(threads, 4)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        log.warning("Could not update torch interop threads after initialization.")
    log.info("Torch CPU threads       : %s", threads)
    log.info("Torch CPU interop       : %s", interop_threads)
    _CPU_THREADS_CONFIGURED = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _prepare_label_space(examples: list[dict]) -> tuple[list[str], dict[str, int], np.ndarray]:
    label_names = sorted({str(example["label"]) for example in examples})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.asarray([label_to_id[str(example["label"])] for example in examples], dtype=np.int64)
    return label_names, label_to_id, y


def _split_examples(
    *,
    examples: list[dict],
    y: np.ndarray,
    train_size: float,
    seed: int,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    indices = np.arange(len(examples))
    label_counts = np.bincount(y)
    stratify = y
    if np.min(label_counts) < 2:
        stratify = None
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_examples = [examples[int(index)] for index in train_idx]
    val_examples = [examples[int(index)] for index in val_idx]
    return train_examples, val_examples, y[train_idx], y[val_idx]


def _tokenize_examples(
    *,
    tokenizer,
    examples: list[dict],
    max_length: int,
) -> dict[str, torch.Tensor]:
    texts = [str(example["model_text"]) for example in examples]
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    *,
    progress_desc: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probs_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    iterator = dataloader
    if progress_desc is not None:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False)
    with torch.no_grad():
        for batch in iterator:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            losses.append(float(loss.detach().cpu().item()))
            probs_chunks.append(probs.detach().cpu().numpy())
            pred_chunks.append(preds.detach().cpu().numpy())
    mean_loss = float(np.mean(losses)) if losses else math.inf
    probabilities = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.empty((0, 0))
    predictions = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty((0,), dtype=np.int64)
    return mean_loss, probabilities, predictions


def _write_trial_artifacts(
    *,
    trial_dir: Path,
    epoch_rows: list[dict],
    val_examples: list[dict],
    y_val: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    trial_summary: dict,
) -> None:
    write_csv(
        trial_dir / "epoch_history.csv",
        epoch_rows,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "top3_accuracy",
        ],
    )

    prediction_rows = build_prediction_rows(
        examples=val_examples,
        y_true=y_val,
        y_pred=y_pred,
        probabilities=probabilities,
        label_names=label_names,
    )
    write_csv(
        trial_dir / "validation_predictions.csv",
        prediction_rows,
        fieldnames=list(prediction_rows[0].keys()) if prediction_rows else [
            "row_index",
            "id",
            "label",
            "predicted_label",
            "predicted_confidence",
            "is_correct",
            "top3_labels",
            "clustering_representation",
            "classifier_representation",
            "propagated_cluster_id",
            "propagation_confidence",
        ],
    )

    confusion_rows = build_confusion_rows(
        y_true=y_val,
        y_pred=y_pred,
        label_names=label_names,
    )
    confusion_fields = ["true_label", *label_names]
    write_csv(trial_dir / "confusion_matrix.csv", confusion_rows, fieldnames=confusion_fields)

    top_error_rows = build_top_error_rows(
        examples=val_examples,
        y_true=y_val,
        y_pred=y_pred,
        probabilities=probabilities,
        label_names=label_names,
    )
    top_error_fields = list(top_error_rows[0].keys()) if top_error_rows else [
        "row_index",
        "id",
        "label",
        "predicted_label",
        "predicted_confidence",
        "propagated_cluster_id",
        "propagation_confidence",
        "base_text_num_chars",
    ]
    write_csv(trial_dir / "top_errors.csv", top_error_rows, fieldnames=top_error_fields)
    write_json(trial_dir / "trial_summary.json", trial_summary)


def run_experiment(
    *,
    examples: list[dict],
    model_alias: str,
    output_dir: Path,
    max_length: int,
    train_size: float,
    optuna_trials: int,
    lr_min: float,
    lr_max: float,
    batch_sizes: list[int],
    epochs_min: int,
    epochs_max: int,
    early_stopping_patience: int,
    weight_decay: float,
    num_workers: int,
    device: str,
    seed: int,
    overwrite: bool,
    experiment_metadata: dict,
) -> dict:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    configure_torch_cpu_threads(device)

    model_spec = MODEL_REGISTRY[model_alias]
    tokenizer = AutoTokenizer.from_pretrained(model_spec["hf_name"])

    label_names, _, y_all = _prepare_label_space(examples)
    train_examples, val_examples, y_train, y_val = _split_examples(
        examples=examples,
        y=y_all,
        train_size=train_size,
        seed=seed,
    )

    train_encodings = _tokenize_examples(
        tokenizer=tokenizer,
        examples=train_examples,
        max_length=max_length,
    )
    val_encodings = _tokenize_examples(
        tokenizer=tokenizer,
        examples=val_examples,
        max_length=max_length,
    )

    train_dataset = EncodedTextDataset(train_encodings, y_train)
    val_dataset = EncodedTextDataset(val_encodings, y_val)

    split_payload = {
        "num_examples_total": len(examples),
        "num_examples_train": len(train_examples),
        "num_examples_val": len(val_examples),
        "train_size": train_size,
        "label_names": label_names,
        "label_counts_total": compute_label_counts([str(example["label"]) for example in examples]),
        "label_counts_train": compute_label_counts([str(example["label"]) for example in train_examples]),
        "label_counts_val": compute_label_counts([str(example["label"]) for example in val_examples]),
        "seed": seed,
    }
    write_json(output_dir / "dataset_split.json", split_payload)

    trial_summaries: list[dict] = []
    best_trial_result: TrialResult | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_trial_result

        learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        batch_size = int(trial.suggest_categorical("batch_size", batch_sizes))
        epochs = int(trial.suggest_int("epochs", epochs_min, epochs_max))

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = trial_dir / "best_checkpoint"

        trial_seed = seed + trial.number
        seed_everything(trial_seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_spec["hf_name"],
            num_labels=len(label_names),
        ).to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = max(1, len(train_loader) * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )

        log.info(
            "Trial %03d start         : lr=%.2e batch_size=%d epochs=%d num_train_batches=%d num_val_batches=%d",
            trial.number,
            learning_rate,
            batch_size,
            epochs,
            len(train_loader),
            len(val_loader),
        )

        best_val_loss = math.inf
        best_epoch = 0
        best_probs: np.ndarray | None = None
        best_preds: np.ndarray | None = None
        epochs_without_improvement = 0
        epoch_rows: list[dict] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            train_progress = tqdm(
                train_loader,
                desc=f"trial {trial.number:03d} epoch {epoch}/{epochs} train",
                leave=False,
            )
            for batch in train_progress:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_loss = float(loss.detach().cpu().item())
                train_losses.append(batch_loss)
                train_progress.set_postfix(loss=f"{batch_loss:.4f}")

            train_loss = float(np.mean(train_losses)) if train_losses else math.inf
            val_loss, probabilities, predictions = _evaluate(
                model,
                val_loader,
                device,
                progress_desc=f"trial {trial.number:03d} epoch {epoch}/{epochs} eval",
            )
            epoch_metrics = compute_metrics(
                y_true=y_val,
                y_pred=predictions,
                probabilities=probabilities,
                label_names=label_names,
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": epoch_metrics["accuracy"],
                    "f1_macro": epoch_metrics["f1_macro"],
                    "f1_weighted": epoch_metrics["f1_weighted"],
                    "top3_accuracy": epoch_metrics["top3_accuracy"],
                }
            )

            log.info(
                "Trial %03d epoch %d/%d   : train_loss=%.4f val_loss=%.4f acc=%.4f f1_macro=%.4f top3=%.4f",
                trial.number,
                epoch,
                epochs,
                train_loss,
                val_loss,
                epoch_metrics["accuracy"],
                epoch_metrics["f1_macro"],
                epoch_metrics["top3_accuracy"],
            )

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch
                best_probs = probabilities
                best_preds = predictions
                epochs_without_improvement = 0
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                log.info(
                    "Trial %03d new best     : epoch=%d val_loss=%.4f checkpoint=%s",
                    trial.number,
                    best_epoch,
                    best_val_loss,
                    checkpoint_dir,
                )
            else:
                epochs_without_improvement += 1
                log.info(
                    "Trial %03d patience     : %d/%d",
                    trial.number,
                    epochs_without_improvement,
                    early_stopping_patience,
                )
                if epochs_without_improvement >= early_stopping_patience:
                    log.info(
                        "Trial %03d early stop   : best_epoch=%d best_val_loss=%.4f",
                        trial.number,
                        best_epoch,
                        best_val_loss,
                    )
                    break

        if best_probs is None or best_preds is None:
            raise RuntimeError("No validation predictions were produced during training.")

        best_metrics = compute_metrics(
            y_true=y_val,
            y_pred=best_preds,
            probabilities=best_probs,
            label_names=label_names,
        )
        trial_summary = {
            "trial_number": trial.number,
            "model_alias": model_alias,
            "model_hf_name": model_spec["hf_name"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "metrics": best_metrics,
            "label_names": label_names,
            "device": device,
        }
        _write_trial_artifacts(
            trial_dir=trial_dir,
            epoch_rows=epoch_rows,
            val_examples=val_examples,
            y_val=y_val,
            y_pred=best_preds,
            probabilities=best_probs,
            label_names=label_names,
            trial_summary=trial_summary,
        )

        trial_summaries.append(
            {
                "trial_number": trial.number,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                **best_metrics,
            }
        )

        candidate = TrialResult(
            trial_number=trial.number,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            metrics=best_metrics,
            output_dir=trial_dir,
            checkpoint_dir=checkpoint_dir,
        )
        if best_trial_result is None or candidate.best_val_loss < best_trial_result.best_val_loss:
            best_trial_result = candidate
        log.info(
            "Trial %03d done          : best_epoch=%d best_val_loss=%.4f acc=%.4f f1_macro=%.4f",
            trial.number,
            best_epoch,
            best_val_loss,
            best_metrics["accuracy"],
            best_metrics["f1_macro"],
        )
        return best_val_loss

    study = optuna.create_study(
        study_name=output_dir.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=optuna_trials)

    if best_trial_result is None:
        raise RuntimeError(f"No successful trial finished for {output_dir}")

    write_csv(
        output_dir / "trial_summaries.csv",
        sorted(trial_summaries, key=lambda item: item["trial_number"]),
        fieldnames=list(trial_summaries[0].keys()),
    )

    experiment_summary = {
        **experiment_metadata,
        "model_alias": model_alias,
        "model_slug": model_spec["model_slug"],
        "model_hf_name": model_spec["hf_name"],
        "best_trial_number": best_trial_result.trial_number,
        "best_trial_output_dir": str(best_trial_result.output_dir),
        "best_checkpoint_dir": str(best_trial_result.checkpoint_dir),
        "best_learning_rate": best_trial_result.learning_rate,
        "best_batch_size": best_trial_result.batch_size,
        "best_epochs": best_trial_result.epochs,
        "best_epoch": best_trial_result.best_epoch,
        "best_val_loss": best_trial_result.best_val_loss,
        **best_trial_result.metrics,
    }
    write_json(output_dir / "experiment_summary.json", experiment_summary)
    return experiment_summary
