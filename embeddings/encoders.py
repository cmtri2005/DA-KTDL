"""Embedding backends used in Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class ModelSpec:
    """Description of one supported encoder."""

    alias: str
    huggingface_name: str
    backend: str
    description: str
    dimension_hint: int
    aliases: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        alias="minilm",
        huggingface_name="sentence-transformers/all-MiniLM-L6-v2",
        backend="sentence_transformers",
        description="Lightweight Sentence-Transformers encoder.",
        dimension_hint=384,
        aliases=("minilm", "all-minilm-l6-v2", "sentence-transformers/all-minilm-l6-v2"),
    ),
    ModelSpec(
        alias="mpnet",
        huggingface_name="sentence-transformers/all-mpnet-base-v2",
        backend="sentence_transformers",
        description="Higher-capacity Sentence-Transformers encoder.",
        dimension_hint=768,
        aliases=("mpnet", "all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
    ),
    ModelSpec(
        alias="specter",
        huggingface_name="allenai/specter",
        backend="transformers",
        description="Scientific-document encoder trained with citation signals.",
        dimension_hint=768,
        aliases=("specter", "allenai/specter"),
    ),
    ModelSpec(
        alias="scibert",
        huggingface_name="allenai/scibert_scivocab_uncased",
        backend="transformers",
        description="Domain-adapted BERT encoder for scientific text.",
        dimension_hint=768,
        aliases=("scibert", "allenai/scibert_scivocab_uncased"),
    ),
)

MODEL_LOOKUP = {
    alias: spec
    for spec in MODEL_SPECS
    for alias in (spec.alias, spec.huggingface_name, *spec.aliases)
}

_SENTENCE_TRANSFORMER_CACHE: dict[tuple[str, str, int], object] = {}
_TOKENIZER_CACHE: dict[str, object] = {}
_TRANSFORMER_MODEL_CACHE: dict[tuple[str, str], object] = {}


def resolve_models(raw_items: list[str]) -> list[ModelSpec]:
    """Resolve CLI model identifiers into unique model specs."""
    normalized: list[str] = []
    for item in raw_items:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        normalized.extend(parts or [item])

    lowered = [item.lower() for item in normalized if item.strip()]
    if not lowered or "all" in lowered:
        return list(MODEL_SPECS)

    resolved: list[ModelSpec] = []
    seen: set[str] = set()
    for item in lowered:
        spec = MODEL_LOOKUP.get(item)
        if spec is None:
            valid = ", ".join(spec.alias for spec in MODEL_SPECS)
            raise ValueError(f"Unsupported model '{item}'. Supported aliases: {valid}")
        if spec.alias in seen:
            continue
        seen.add(spec.alias)
        resolved.append(spec)
    return resolved


def choose_device(requested_device: str) -> str:
    """Resolve the execution device without importing torch unless needed."""
    requested = requested_device.strip().lower()
    if requested != "auto":
        return requested_device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Apply safe row-wise L2 normalization."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _encode_with_sentence_transformers(
    texts: list[str],
    spec: ModelSpec,
    batch_size: int,
    max_length: int,
    device: str,
    normalize: bool,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'sentence-transformers'. "
            "Install the packages listed in phase2_embeddings/requirements-phase2.txt"
        ) from exc

    cache_key = (spec.huggingface_name, device, max_length)
    model = _SENTENCE_TRANSFORMER_CACHE.get(cache_key)
    if model is None:
        model = SentenceTransformer(spec.huggingface_name, device=device)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max_length
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = model

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _pool_hidden_states(last_hidden_state, attention_mask, pooling: str):
    """Pool token embeddings into a single vector."""
    if pooling == "cls":
        return last_hidden_state[:, 0]

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _iter_batches(texts: list[str], batch_size: int) -> Iterable[list[str]]:
    """Yield slices of texts with a fixed batch size."""
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]


def _encode_with_transformers(
    texts: list[str],
    spec: ModelSpec,
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    normalize: bool,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers' and/or 'torch'. "
            "Install the packages listed in phase2_embeddings/requirements-phase2.txt"
        ) from exc

    tokenizer = _TOKENIZER_CACHE.get(spec.huggingface_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(spec.huggingface_name)
        _TOKENIZER_CACHE[spec.huggingface_name] = tokenizer

    model_cache_key = (spec.huggingface_name, device)
    model = _TRANSFORMER_MODEL_CACHE.get(model_cache_key)
    if model is None:
        model = AutoModel.from_pretrained(spec.huggingface_name)
        model.to(device)
        model.eval()
        _TRANSFORMER_MODEL_CACHE[model_cache_key] = model

    all_vectors = []
    with torch.no_grad():
        for batch_texts in tqdm(
            _iter_batches(texts, batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=f"Encoding {spec.alias}",
        ):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = _pool_hidden_states(outputs.last_hidden_state, encoded["attention_mask"], pooling)
            all_vectors.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.vstack(all_vectors)
    return l2_normalize(embeddings) if normalize else embeddings


def encode_texts(
    texts: list[str],
    spec: ModelSpec,
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    normalize: bool,
) -> np.ndarray:
    """Encode documents with the backend associated with the chosen model."""
    if spec.backend == "sentence_transformers":
        return _encode_with_sentence_transformers(
            texts=texts,
            spec=spec,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            normalize=normalize,
        )
    if spec.backend == "transformers":
        return _encode_with_transformers(
            texts=texts,
            spec=spec,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            pooling=pooling,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported backend: {spec.backend}")
