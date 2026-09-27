"""Target embedding + cosine similarity (training aug + eval metrics).

Every caller reaches the task's semantic space through the functions here. There
is a single encoding path for every task — decorate the target with the task's
``embedding_prompt``, then encode that string with a sentence-transformers model
— and tasks differ only in the prompt and in which model is named by
``task_config["<task>"]["embedding_model"]``. Current tasks name the same
general text encoder, so molopt SMILES are embedded as strings; the per-task
lookup exists so that does not have to stay true.

Rows come back L2-normalized, so cosine similarity is a dot product and all
downstream metrics (embed_sim, MM/rank losses, RECON/DIST/GENZ/LIPZ) are
task-agnostic.

Structural (fingerprint) similarity for molecules is deliberately *not* here;
see :mod:`boreft.chem` for Morgan/Tanimoto.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from boreft.task_config import (
    definition_embedding_text,
    substitute_placeholders,
    task_config,
    task_embedding_model,
    task_supports_definition_embeds,
)

# One loaded model per model name: a process that evaluates two tasks (or a
# checkpoint whose model differs from the current default) holds both.
_EMBED_MODEL_CACHE: dict[str, SentenceTransformer] = {}
EMBED_NORMALIZE = True


def embedding_model_name(task: str = "semantle") -> str:
    """Name of the sentence-transformers model that embeds this task's targets.

    Always go through this rather than :data:`DEFAULT_EMBEDDING_MODEL`: the default
    is only the fallback for tasks that do not name a model, and reading it directly
    silently reports the wrong model for any task that does.
    """
    return task_embedding_model(task)


def embedding_prompt_template(task: str = "semantle") -> str:
    """Template each of the task's targets is wrapped in before encoding."""
    return task_config[task]["embedding_prompt"]


def _repo_root() -> str:
    path = os.path.abspath(__file__)
    for _ in range(8):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_definitions_path(task: str = "semantle") -> str:
    """Conventional definitions JSONL for a task.

    Most tasks ship ``data/<task>/train/definitions.jsonl``. Hypogen keeps each
    dataset in its own folder, so the v1 corpus lives next to its CSV.
    """
    if task == "hypogen":
        return os.path.join(
            _repo_root(), "data", "hypogen", "evo-fresh-fish", "definitions.jsonl"
        )
    return os.path.join(_repo_root(), "data", task, "train", "definitions.jsonl")


def default_rdkit_definitions_path() -> str:
    from boreft.chem import default_rdkit_definitions_path as _default

    return _default()


def default_rdkit_definitions_map_path() -> str:
    from boreft.chem import default_rdkit_map_path

    return default_rdkit_map_path()


def definitions_row_target(row: Mapping[str, Any]) -> str:
    """Target string from a definitions JSONL row (``target``, legacy ``word``)."""
    if row.get("target") is not None:
        return str(row["target"]).strip()
    if row.get("word") is not None:
        return str(row["word"]).strip()
    raise KeyError("definitions row needs 'target' (or legacy 'word')")


def definition_embed_text(
    word: str, definition: str, *, task: str = "semantle"
) -> str:
    return definition_embedding_text(task, word, definition)


def load_definition_embed_lookup(
    path: str,
    *,
    task: str = "semantle",
    append_rdkit_definitions: bool = False,
    omit_molt5_definitions: bool = False,
    rdkit_definitions_path: Optional[str] = None,
) -> dict[str, str]:
    rdkit_lookup = (
        load_rdkit_definition_lookup(
            rdkit_definitions_path or default_rdkit_definitions_path()
        )
        if append_rdkit_definitions
        else None
    )
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = definitions_row_target(row)
            definition = str(row["definition"]).strip()
            if rdkit_lookup is not None:
                definition = append_rdkit_definition_values(
                    target,
                    definition,
                    rdkit_lookup=rdkit_lookup,
                    require_lookup=True,
                    omit_base_definition=omit_molt5_definitions,
                )
            lookup[target] = definition_embed_text(
                target, definition, task=task
            )
    return lookup


def load_category_normalized_lookup(path: str) -> dict[str, str]:
    """Map ``target -> category_normalized`` from ``definitions.jsonl``."""
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = definitions_row_target(row)
            lookup[target] = str(
                row.get("category_normalized", row.get("category", "unknown"))
            ).strip()
    return lookup


def load_raw_definitions(path: str) -> dict[str, str]:
    """Map ``target -> raw definition text`` (no embedding decoration).

    Used by SDPO to place the definition verbatim in the teacher's context, as
    opposed to :func:`load_definition_embed_lookup` which decorates the text for
    sentence-transformer embedding.
    """
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lookup[definitions_row_target(row)] = str(row["definition"]).strip()
    return lookup


@lru_cache(maxsize=16)
def _load_rdkit_definition_lookup_cached(
    resolved: str, content_sha256: str
) -> dict[str, tuple[float | int, ...]]:
    del content_sha256  # cache-key component; content is read from ``resolved``.
    from boreft.chem import RDKIT_DESCRIPTOR_DIM

    lookup: dict[str, tuple[float | int, ...]] = {}
    with open(resolved, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            target = definitions_row_target(row)
            values = row.get("definition")
            if not isinstance(values, list) or len(values) != RDKIT_DESCRIPTOR_DIM:
                raise ValueError(
                    f"{resolved}:{line_number}: RDKit definition must be an array "
                    f"of {RDKIT_DESCRIPTOR_DIM} scalars"
                )
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise ValueError(
                    f"{resolved}:{line_number}: RDKit definition values must be numeric"
                )
            lookup[target] = tuple(values)
    return lookup


def load_rdkit_definition_lookup(path: str) -> dict[str, tuple[float | int, ...]]:
    """Map target to descriptors, invalidating cache when sidecar content changes."""
    from boreft.chem import file_sha256

    resolved = os.path.abspath(path)
    return _load_rdkit_definition_lookup_cached(resolved, file_sha256(resolved))


def stringify_rdkit_definition(values: Sequence[float | int]) -> str:
    """Semicolon-separated, labeled 2D properties in stored vector order."""
    from boreft.chem import RDKIT_DESCRIPTOR_DIM, RDKIT_DESCRIPTOR_MAP

    if len(values) != RDKIT_DESCRIPTOR_DIM:
        raise ValueError(
            f"RDKit definition must have {RDKIT_DESCRIPTOR_DIM} values, "
            f"got {len(values)}"
        )
    return "; ".join(
        f"{descriptor['definition_label']} {value}"
        for descriptor, value in zip(RDKIT_DESCRIPTOR_MAP, values)
    )


def append_rdkit_definition_values(
    target: str,
    definition: str,
    *,
    rdkit_lookup: Optional[Mapping[str, Sequence[float | int]]] = None,
    require_lookup: bool = False,
    omit_base_definition: bool = False,
) -> str:
    """Append a labeled ``2D properties: ...`` sentence to a definition."""
    values = rdkit_lookup.get(target) if rdkit_lookup is not None else None
    if values is None and require_lookup:
        raise ValueError(f"target {target!r} missing from RDKit definitions sidecar")
    if values is None:
        from boreft.chem import rdkit_descriptor_values

        values = rdkit_descriptor_values(target)
    if values is None:
        raise ValueError(
            f"cannot append RDKit definition properties: {target!r} is not valid SMILES"
        )
    base = "" if omit_base_definition else str(definition).strip()
    suffix = stringify_rdkit_definition(values)
    return f"{base} 2D properties: {suffix}" if base else f"2D properties: {suffix}"


def rdkit_definition_lookup_for_cfg(
    cfg: Mapping[str, object],
) -> Optional[dict[str, tuple[float | int, ...]]]:
    if not cfg.get("append_rdkit_definitions"):
        return None
    path = str(
        cfg.get("rdkit_definitions_path") or default_rdkit_definitions_path()
    )
    expected_hash = cfg.get("rdkit_definitions_sha256")
    if expected_hash:
        from boreft.chem import file_sha256

        actual_hash = file_sha256(path)
        if actual_hash != str(expected_hash):
            raise ValueError(
                "RDKit definitions sidecar content differs from checkpoint "
                f"provenance: {path}"
            )
    return load_rdkit_definition_lookup(path)


def rdkit_map_path_for_cfg(cfg: Mapping[str, object]) -> Optional[str]:
    """Resolved molopt map path, checked against saved content provenance."""
    if str(cfg.get("task", "semantle")) != "molopt":
        return None
    path = os.path.abspath(
        str(
            cfg.get("rdkit_definitions_map_path")
            or default_rdkit_definitions_map_path()
        )
    )
    from boreft.chem import file_sha256, load_rdkit_descriptor_map

    expected_hash = cfg.get("rdkit_definitions_map_sha256")
    if expected_hash and file_sha256(path) != str(expected_hash):
        raise ValueError(
            "RDKit descriptor map content differs from checkpoint provenance: "
            f"{path}"
        )
    load_rdkit_descriptor_map(path)
    return path


def definition_text_for_cfg(
    cfg: Mapping[str, object],
    target: str,
    definition: str,
    *,
    rdkit_lookup: Optional[Mapping[str, Sequence[float | int]]] = None,
    require_lookup: bool = False,
) -> str:
    """Natural definition, optionally suffixed according to checkpoint config."""
    if not cfg.get("append_rdkit_definitions"):
        return str(definition).strip()
    if rdkit_lookup is None:
        rdkit_lookup = rdkit_definition_lookup_for_cfg(cfg)
    return append_rdkit_definition_values(
        target,
        definition,
        rdkit_lookup=rdkit_lookup,
        require_lookup=require_lookup,
        omit_base_definition=bool(cfg.get("omit_molt5_definitions")),
    )


def validate_vocabulary_definitions(
    words: Sequence[str], lookup: Mapping[str, str]
) -> None:
    missing = [str(w).strip() for w in words if str(w).strip() not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} vocabulary words missing from definitions "
            f"(e.g. {preview})"
        )


def embedding_provenance(
    *,
    model_name: Optional[str] = None,
    use_definition_embeds: bool = False,
    definitions_path: str = "",
    task: str = "semantle",
    append_rdkit_definitions: bool = False,
    omit_molt5_definitions: bool = False,
    rdkit_definitions_path: str = "",
    rdkit_definitions_map_path: str = "",
) -> dict[str, str | bool | int]:
    """Provenance fields for embed_cache.json and intervention_config.json."""
    cfg = task_config[task]
    prov: dict[str, str | bool | int] = {
        "sentence_transformer_model": model_name or task_embedding_model(task),
        "task": task,
        "embedding_prompt": cfg["embedding_prompt"],
        "use_definition_embeds": bool(use_definition_embeds),
        "definitions_path": (
            os.path.abspath(definitions_path) if definitions_path else ""
        ),
        "normalize_embeddings": EMBED_NORMALIZE,
    }
    if use_definition_embeds:
        prov["embedding_prompt_defn"] = cfg["embedding_prompt_defn"]
        prov["append_rdkit_definitions"] = bool(append_rdkit_definitions)
        prov["omit_molt5_definitions"] = bool(omit_molt5_definitions)
        if append_rdkit_definitions:
            from boreft.chem import RDKIT_DESCRIPTOR_SCHEMA_VERSION, file_sha256

            rdkit_path = os.path.abspath(
                rdkit_definitions_path or default_rdkit_definitions_path()
            )
            rdkit_map_path = os.path.abspath(
                rdkit_definitions_map_path or default_rdkit_definitions_map_path()
            )
            prov["rdkit_definitions_path"] = rdkit_path
            prov["rdkit_definitions_sha256"] = file_sha256(rdkit_path)
            prov["rdkit_definitions_map_path"] = rdkit_map_path
            prov["rdkit_definitions_map_sha256"] = file_sha256(rdkit_map_path)
            prov["rdkit_descriptor_schema_version"] = (
                RDKIT_DESCRIPTOR_SCHEMA_VERSION
            )
    return prov


def eval_embed_cache_provenance(
    *, model_name: Optional[str] = None, task: str = "semantle"
) -> dict[str, str | bool | int]:
    """Provenance for the prompt-based eval reference cache."""
    return embedding_provenance(
        model_name=model_name, use_definition_embeds=False, task=task
    )


def bias_network_embed_model_from_cfg(
    cfg: Mapping[str, object],
) -> Optional[str]:
    """Encoder a checkpoint's bias network was trained against, when overridden.

    ``None`` means the run used the task's own model, so callers fall through to
    :func:`embedding_model_name` and stay subject to the staleness check in
    :func:`warn_if_embedding_provenance_stale`.
    """
    model = cfg.get("bias_network_embed_model")
    return str(model) if model else None


def training_embed_cache_provenance_from_cfg(
    cfg: Mapping[str, object],
) -> dict[str, str | bool | int]:
    return embedding_provenance(
        model_name=bias_network_embed_model_from_cfg(cfg),
        use_definition_embeds=bool(cfg.get("use_definition_embeds")),
        definitions_path=str(cfg.get("definitions_path") or ""),
        task=str(cfg.get("task", "semantle")),
        append_rdkit_definitions=bool(cfg.get("append_rdkit_definitions")),
        omit_molt5_definitions=bool(cfg.get("omit_molt5_definitions")),
        rdkit_definitions_path=str(cfg.get("rdkit_definitions_path") or ""),
        rdkit_definitions_map_path=str(
            cfg.get("rdkit_definitions_map_path") or ""
        ),
    )


def definition_lookup_for_cfg(cfg: Mapping[str, object]) -> Optional[dict[str, str]]:
    if not cfg.get("use_definition_embeds"):
        return None
    path = cfg.get("definitions_path")
    if not path:
        raise ValueError(
            "Checkpoint has use_definition_embeds=true but definitions_path is missing"
        )
    return load_definition_embed_lookup(
        str(path),
        task=str(cfg.get("task", "semantle")),
        append_rdkit_definitions=bool(cfg.get("append_rdkit_definitions")),
        omit_molt5_definitions=bool(cfg.get("omit_molt5_definitions")),
        rdkit_definitions_path=str(cfg.get("rdkit_definitions_path") or "")
        or None,
    )


def training_cache_params(
    *,
    use_definition_embeds: bool,
    task: str,
    words: Sequence[str],
    definitions_path: Optional[str] = None,
    model_name: Optional[str] = None,
    append_rdkit_definitions: bool = False,
    omit_molt5_definitions: bool = False,
    rdkit_definitions_path: Optional[str] = None,
    rdkit_definitions_map_path: Optional[str] = None,
) -> tuple[dict[str, str | bool | int], Optional[dict[str, str]], str]:
    """Return (train_provenance, definition_lookup, definitions_path_abs).

    ``model_name`` names the encoder for the *training* cache only, which is what
    the bias network reads; the eval reference cache is built from the task's own
    model so similarity metrics stay comparable across runs.
    """
    if not use_definition_embeds:
        return embedding_provenance(model_name=model_name, task=task), None, ""
    if not task_supports_definition_embeds(task):
        raise ValueError(
            f"--use-definition-embeds needs an 'embedding_prompt_defn' template in "
            f"task_config for task {task!r}"
        )
    path_abs = os.path.abspath(definitions_path or default_definitions_path(task))
    lookup = load_definition_embed_lookup(
        path_abs,
        task=task,
        append_rdkit_definitions=append_rdkit_definitions,
        omit_molt5_definitions=omit_molt5_definitions,
        rdkit_definitions_path=rdkit_definitions_path,
    )
    validate_vocabulary_definitions(words, lookup)
    return (
        embedding_provenance(
            model_name=model_name,
            use_definition_embeds=True,
            definitions_path=path_abs,
            task=task,
            append_rdkit_definitions=append_rdkit_definitions,
            omit_molt5_definitions=omit_molt5_definitions,
            rdkit_definitions_path=rdkit_definitions_path or "",
            rdkit_definitions_map_path=rdkit_definitions_map_path or "",
        ),
        lookup,
        path_abs,
    )


_EVAL_PROVENANCE_KEYS = (
    "sentence_transformer_model",
    "task",
    "embedding_prompt",
    "normalize_embeddings",
)

_TRAIN_DEFINITION_PROVENANCE_KEYS = (
    "sentence_transformer_model",
    "task",
    "embedding_prompt",
    "embedding_prompt_defn",
    "append_rdkit_definitions",
    "omit_molt5_definitions",
    "normalize_embeddings",
)

_TRAIN_RDKIT_PROVENANCE_KEYS = (
    "rdkit_definitions_path",
    "rdkit_definitions_sha256",
    "rdkit_definitions_map_path",
    "rdkit_definitions_map_sha256",
    "rdkit_descriptor_schema_version",
)


def _provenance_keys_for_cfg(saved_cfg: dict) -> tuple[str, ...]:
    if saved_cfg.get("use_definition_embeds"):
        if saved_cfg.get("append_rdkit_definitions"):
            return (
                _TRAIN_DEFINITION_PROVENANCE_KEYS
                + _TRAIN_RDKIT_PROVENANCE_KEYS
            )
        return _TRAIN_DEFINITION_PROVENANCE_KEYS
    return _EVAL_PROVENANCE_KEYS


def warn_if_embedding_provenance_stale(
    saved_cfg: Optional[dict], *, context: str = "eval"
) -> None:
    """Warn when checkpoint embedding settings differ from current code.

    The provenance a checkpoint stores describes its *training* cache, which is the
    eval cache too unless the run built them separately — definition embeds or a
    bias-network encoder override. Comparing against the wrong one of those two
    would report a mismatch for every such run.
    """
    import warnings

    if not saved_cfg:
        return
    task = str(saved_cfg.get("task", "semantle"))
    if saved_cfg.get("use_definition_embeds") or saved_cfg.get(
        "bias_network_embed_model"
    ):
        expected = training_embed_cache_provenance_from_cfg(saved_cfg)
        detail = (
            "embed_sim and/or training embed_cache metrics may not match "
            "this checkpoint"
        )
    else:
        expected = eval_embed_cache_provenance(task=task)
        detail = "embed_sim metrics may not match training-time logs for this checkpoint"
    mismatches: list[str] = []
    for key in _provenance_keys_for_cfg(saved_cfg):
        want = expected[key]
        got = saved_cfg.get(key)
        if got is not None and got != want:
            mismatches.append(f"{key}: checkpoint={got!r} current={want!r}")
    if mismatches:
        warnings.warn(
            f"[{context}] Checkpoint embedding provenance differs from current code "
            f"({'; '.join(mismatches)}). {detail}.",
            stacklevel=3,
        )


def _get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBED_MODEL_CACHE:
        print(
            f"[text_similarity] Loading sentence-transformers '{model_name}'...",
            flush=True,
        )
        try:
            _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except AttributeError as e:
            if "qwen3" not in str(e).lower():
                raise
            import transformers

            raise ImportError(
                "Qwen3-Embedding needs transformers>=4.51.0 (Qwen3 was added "
                f"in 4.51); this env has {transformers.__version__}. "
                "PyTDC extras pin transformers<4.51. Restore with: "
                "pip install transformers==4.57.6 && pip install PyTDC --no-deps"
            ) from e
        print(
            f"[text_similarity] Sentence-transformers '{model_name}' ready.",
            flush=True,
        )
    return _EMBED_MODEL_CACHE[model_name]


def _encode(
    texts: Sequence[str], *, task: str, model_name: Optional[str] = None
) -> np.ndarray:
    """Encode already-decorated strings; normalized rows.

    ``model_name`` overrides the task's encoder. It exists for the bias network,
    which can be trained against a different embedding space than the one the
    similarity metrics are computed in (``--bias-network-embed-model``).
    """
    payload = list(texts)
    if not payload:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_embed_model(model_name or task_embedding_model(task))
    return np.asarray(
        model.encode(payload, normalize_embeddings=EMBED_NORMALIZE),
        dtype=np.float32,
    )


def format_text_for_embedding(text: str, *, task: str = "semantle") -> str:
    """Decorate target text with the task's ``embedding_prompt`` (embed_sim)."""
    return substitute_placeholders(
        embedding_prompt_template(task), text=str(text).strip()
    )


def format_text_for_cache_embedding(
    text: str,
    definition_lookup: Optional[Mapping[str, str]] = None,
    *,
    task: str = "semantle",
) -> str:
    """Text to embed when building embed_cache (optional definition lookup)."""
    word = str(text).strip()
    if definition_lookup is not None and word in definition_lookup:
        return definition_lookup[word]
    return format_text_for_embedding(word, task=task)


def _encode_normalized(
    texts: Sequence[str],
    *,
    definition_lookup: Optional[Mapping[str, str]] = None,
    task: str = "semantle",
    model_name: Optional[str] = None,
) -> np.ndarray:
    if definition_lookup is None:
        decorated = [format_text_for_embedding(t, task=task) for t in texts]
    else:
        decorated = [
            format_text_for_cache_embedding(t, definition_lookup, task=task)
            for t in texts
        ]
    return _encode(decorated, task=task, model_name=model_name).astype(np.float32)


def encode_texts_normalized(texts: Sequence[str], *, task: str = "semantle") -> np.ndarray:
    """Encode targets with the task's embedding model; L2-normalized rows."""
    return _encode_normalized(texts, task=task)


def encode_texts_as_is(
    texts: Sequence[str], *, task: str = "semantle", model_name: Optional[str] = None
) -> np.ndarray:
    """Encode strings with no prompt or definition decoration.

    Each entry is passed verbatim to ``model_name``, or to the task's embedding
    model when that is not given.
    """
    payload = [str(t) for t in texts]
    if not payload:
        return np.zeros((0, 0), dtype=np.float64)
    return _encode(payload, task=task, model_name=model_name).astype(np.float64)


def encode_reference_embeddings(
    targets: Sequence[str],
    *,
    batch_size: int = 64,
    definition_lookup: Optional[Mapping[str, str]] = None,
    task: str = "semantle",
    model_name: Optional[str] = None,
) -> np.ndarray:
    """Batch-encode vocabulary targets for embed_cache; L2-normalized rows."""
    texts = [str(t).strip() for t in targets]
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunks.append(
            _encode_normalized(
                texts[i : i + batch_size],
                definition_lookup=definition_lookup,
                task=task,
                model_name=model_name,
            )
        )
    return np.concatenate(chunks, axis=0).astype(np.float64)


def embedding_sim_per_text(
    targets: list[str], generated: list[str], *, task: str = "semantle"
) -> np.ndarray:
    """Cosine similarity between target and generated text embeddings. Shape [N]."""
    emb_tgt = _encode_normalized(targets, task=task)
    emb_gen = _encode_normalized(generated, task=task)
    return np.sum(emb_tgt * emb_gen, axis=1)


def pairwise_embedding_similarity_matrix(
    words: list[str], *, task: str = "semantle"
) -> torch.Tensor:
    """Full pairwise cosine similarities; shape [len(words), len(words)], CPU."""
    if len(words) == 0:
        return torch.zeros(0, 0)
    e = _encode_normalized(words, task=task)
    return torch.from_numpy(e @ e.T)
