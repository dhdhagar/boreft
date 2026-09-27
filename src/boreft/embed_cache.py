"""Target embedding cache: embed_cache.pt + embed_cache.json under repo-root embed_cache/."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from boreft.data import ReftItem
from boreft.text_similarity import (
    EMBED_NORMALIZE,
    embedding_provenance,
    eval_embed_cache_provenance,
    encode_reference_embeddings,
)

PROVENANCE_FIELD_NAMES = (
    "sentence_transformer_model",
    "task",
    "embedding_prompt",
    "embedding_prompt_defn",
    "use_definition_embeds",
    "definitions_path",
    "append_rdkit_definitions",
    "omit_molt5_definitions",
    "rdkit_definitions_path",
    "rdkit_definitions_sha256",
    "rdkit_definitions_map_path",
    "rdkit_definitions_map_sha256",
    "rdkit_descriptor_schema_version",
    "normalize_embeddings",
)

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

EMBED_CACHE_DIR_NAME = "embed_cache"
EMBED_CACHE_LOCKS_DIRNAME = ".locks"
EMBED_CACHE_PT_FILENAME = "embed_cache.pt"
EMBED_CACHE_INDEX_FILENAME = "embed_cache.json"
ENV_EMBED_CACHE_DIR = "BOREFT_EMBED_CACHE_DIR"


def repo_root() -> str:
    """Walk upward from this package until ``pyproject.toml`` is found."""
    path = os.path.abspath(__file__)
    for _ in range(8):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_embed_cache_dir() -> str:
    override = os.environ.get(ENV_EMBED_CACHE_DIR)
    if override:
        return os.path.abspath(override)
    return os.path.join(repo_root(), EMBED_CACHE_DIR_NAME)


def embed_texts_from_items(items: Sequence[ReftItem]) -> List[str]:
    """Vocabulary strings to encode (raw word, not chat-formatted target)."""
    return [str(getattr(it, "_raw_word", it.target)).strip() for it in items]


def expected_texts_for_rows(
    texts: Sequence[str],
    *,
    num_rows: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> List[str]:
    """Row-aligned text list for a cache slice (matches ``load_embed_cache_tensor``)."""
    if indices is not None:
        return [str(texts[int(i)]).strip() for i in indices]
    if num_rows is not None:
        return [str(t).strip() for t in texts[:num_rows]]
    return [str(t).strip() for t in texts]


def compute_vocab_key(
    texts: Sequence[str],
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "texts": [str(t).strip() for t in texts],
        **(provenance or embedding_provenance()),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_entry_dir(vocab_key: str, cache_dir: Optional[str] = None) -> str:
    base = cache_dir or default_embed_cache_dir()
    return os.path.join(base, vocab_key)


def _vocab_lock_path(vocab_key: str, cache_dir: str) -> str:
    locks_dir = os.path.join(cache_dir, EMBED_CACHE_LOCKS_DIRNAME)
    os.makedirs(locks_dir, exist_ok=True)
    return os.path.join(locks_dir, f"{vocab_key}.lock")


@contextlib.contextmanager
def _vocab_cache_lock(vocab_key: str, cache_dir: str):
    """Exclusive lock while checking/building one vocabulary cache entry."""
    lock_path = _vocab_lock_path(vocab_key, cache_dir)
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        if _HAS_FCNTL:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if _HAS_FCNTL:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()


def sibling_index_path(pt_path: str) -> str:
    parent = os.path.dirname(pt_path)
    return os.path.join(parent, EMBED_CACHE_INDEX_FILENAME)


def load_index(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_index_for_pt(pt_path: str, *, required: bool = True) -> Dict[str, Any]:
    """Load sibling ``embed_cache.json``; fail when ``required`` and missing."""
    index_path = sibling_index_path(pt_path)
    if not os.path.isfile(index_path):
        if required:
            raise FileNotFoundError(
                f"embed_cache requires sibling {EMBED_CACHE_INDEX_FILENAME} "
                f"next to {pt_path}"
            )
        return {}
    return load_index(index_path)


def provenance_from_index(index: Dict[str, Any]) -> Dict[str, Any]:
    return {k: index[k] for k in PROVENANCE_FIELD_NAMES if k in index}


def validate_index_provenance(
    index: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Check embedding provenance fields only (not the full text list)."""
    expected_prov = provenance or provenance_from_index(index) or embedding_provenance()
    for key, want in expected_prov.items():
        got = index.get(key)
        if got is None:
            return f"missing provenance field {key!r}"
        if got != want:
            return f"{key}: cache={got!r} expected={want!r}"
    return None


def validate_index(
    index: Dict[str, Any],
    texts: Sequence[str],
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a mismatch reason, or ``None`` when the cache entry is usable."""
    expected_prov = provenance or embedding_provenance()
    texts_norm = [str(t).strip() for t in texts]
    index_texts = index.get("texts")
    if index_texts is None:
        return "index missing texts"
    if list(index_texts) != texts_norm:
        return "texts differ"
    reason = validate_index_provenance(index, expected_prov)
    if reason is not None:
        return reason
    vocab_key = compute_vocab_key(texts_norm, expected_prov)
    if index.get("vocab_key") and index["vocab_key"] != vocab_key:
        return "vocab_key mismatch"
    return None


def _read_embed_cache_pt(path: str) -> torch.Tensor:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "embeddings" in blob:
        t = blob["embeddings"]
    else:
        raise ValueError(f"Unrecognized embed cache format in {path}")
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    if t.dim() != 2:
        raise ValueError(
            f"embed_cache must be 2D [num_rows, dim], got shape {tuple(t.shape)}"
        )
    return t


def _slice_rows(
    t: torch.Tensor,
    *,
    num_rows: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if indices is not None:
        idx = [int(i) for i in indices]
        if not idx:
            raise ValueError("indices must be non-empty when provided")
        if min(idx) < 0 or max(idx) >= t.shape[0]:
            raise ValueError(
                f"embed_cache row indices out of range for {t.shape[0]} rows: "
                f"min={min(idx)}, max={max(idx)}"
            )
        return t[idx].float().contiguous()
    if num_rows is None:
        return t.float().contiguous()
    if t.shape[0] < num_rows:
        raise ValueError(
            f"embed_cache has {t.shape[0]} rows but expected at least {num_rows}"
        )
    return t[:num_rows].float().contiguous()


def _maybe_normalize(
    t: torch.Tensor, *, normalize: Optional[bool] = None
) -> torch.Tensor:
    # SentenceTransformer L2-normalizes when enabled; re-applying on load is
    # idempotent and uses the index flag when present.
    flag = EMBED_NORMALIZE if normalize is None else bool(normalize)
    if flag:
        return F.normalize(t, dim=-1, eps=1e-12)
    return t


def _validate_loaded_rows(
    index: Dict[str, Any],
    *,
    expected_texts: Sequence[str],
    num_rows: Optional[int],
    indices: Optional[Sequence[int]],
    path: str,
) -> None:
    if indices is not None:
        file_texts = [index["texts"][i] for i in indices]
    elif num_rows is not None:
        file_texts = index["texts"][:num_rows]
    else:
        file_texts = index["texts"]
    want = [str(t).strip() for t in expected_texts]
    if list(file_texts) != want:
        raise ValueError(
            f"embed_cache texts at {path} do not match expected vocabulary "
            f"({len(want)} rows)"
        )
    stored_prov = provenance_from_index(index)
    if stored_prov and index.get("vocab_key"):
        if compute_vocab_key(index["texts"], stored_prov) != index["vocab_key"]:
            raise ValueError(f"embed_cache vocab_key integrity check failed for {path}")


def load_embed_cache_tensor(
    path: str,
    num_rows: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    *,
    expected_texts: Optional[Sequence[str]] = None,
    index: Optional[Dict[str, Any]] = None,
    require_index: bool = True,
    return_index: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Load embeddings from embed_cache.pt, optionally subselecting rows."""
    t = _read_embed_cache_pt(path)
    if index is None:
        index = load_index_for_pt(path, required=require_index)
    elif require_index and not index:
        raise FileNotFoundError(
            f"embed_cache requires sibling {EMBED_CACHE_INDEX_FILENAME} next to {path}"
        )
    if expected_texts is not None:
        if not index or "texts" not in index:
            raise FileNotFoundError(
                f"expected_texts provided but no index found for {path}"
            )
        _validate_loaded_rows(
            index,
            expected_texts=expected_texts,
            num_rows=num_rows,
            indices=indices,
            path=path,
        )
    normalize = None
    if index:
        if "normalize_embeddings" in index:
            normalize = bool(index["normalize_embeddings"])
    out = _maybe_normalize(
        _slice_rows(t, num_rows=num_rows, indices=indices),
        normalize=normalize,
    )
    if return_index:
        return out, index or {}
    return out


def embed_cache_tensor_to_numpy(
    tensor: torch.Tensor,
    *,
    num_rows: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Slice an in-memory embed cache tensor and return float64 numpy rows."""
    return (
        _slice_rows(tensor, num_rows=num_rows, indices=indices)
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def save_embed_cache(
    entry_dir: str,
    embeddings: torch.Tensor,
    texts: Sequence[str],
    *,
    provenance: Optional[Dict[str, Any]] = None,
    sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write embed_cache.pt + embed_cache.json atomically; return the index dict."""
    texts_norm = [str(t).strip() for t in texts]
    prov = dict(provenance or embedding_provenance())
    vocab_key = compute_vocab_key(texts_norm, prov)
    ref = embeddings.detach().cpu().float().contiguous()
    if ref.shape[0] != len(texts_norm):
        raise ValueError(
            f"embeddings rows {ref.shape[0]} != texts {len(texts_norm)}"
        )

    index: Dict[str, Any] = {
        "vocab_key": vocab_key,
        "num_rows": int(ref.shape[0]),
        "embed_dim": int(ref.shape[1]),
        "texts": texts_norm,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **prov,
    }
    if sources:
        index["sources"] = sources

    os.makedirs(entry_dir, exist_ok=True)
    tmp_dir = f"{entry_dir}.tmp.{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        tmp_pt = os.path.join(tmp_dir, EMBED_CACHE_PT_FILENAME)
        tmp_json = os.path.join(tmp_dir, EMBED_CACHE_INDEX_FILENAME)
        torch.save(
            {
                "embeddings": ref,
                "num_rows": int(ref.shape[0]),
                "embed_dim": int(ref.shape[1]),
            },
            tmp_pt,
        )
        final_pt = os.path.join(entry_dir, EMBED_CACHE_PT_FILENAME)
        index["cache_pt"] = os.path.abspath(final_pt)
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp_pt, final_pt)
        os.replace(tmp_json, os.path.join(entry_dir, EMBED_CACHE_INDEX_FILENAME))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        f"[embed_cache] Saved {tuple(ref.shape)} to {index['cache_pt']}",
        flush=True,
    )
    return index


def bias_network_input_dim_from_cfg(
    saved_cfg: Optional[Dict[str, Any]],
    *,
    embed_cache_path: Optional[str] = None,
) -> Optional[int]:
    """ST-row bias-network MLP width without loading a sentence transformer.

    Train writes ``bias_network_embed_dim`` from the frozen cache width.
    ``bias_network_input_dim`` is the llm_encoder / explicit-override field.
    An on-disk ``embed_cache.pt`` is the last resort so decode-only loads
    (``skip_embed_cache``) can size the MLP without Qwen3.
    """
    if saved_cfg:
        for key in ("bias_network_input_dim", "bias_network_embed_dim"):
            val = saved_cfg.get(key)
            if val is not None:
                return int(val)
    if embed_cache_path and os.path.isfile(embed_cache_path):
        return int(_read_embed_cache_pt(embed_cache_path).shape[1])
    return None


def resolve_embed_cache_path(
    output_dir: str,
    explicit: Optional[str] = None,
    saved_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve training embed_cache.pt (definition-based when configured)."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"embed_cache path does not exist: {explicit}")
        load_index_for_pt(explicit, required=True)
        return os.path.abspath(explicit)
    cfg = saved_cfg
    if cfg is None:
        cfg_path = os.path.join(output_dir, "intervention_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
    if cfg:
        p = cfg.get("embed_cache_path")
        if p and os.path.isfile(p):
            load_index_for_pt(p, required=True)
            return os.path.abspath(p)
    fallback = os.path.join(output_dir, EMBED_CACHE_PT_FILENAME)
    if os.path.isfile(fallback):
        load_index_for_pt(fallback, required=True)
        return os.path.abspath(fallback)
    return None


def resolve_eval_embed_cache_path(
    output_dir: str,
    explicit: Optional[str] = None,
    saved_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve prompt-based eval reference embed_cache.pt."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"embed_cache path does not exist: {explicit}")
        load_index_for_pt(explicit, required=True)
        return os.path.abspath(explicit)
    cfg = saved_cfg
    if cfg is None:
        cfg_path = os.path.join(output_dir, "intervention_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
    if cfg:
        p = cfg.get("eval_embed_cache_path")
        if p and os.path.isfile(p):
            load_index_for_pt(p, required=True)
            return os.path.abspath(p)
        if not cfg.get("use_definition_embeds"):
            p = cfg.get("embed_cache_path")
            if p and os.path.isfile(p):
                load_index_for_pt(p, required=True)
                return os.path.abspath(p)
    return None


def resolve_or_build_embed_cache(
    texts: Sequence[str],
    *,
    explicit_path: Optional[str] = None,
    indices: Optional[Sequence[int]] = None,
    sources: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    definition_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """Load a matching cache entry or encode, save under embed_cache/, and return."""
    texts_norm = [str(t).strip() for t in texts]
    if not texts_norm:
        raise ValueError("resolve_or_build_embed_cache requires non-empty texts")
    provenance = dict(provenance or eval_embed_cache_provenance())
    cache_dir = cache_dir or default_embed_cache_dir()

    if explicit_path and os.path.isfile(explicit_path):
        index = load_index_for_pt(explicit_path, required=True)
        tensor = load_embed_cache_tensor(
            explicit_path,
            num_rows=len(texts_norm) if indices is None else None,
            indices=indices,
            expected_texts=texts_norm,
            index=index,
            require_index=True,
        )
        print(f"[embed_cache] Loaded {explicit_path}", flush=True)
        return tensor, os.path.abspath(explicit_path), index

    vocab_key = compute_vocab_key(texts_norm, provenance)
    entry_dir = cache_entry_dir(vocab_key, cache_dir)
    pt_path = os.path.join(entry_dir, EMBED_CACHE_PT_FILENAME)
    json_path = os.path.join(entry_dir, EMBED_CACHE_INDEX_FILENAME)

    with _vocab_cache_lock(vocab_key, cache_dir):
        if os.path.isfile(pt_path) and os.path.isfile(json_path):
            index = load_index(json_path)
            reason = validate_index(index, texts_norm, provenance)
            if reason is None:
                tensor = load_embed_cache_tensor(
                    pt_path,
                    expected_texts=texts_norm,
                    index=index,
                    require_index=True,
                )
                print(
                    f"[embed_cache] Hit {entry_dir} ({len(texts_norm)} rows)",
                    flush=True,
                )
                return tensor, os.path.abspath(pt_path), index
            print(
                f"[embed_cache] Stale cache at {entry_dir}: {reason}; rebuilding",
                flush=True,
            )

        print(
            f"[embed_cache] Encoding {len(texts_norm)} targets with "
            f"{provenance['sentence_transformer_model']!r} ...",
            flush=True,
        )
        # Encode with the model the provenance names, not the task default: the two
        # differ when the bias network was given its own encoder, and labelling a
        # cache with a model that did not produce it would be worse than useless.
        ref_np = encode_reference_embeddings(
            texts_norm,
            batch_size=batch_size,
            definition_lookup=definition_lookup,
            task=str(provenance.get("task", "semantle")),
            model_name=str(provenance.get("sentence_transformer_model") or "") or None,
        )
        tensor = torch.from_numpy(ref_np).float()
        index = save_embed_cache(
            entry_dir,
            tensor,
            texts_norm,
            provenance=provenance,
            sources=sources,
        )
        tensor = _maybe_normalize(
            tensor,
            normalize=bool(provenance.get("normalize_embeddings", EMBED_NORMALIZE)),
        )
        return tensor, os.path.abspath(pt_path), index
