#!/usr/bin/env python
"""Compare Qwen, Qwen3-1.7B, Llama, and MiST clustering on molopt text.

Two input strings, scored separately:

  smiles_defn    ``embedding_prompt_defn``:
                 ``The molecule '{SMILES}' is: {definition}``
  smiles_prompt  ``embedding_prompt`` (no definition):
                 ``The description for molecule '{SMILES}'.``

Encoders:

  qwen       ``Qwen/Qwen3-Embedding-0.6B`` via sentence-transformers
  qwen_llm   ``Qwen/Qwen3-1.7B`` last-layer last-token (causal LM)
  llama      Llama-3.2-1B-Instruct last-layer. On ``smiles_defn`` this is the
             bias-network ``last_instruction`` pool. On ``smiles_prompt`` there
             is no definition span, so it is the last non-pad token.
  mist       MiST chemistry Qwen2.5-3B (local checkpoint, not a Hub id),
             last-layer last non-pad token. Looks for
             ``Qwen2.5-3B_pretrained-v4-cot`` under ``$HF_HOME/mist``, then the
             ``preferred`` path in ``mist_models.json``.

``--max-length`` (default 128, same as ``--bias-encoder-max-length``) truncates
every encoder.

  python scripts/analyze_molopt_encoder_cluster.py --n 0
  python scripts/analyze_molopt_encoder_cluster.py --variants smiles_prompt
  python scripts/analyze_molopt_encoder_cluster.py --reuse-previous
  sbatch scripts/analyze_molopt_encoder_cluster.sh

Non-default flags are appended to output names. The prompt-only run is
``encoder_cluster_prompt.json`` so it does not overwrite the definition run.
Encoder embeddings are written to ``encoder_cluster_{name}_emb*.npy``;
``--reuse-previous`` reloads compatible caches instead of re-encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Callable, Optional, Sequence

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
for _path in (_SCRIPT_DIR, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_molopt_text_variants as mtv  # noqa: E402
from boreft.bo.plotting import save_plot  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_encoder_cluster  # noqa: E402
from boreft.task_config import task_embedding_model  # noqa: E402
from boreft.text_similarity import (  # noqa: E402
    default_definitions_path,
    default_rdkit_definitions_path,
    format_text_for_embedding,
)

TASK = "molopt"
VARIANT_DEFN = "smiles_defn"
VARIANT_PROMPT = "smiles_prompt"
VARIANT = VARIANT_DEFN  # definition run; kept for cache metadata and Semantle imports
VARIANT_NAMES = (VARIANT_DEFN, VARIANT_PROMPT)
# Semantle imports this three-encoder list. Molopt adds MiST on top.
ENCODER_NAMES = ("qwen", "qwen_llm", "llama")
MOLOPT_ENCODER_NAMES = ENCODER_NAMES + ("mist",)
DEFAULT_QWEN_MODEL = task_embedding_model(TASK)
DEFAULT_QWEN_LLM_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_MIST_MODEL = "Qwen2.5-3B_pretrained-v4-cot"
DEFAULT_CACHE_DIR = None
DEFAULT_POOLING = "last_instruction"
DEFAULT_MAX_LENGTH = 128
DEFAULT_QWEN_BATCH_SIZE = 64
DEFAULT_QWEN_LLM_BATCH_SIZE = 16
DEFAULT_LLAMA_BATCH_SIZE = 16
DEFAULT_MIST_BATCH_SIZE = 4

EncodeFn = Callable[..., np.ndarray]
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def selected_encoders(
    names: Sequence[str], catalog: Optional[Sequence[str]] = None
) -> list[str]:
    allowed = tuple(catalog) if catalog is not None else ENCODER_NAMES
    if not names:
        return list(allowed)
    chosen: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in allowed:
            raise ValueError(f"unknown encoder {name!r}")
        if name in seen:
            continue
        seen.add(name)
        chosen.append(name)
    return chosen


def selected_variants(names: Sequence[str]) -> list[str]:
    if not names:
        return list(VARIANT_NAMES)
    chosen: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in VARIANT_NAMES:
            raise ValueError(f"unknown variant {name!r}")
        if name in seen:
            continue
        seen.add(name)
        chosen.append(name)
    return chosen


def _catalog_for(args: argparse.Namespace) -> tuple[str, ...]:
    """Molopt argparse has ``mist_model``; Semantle reuses the three-encoder list."""
    if hasattr(args, "mist_model"):
        return MOLOPT_ENCODER_NAMES
    return ENCODER_NAMES


def _slug(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_slug(item) for item in value)
    text = str(value).strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    text = _SLUG_RE.sub("-", text).strip("-.")
    return text or "x"


def _flag_tag(comparisons: Sequence[tuple[str, object, object]]) -> str:
    parts: list[str] = []
    for label, value, default in comparisons:
        if value == default:
            continue
        if label in ("keepunknown", "prompt"):
            parts.append(label)
            continue
        if label in ("n", "seed", "maxlen"):
            parts.append(f"{label}{value}")
            continue
        parts.append(f"{label}-{_slug(value)}")
    return ("_" + "_".join(parts)) if parts else ""


def artifact_tag(args: argparse.Namespace, variant: Optional[str] = None) -> str:
    """Filename suffix for flags that differ from argparse defaults.

    Paths, device, cache dir, batch sizes, and ``--reuse-previous`` are omitted
    so a cluster default run still writes ``encoder_cluster.json``. The
    prompt-only variant adds ``_prompt``.
    """
    catalog = _catalog_for(args)
    comparisons: list[tuple[str, object, object]] = [
        ("n", args.n, mtv.DEFAULT_N),
        ("seed", args.seed, 0),
        ("label", args.label_field, "category_normalized"),
        ("keepunknown", args.keep_unknown, False),
        ("encoders", selected_encoders(args.encoders, catalog), list(catalog)),
        ("qwen", args.qwen_model, DEFAULT_QWEN_MODEL),
        ("qwenllm", args.qwen_llm_model, DEFAULT_QWEN_LLM_MODEL),
        ("llama", args.llama_model, DEFAULT_LLAMA_MODEL),
    ]
    if hasattr(args, "mist_model"):
        comparisons.append(("mist", args.mist_model, DEFAULT_MIST_MODEL))
    comparisons.extend(
        [
            ("pooling", args.pooling, DEFAULT_POOLING),
            ("maxlen", args.max_length, DEFAULT_MAX_LENGTH),
        ]
    )
    if variant not in (None, VARIANT_DEFN):
        comparisons.append(("prompt", True, False))
    return _flag_tag(comparisons)


def cache_tag(args: argparse.Namespace, variant: Optional[str] = None) -> str:
    """Sample-level suffix shared by encoder embedding caches.

    Omits ``--encoders`` and model ids so a qwen-only run can be reused by a
    later run. Model and pooling are checked in the sidecar. The prompt-only
    variant adds ``_prompt`` so it does not clobber definition caches.
    """
    comparisons: list[tuple[str, object, object]] = [
        ("n", args.n, mtv.DEFAULT_N),
        ("seed", args.seed, 0),
        ("label", args.label_field, "category_normalized"),
        ("keepunknown", args.keep_unknown, False),
        ("maxlen", args.max_length, DEFAULT_MAX_LENGTH),
    ]
    if variant not in (None, VARIANT_DEFN):
        comparisons.append(("prompt", True, False))
    return _flag_tag(comparisons)


def artifact_path(out_dir: str, stem: str, tag: str, ext: str) -> str:
    return os.path.join(out_dir, f"{stem}{tag}.{ext}")


def encoder_cache_paths(out_dir: str, name: str, tag: str) -> tuple[str, str]:
    return (
        artifact_path(out_dir, f"encoder_cluster_{name}_emb", tag, "npy"),
        artifact_path(out_dir, f"encoder_cluster_{name}_cache", tag, "json"),
    )


def texts_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def encoder_model_id(args: argparse.Namespace, name: str) -> str:
    if name == "qwen":
        return args.qwen_model
    if name == "qwen_llm":
        return args.qwen_llm_model
    if name == "mist":
        resolved = getattr(args, "mist_resolved", None)
        if resolved:
            return os.path.basename(str(resolved).rstrip(os.sep))
        return args.mist_model
    return args.llama_model


def encoder_pooling(
    args: argparse.Namespace, name: str, variant: Optional[str] = None
) -> Optional[str]:
    if name != "llama":
        return None
    # The prompt string has no {definition} span, so instruction-mask pooling
    # would look for tokens that are not in the input.
    if variant == VARIANT_PROMPT:
        return "last_token"
    return args.pooling


def texts_for_variant(molecules, variant: str) -> list[str]:
    if variant == VARIANT_DEFN:
        return [mtv.variant_text(mol, VARIANT_DEFN) for mol in molecules]
    if variant == VARIANT_PROMPT:
        return [format_text_for_embedding(mol.smiles, task=TASK) for mol in molecules]
    raise ValueError(f"unknown variant {variant!r}")


def _is_mist_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _mist_search_roots() -> list[str]:
    roots: list[str] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "mist"))
    roots.append(os.path.join(os.getcwd(), "models", "mist"))
    roots.append(os.path.join(_REPO_ROOT, "models", "mist"))
    seen: set[str] = set()
    unique: list[str] = []
    for root in roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def _find_named_checkpoint(root: str, name: str, max_depth: int = 4) -> Optional[str]:
    """Return ``root``/``name`` or a nested directory of that name with config.json."""
    if not os.path.isdir(root):
        return None
    direct = os.path.join(root, name)
    if _is_mist_checkpoint(direct):
        return direct
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames.clear()
            continue
        if os.path.basename(dirpath) == name and "config.json" in filenames:
            return dirpath
    return None


def _preferred_checkpoint(roots: Sequence[str]) -> Optional[str]:
    for root in roots:
        manifest = os.path.join(root, "mist_models.json")
        if not os.path.isfile(manifest):
            continue
        with open(manifest, encoding="utf-8") as handle:
            preferred = json.load(handle).get("preferred")
        if not preferred:
            continue
        path = os.path.abspath(os.path.expanduser(str(preferred)))
        if _is_mist_checkpoint(path):
            return path
    return None


def resolve_mist_model(spec: str) -> str:
    """Return a local MiST checkpoint directory (one that contains config.json).

    ``spec`` may be that directory. Otherwise it is a folder name. The default
    name ``Qwen2.5-3B_pretrained-v4-cot`` is tried under ``$HF_HOME/mist``
    (including one nested level, as in ``mist/models/<name>``). If that name is
    missing, ``mist_models.json`` ``preferred`` is used. The Figshare zip on
    this cluster extracted ``qwen_pretranined_v6``, not v4-cot.
    """
    raw = os.path.expanduser(spec.strip())
    direct = os.path.abspath(raw)
    if os.path.isdir(direct):
        if _is_mist_checkpoint(direct):
            return direct
        raise FileNotFoundError(
            f"MiST path {direct} exists but has no config.json"
        )
    name = os.path.basename(raw.rstrip("/"))
    roots = _mist_search_roots()
    for root in roots:
        found = _find_named_checkpoint(root, name)
        if found:
            return found
    preferred = _preferred_checkpoint(roots)
    if preferred is not None and name == DEFAULT_MIST_MODEL:
        print(
            f"[analyze] MiST checkpoint {spec!r} not found; "
            f"using mist_models.json preferred: {preferred}",
            flush=True,
        )
        return preferred
    hint = f" manifest preferred={preferred}." if preferred else ""
    raise FileNotFoundError(
        f"MiST checkpoint {spec!r} not found under: {', '.join(roots)}.{hint} "
        "Pass --mist-model with the local path in mist_models.json."
    )


def prepare_mist_checkpoint(
    args: argparse.Namespace,
    encoders: Sequence[str],
    encode_fns: Optional[dict[str, EncodeFn]],
) -> None:
    """Resolve MiST before any encoding so a missing checkpoint fails immediately."""
    if encode_fns is not None or "mist" not in encoders:
        return
    path = resolve_mist_model(args.mist_model)
    args.mist_resolved = path
    print(f"[analyze] MiST checkpoint: {path}", flush=True)


def encoder_cache_compatible(
    meta: object,
    args: argparse.Namespace,
    name: str,
    smiles: Sequence[str],
    texts: Sequence[str],
    *,
    variant: str = VARIANT,
) -> Optional[str]:
    """Return a short reason if ``meta`` cannot be reused, else ``None``."""
    if not isinstance(meta, dict):
        return "invalid metadata"
    if meta.get("encoder") != name:
        return f"encoder {meta.get('encoder')!r} != {name!r}"
    if meta.get("variant") != variant:
        return "variant mismatch"
    if meta.get("model") != encoder_model_id(args, name):
        return "model mismatch"
    if int(meta.get("max_length", -1)) != int(args.max_length):
        return "max_length mismatch"
    pooling = encoder_pooling(args, name, variant)
    if pooling is not None and meta.get("pooling") != pooling:
        return "pooling mismatch"
    if list(meta.get("smiles") or []) != list(smiles):
        return "molecule sample mismatch"
    if meta.get("texts_sha256") != texts_digest(texts):
        return "input text mismatch"
    shape = meta.get("embedding_shape") or []
    if len(shape) != 2 or int(shape[0]) != len(smiles):
        return "embedding shape mismatch"
    return None


def save_encoder_cache(
    out_dir: str,
    tag: str,
    name: str,
    args: argparse.Namespace,
    emb: np.ndarray,
    smiles: Sequence[str],
    labels: Sequence[str],
    texts: Sequence[str],
    n_truncated: int,
    *,
    variant: str = VARIANT,
) -> tuple[str, str]:
    emb_path, meta_path = encoder_cache_paths(out_dir, name, tag)
    np.save(emb_path, np.asarray(emb, dtype=np.float32))
    meta = {
        "encoder": name,
        "variant": variant,
        "model": encoder_model_id(args, name),
        "pooling": encoder_pooling(args, name, variant),
        "max_length": int(args.max_length),
        "n_requested": int(args.n),
        "seed": int(args.seed),
        "label_field": args.label_field,
        "keep_unknown": bool(args.keep_unknown),
        "n": int(emb.shape[0]),
        "embedding_shape": [int(emb.shape[0]), int(emb.shape[1])],
        "n_truncated": int(n_truncated),
        "texts_sha256": texts_digest(texts),
        "smiles": list(smiles),
        "labels": list(labels),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return emb_path, meta_path


def load_reusable_encoder(
    out_dir: str,
    tag: str,
    name: str,
    args: argparse.Namespace,
    smiles: Sequence[str],
    texts: Sequence[str],
    *,
    variant: str = VARIANT,
) -> Optional[tuple[np.ndarray, int]]:
    emb_path, meta_path = encoder_cache_paths(out_dir, name, tag)
    if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
        print(
            f"[analyze] no reusable {name} cache at {emb_path}",
            flush=True,
        )
        return None
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    reason = encoder_cache_compatible(
        meta, args, name, smiles, texts, variant=variant
    )
    if reason is not None:
        print(
            f"[analyze] ignoring {name} cache ({reason}): {meta_path}",
            flush=True,
        )
        return None
    emb = np.asarray(np.load(emb_path), dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != len(smiles):
        print(
            f"[analyze] ignoring {name} cache (npy shape {emb.shape}): {emb_path}",
            flush=True,
        )
        return None
    n_truncated = int(meta.get("n_truncated", 0))
    print(f"[analyze] reusing {name} embeddings from {emb_path}", flush=True)
    return mtv._l2_normalize(emb), n_truncated


def _release_cuda() -> None:
    """Drop CUDA blocks held by a model that has just been deleted.

    MiST is ~6GB in bf16. On a 16GB GPU the previous encoder must actually be
    freed, not merely unreferenced, before the next ``from_pretrained``.
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _close_encode_fn(encode_fn: Optional[object]) -> None:
    closer = getattr(encode_fn, "close", None)
    if callable(closer):
        closer()
        return
    _release_cuda()


def _count_truncated(tokenizer, texts: Sequence[str], max_length: int) -> int:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return sum(len(ids) > max_length for ids in encoded["input_ids"])


def _call_encode(encode_fn: EncodeFn, molecules, texts: list[str]) -> np.ndarray:
    try:
        return np.asarray(encode_fn(molecules, texts), dtype=np.float64)
    except TypeError:
        return np.asarray(encode_fn(texts), dtype=np.float64)


class _QwenEncoder:
    def __init__(
        self, model_id: str, device: str, batch_size: int, max_length: int
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = mtv._torch_device(device)
        print(f"[analyze] loading {model_id} on {self.device}...", flush=True)
        self.model = SentenceTransformer(model_id, device=str(self.device))
        native = int(getattr(self.model, "max_seq_length", 0) or 0)
        if max_length > 0:
            self.model.max_seq_length = max_length
        self.tokenizer = self.model.tokenizer
        applied = int(getattr(self.model, "max_seq_length", max_length) or max_length)
        extra = f" native_max_seq_length={native}" if native else ""
        print(
            f"[analyze] qwen embedding max_seq_length={applied}{extra}",
            flush=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    def close(self) -> None:
        del self.model
        _release_cuda()


class _LlamaEncoder:
    """Frozen Llama last-layer embeddings, pooled like the bias-network encoder."""

    def __init__(
        self,
        model_id: str,
        pooling: str,
        max_length: int,
        device: str,
        batch_size: int,
        cache_dir: Optional[str] = None,
        task: str = TASK,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from boreft.pyreft.semantic_encoder import (
            _base_inner_model,
            pooling_uses_instruction_mask,
        )

        self.task = task
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self._uses_instruction_mask = pooling_uses_instruction_mask(pooling)
        self.device = mtv._torch_device(device)
        hf_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        print(
            f"[analyze] loading {model_id} on {self.device}"
            + (f" cache_dir={cache_dir}" if cache_dir else "")
            + "...",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, **hf_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **hf_kwargs
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.inner = _base_inner_model(self.model)

    def close(self) -> None:
        del self.model
        del self.inner
        del self.tokenizer
        _release_cuda()

    def encode(self, texts: Sequence[str], pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        import torch

        from boreft.pyreft.semantic_encoder import (
            _pool_hidden_states,
            encode_definition_inputs,
        )

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                end = start + self.batch_size
                input_ids, attention_mask, instruction_mask = encode_definition_inputs(
                    self.tokenizer,
                    list(texts[start:end]),
                    device=self.device,
                    max_length=self.max_length,
                    task=self.task,
                    word_definition_pairs=list(pairs[start:end]),
                    build_instruction_mask=self._uses_instruction_mask,
                )
                # last_hidden_state == copied last block + RMSNorm at LoRA-zero.
                hidden = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                pooled = _pool_hidden_states(
                    hidden,
                    attention_mask,
                    pooling=self.pooling,
                    instruction_mask=instruction_mask,
                )
                chunks.append(mtv._l2_normalize(pooled.float().cpu().numpy()))
        return np.concatenate(chunks, axis=0)


class _CausalLastTokenEncoder:
    """Frozen causal-LM last-layer last-token embeddings (no instruction mask)."""

    def __init__(
        self,
        model_id: str,
        max_length: int,
        device: str,
        batch_size: int,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from boreft.pyreft.semantic_encoder import _base_inner_model

        self.max_length = max_length
        self.batch_size = batch_size
        self.device = mtv._torch_device(device)
        hf_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        if local_files_only:
            hf_kwargs["local_files_only"] = True
        print(
            f"[analyze] loading {model_id} on {self.device} "
            f"(last-layer last-token)"
            + (f" cache_dir={cache_dir}" if cache_dir else "")
            + "...",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, **hf_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **hf_kwargs
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.inner = _base_inner_model(self.model)

    def close(self) -> None:
        del self.model
        del self.inner
        del self.tokenizer
        _release_cuda()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        from boreft.pyreft.semantic_encoder import _pool_hidden_states

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                hidden = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                # last_token is a last_instruction alias; without an instruction
                # mask this is the last non-pad token of the truncated sequence.
                pooled = _pool_hidden_states(
                    hidden,
                    attention_mask,
                    pooling="last_token",
                    instruction_mask=None,
                )
                chunks.append(mtv._l2_normalize(pooled.float().cpu().numpy()))
        return np.concatenate(chunks, axis=0)


def load_qwen_encode_fn(
    model_id: str, device: str, batch_size: int, max_length: int
) -> EncodeFn:
    encoder = _QwenEncoder(
        model_id, device=device, batch_size=batch_size, max_length=max_length
    )

    def encode(molecules, texts: Sequence[str]) -> np.ndarray:
        del molecules
        return encoder.encode(texts)

    encode.close = encoder.close  # type: ignore[attr-defined]
    encode.tokenizer = encoder.tokenizer  # type: ignore[attr-defined]
    return encode


def load_qwen_llm_encode_fn(
    model_id: str,
    max_length: int,
    device: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> EncodeFn:
    encoder = _CausalLastTokenEncoder(
        model_id,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    def encode(molecules, texts: Sequence[str]) -> np.ndarray:
        del molecules
        return encoder.encode(texts)

    encode.close = encoder.close  # type: ignore[attr-defined]
    encode.tokenizer = encoder.tokenizer  # type: ignore[attr-defined]
    return encode


def item_word_definition_pair(item) -> tuple[str, str]:
    """``(target, definition)`` for instruction-span pooling.

    Molopt items store the target on ``smiles``; semantle items use ``target``.
    """
    text = getattr(item, "smiles", None)
    if text is None:
        text = item.target
    return text, item.definition


def load_llama_encode_fn(
    model_id: str,
    pooling: str,
    max_length: int,
    device: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
    task: str = TASK,
) -> EncodeFn:
    encoder = _LlamaEncoder(
        model_id,
        pooling=pooling,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        task=task,
    )

    def encode(items, texts: Sequence[str]) -> np.ndarray:
        pairs = [item_word_definition_pair(item) for item in items]
        return encoder.encode(texts, pairs)

    encode.close = encoder.close  # type: ignore[attr-defined]
    encode.tokenizer = encoder.tokenizer  # type: ignore[attr-defined]
    return encode


def plot_encoder_purity(
    scores: dict[str, dict],
    save_path: str,
    title: str = "Production SMILES+defn clustering by encoder",
) -> Optional[str]:
    plt = mtv._plt()
    if plt is None or not scores:
        return None
    names = list(scores)
    knn5 = [float(scores[n]["knn_purity_at_5"]) for n in names]
    knn10 = [float(scores[n]["knn_purity_at_10"]) for n in names]
    kmeans = [float(scores[n]["kmeans_purity"]) for n in names]
    first = next(iter(scores.values()))
    chance_knn = float(first["chance_knn_purity"])
    chance_kmeans = float(first["chance_kmeans_purity"])
    majority = float(first["majority_class_fraction"])

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x - width, knn5, width, label="k-NN purity @5", color="#4C78A8")
    ax.bar(x, knn10, width, label="k-NN purity @10", color="#F58518")
    ax.bar(x + width, kmeans, width, label="k-means purity", color="#54A24B")
    ax.axhline(chance_knn, color="#4C78A8", ls="--", lw=1.0, label="chance k-NN")
    ax.axhline(chance_kmeans, color="#54A24B", ls=":", lw=1.0, label="chance k-means")
    ax.axhline(majority, color="#B279A2", ls="-.", lw=1.0, label="majority class")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Purity")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    mtv._style(ax)
    fig.tight_layout()
    save_plot(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_encoder_metrics(
    scores: dict[str, dict],
    save_path: str,
    title: str = "Agreement and cluster geometry by encoder",
) -> Optional[str]:
    plt = mtv._plt()
    if plt is None or not scores:
        return None
    names = list(scores)
    nmi = [float(scores[n].get("kmeans_nmi", float("nan"))) for n in names]
    ari = [float(scores[n].get("kmeans_ari", float("nan"))) for n in names]
    sil = [float(scores[n]["silhouette_cosine"]) for n in names]
    finite = [v for v in nmi + ari + sil if np.isfinite(v)]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x - width, nmi, width, label="k-means NMI", color="#72B7B2")
    ax.bar(x, ari, width, label="k-means ARI", color="#E45756")
    ax.bar(x + width, sil, width, label="cosine silhouette", color="#B279A2")
    ax.axhline(0.0, color="#9DA8B3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    if finite:
        ymin = min(-0.05, min(finite) - 0.04)
        ymax = max(1.02, max(finite) + 0.04)
        ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    mtv._style(ax)
    fig.tight_layout()
    save_plot(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_encoder_pca_grid(
    embeddings: dict[str, np.ndarray],
    labels: Sequence[str],
    unique_clusters: Sequence[str],
    style_map: dict,
    save_path: str,
    suptitle: str = "PCA of production SMILES+defn embeddings",
) -> Optional[str]:
    plt = mtv._plt()
    if plt is None or not embeddings:
        return None
    from sklearn.decomposition import PCA

    names = list(embeddings)
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 5.4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        emb = embeddings[name]
        n_components = min(2, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components)
        xy = pca.fit_transform(emb)
        if xy.ndim == 1:
            xy = np.column_stack([xy, np.zeros_like(xy)])
        elif xy.shape[1] == 1:
            xy = np.column_stack([xy[:, 0], np.zeros(xy.shape[0])])
        mtv._scatter_by_cluster(ax, xy, labels, unique_clusters, style_map, 22.0)
        ax.set_xlabel(mtv._pc_axis_label(pca, 0), fontsize=10)
        ax.set_ylabel(
            mtv._pc_axis_label(pca, 1) if n_components > 1 else "PC2",
            fontsize=10,
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        mtv._style(ax)
    fig.legend(
        handles=mtv._legend_handles(unique_clusters, style_map),
        title="Category",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        markerscale=1.1,
        borderaxespad=0.4,
    )
    fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()
    save_plot(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def print_encoder_report(
    scores: dict[str, dict],
    labels: Sequence[str],
    example: str,
    *,
    heading: str = "Qwen embedding vs Qwen3-1.7B last-token vs Llama clustering",
    variant: Optional[str] = None,
) -> None:
    print("\n" + "=" * 96)
    print(heading)
    print("=" * 96)
    print(f"  n = {len(labels)}   labels = {len(set(labels))}")
    print(f"  label counts: {mtv.label_counts(labels)}")
    preview = example if len(example) <= 110 else example[:107] + "..."
    print(f"\nexample {variant or VARIANT}: {preview}")

    header = (
        f"\n{'encoder':<16}{'kNN@5':>8}{'kNN@10':>8}{'k-purity':>10}"
        f"{'NMI':>8}{'ARI':>8}{'sil':>8}"
    )
    print(header)
    print("-" * len(header.strip("\n")))
    for name, s in scores.items():
        print(
            f"{name:<16}"
            f"{s.get('knn_purity_at_5', float('nan')):>8.4f}"
            f"{s.get('knn_purity_at_10', float('nan')):>8.4f}"
            f"{s.get('kmeans_purity', float('nan')):>10.4f}"
            f"{s.get('kmeans_nmi', float('nan')):>8.4f}"
            f"{s.get('kmeans_ari', float('nan')):>8.4f}"
            f"{s.get('silhouette_cosine', float('nan')):>8.4f}"
        )
    first = next(iter(scores.values()))
    print(
        f"\nchance k-NN purity = {float(first.get('chance_knn_purity', float('nan'))):.4f}    "
        f"chance k-means purity = {float(first.get('chance_kmeans_purity', float('nan'))):.4f}    "
        f"majority class = {float(first.get('majority_class_fraction', float('nan'))):.4f}"
    )
    print("=" * 96 + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--definitions",
        default=default_definitions_path(TASK),
        help="definitions.jsonl with category / category_normalized fields.",
    )
    parser.add_argument(
        "--rdkit-definitions",
        default=default_rdkit_definitions_path(),
        help="definitions_rdkit.jsonl (ten-d descriptor sidecar).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=mtv.DEFAULT_N,
        help="Sample size. 0 uses every eligible molecule.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-field",
        default="category_normalized",
        choices=("category_normalized", "category"),
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep molecules whose label is 'unknown' / 'multi-cluster'.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        choices=list(VARIANT_NAMES),
        help="Text variants to score. Default: smiles_defn and smiles_prompt.",
    )
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=[],
        choices=list(MOLOPT_ENCODER_NAMES),
        help="Subset of encoders to run. Default: qwen, qwen_llm, llama, and mist.",
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-llm-model", default=DEFAULT_QWEN_LLM_MODEL)
    parser.add_argument("--llama-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument(
        "--mist-model",
        default=DEFAULT_MIST_MODEL,
        help="Local MiST checkpoint directory, or its folder name under "
        "$HF_HOME/mist (default: Qwen2.5-3B_pretrained-v4-cot).",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="HuggingFace cache for gated Llama weights. "
        "Empty string uses the HF default. Qwen still uses HF_HOME.",
    )
    parser.add_argument(
        "--pooling",
        default=DEFAULT_POOLING,
        choices=["last_instruction", "instruction_mean", "last_token"],
        help="Llama pooling. last_token is a legacy alias of last_instruction.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Truncate every encoder to this many tokens "
        "(default 128, matching --bias-encoder-max-length).",
    )
    parser.add_argument("--qwen-batch-size", type=int, default=DEFAULT_QWEN_BATCH_SIZE)
    parser.add_argument(
        "--qwen-llm-batch-size", type=int, default=DEFAULT_QWEN_LLM_BATCH_SIZE
    )
    parser.add_argument("--llama-batch-size", type=int, default=DEFAULT_LLAMA_BATCH_SIZE)
    parser.add_argument("--mist-batch-size", type=int, default=DEFAULT_MIST_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("data", "molopt", "analysis"),
    )
    parser.add_argument(
        "--reuse-previous",
        action="store_true",
        help="Reload compatible encoder embeddings from --out-dir instead of "
        "re-encoding. Caches are written on every run as "
        "encoder_cluster_{name}_emb*.npy.",
    )
    add_wandb_cli(parser)
    return parser.parse_args(argv)


_VARIANT_TITLES = {
    VARIANT_DEFN: {
        "heading": "Qwen vs Qwen3-1.7B vs Llama vs MiST on SMILES+definition",
        "purity": "Production SMILES+defn clustering by encoder",
        "metrics": "Agreement and cluster geometry by encoder",
        "pca": "PCA of production SMILES+defn embeddings",
    },
    VARIANT_PROMPT: {
        "heading": "Qwen vs Qwen3-1.7B vs Llama vs MiST on the SMILES prompt",
        "purity": "SMILES-prompt clustering by encoder",
        "metrics": "Agreement and cluster geometry on the SMILES prompt",
        "pca": "PCA of SMILES-prompt embeddings",
    },
}


def _load_real_encode_fn(
    name: str, args: argparse.Namespace, variant: str
) -> EncodeFn:
    if name == "qwen":
        return load_qwen_encode_fn(
            args.qwen_model,
            args.device,
            args.qwen_batch_size,
            args.max_length,
        )
    if name == "qwen_llm":
        return load_qwen_llm_encode_fn(
            args.qwen_llm_model,
            max_length=args.max_length,
            device=args.device,
            batch_size=args.qwen_llm_batch_size,
        )
    if name == "mist":
        return load_qwen_llm_encode_fn(
            getattr(args, "mist_resolved", None) or resolve_mist_model(args.mist_model),
            max_length=args.max_length,
            device=args.device,
            batch_size=args.mist_batch_size,
            local_files_only=True,
        )
    if name == "llama" and variant == VARIANT_PROMPT:
        return load_qwen_llm_encode_fn(
            args.llama_model,
            max_length=args.max_length,
            device=args.device,
            batch_size=args.llama_batch_size,
            cache_dir=args.cache_dir or None,
        )
    if name == "llama":
        return load_llama_encode_fn(
            args.llama_model,
            pooling=args.pooling,
            max_length=args.max_length,
            device=args.device,
            batch_size=args.llama_batch_size,
            cache_dir=args.cache_dir or None,
        )
    raise ValueError(f"unknown encoder {name!r}")


def _encode_and_cache(
    encode_fn: EncodeFn,
    *,
    count_truncation: bool,
    args: argparse.Namespace,
    name: str,
    variant: str,
    molecules,
    texts: list[str],
    smiles: list[str],
    labels: list[str],
    ctag: str,
) -> tuple[np.ndarray, int]:
    n_truncated = 0
    if count_truncation:
        n_truncated = _count_truncated(
            encode_fn.tokenizer,  # type: ignore[attr-defined]
            texts,
            args.max_length,
        )
        if n_truncated:
            print(
                f"[analyze] WARNING: {n_truncated}/{len(texts)} {name} "
                f"{variant} inputs exceed --max-length={args.max_length} "
                "and will be truncated",
                flush=True,
            )
    emb = mtv._l2_normalize(_call_encode(encode_fn, molecules, texts))
    emb_path, _meta_path = save_encoder_cache(
        args.out_dir,
        ctag,
        name,
        args,
        emb,
        smiles,
        labels,
        texts,
        n_truncated,
        variant=variant,
    )
    print(f"[analyze] wrote {emb_path}", flush=True)
    return emb, n_truncated


def run(
    args: argparse.Namespace,
    encode_fns: Optional[dict[str, EncodeFn]] = None,
) -> list[dict]:
    """Score each selected text variant. Returns one report per variant."""
    encoders = selected_encoders(args.encoders, _catalog_for(args))
    variants = selected_variants(args.variants)
    if args.n < 0:
        raise ValueError(f"--n must be >= 0, got {args.n}")
    prepare_mist_checkpoint(args, encoders, encode_fns)

    print(f"[analyze] loading molecules from {args.definitions}", flush=True)
    rows = mtv.load_molecules(args.definitions, args.rdkit_definitions)
    molecules = mtv.sample_molecules(
        rows,
        n=args.n,
        seed=args.seed,
        label_field=args.label_field,
        drop_unknown=not args.keep_unknown,
    )
    smiles = [mol.smiles for mol in molecules]
    labels = [mtv.label_of(mol, args.label_field) for mol in molecules]
    print(
        f"[analyze] using {len(molecules)} molecules "
        f"(requested n={args.n}, seed={args.seed}, label={args.label_field})",
        flush=True,
    )
    print(f"[analyze] variants={variants} encoders={encoders}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    prepared: dict[str, dict] = {}
    for variant in variants:
        texts = texts_for_variant(molecules, variant)
        example = texts[0]
        preview = example if len(example) <= 140 else example[:137] + "..."
        print(f"[analyze] example {variant}: {preview}", flush=True)
        tag = artifact_tag(args, variant)
        ctag = cache_tag(args, variant)
        if tag:
            print(f"[analyze] {variant} artifact suffix{tag}", flush=True)
        reused: dict[str, tuple[np.ndarray, int]] = {}
        if args.reuse_previous:
            for name in encoders:
                loaded = load_reusable_encoder(
                    args.out_dir,
                    ctag,
                    name,
                    args,
                    smiles,
                    texts,
                    variant=variant,
                )
                if loaded is not None:
                    reused[name] = loaded
        prepared[variant] = {
            "texts": texts,
            "example": example,
            "tag": tag,
            "ctag": ctag,
            "reused": reused,
            "embeddings": {},
            "scores": {},
            "truncated": {},
        }

    if encode_fns is not None:
        missing = [
            name
            for name in encoders
            if name not in encode_fns
            and any(name not in prepared[variant]["reused"] for variant in variants)
        ]
        if missing:
            raise ValueError(f"encode_fns missing encoder(s): {missing}")

    for name in encoders:
        pending = [
            variant for variant in variants if name not in prepared[variant]["reused"]
        ]
        for variant in variants:
            if name in prepared[variant]["reused"]:
                emb, n_truncated = prepared[variant]["reused"][name]
                prepared[variant]["embeddings"][name] = emb
                prepared[variant]["truncated"][name] = n_truncated

        if not pending:
            continue

        # Llama's definition pool and the prompt's last-token pool are different
        # forwards. Everything else can encode every pending variant per load.
        groups: list[list[str]]
        if name == "llama" and encode_fns is None:
            groups = [[variant] for variant in pending]
        else:
            groups = [pending]

        for group in groups:
            encode_fn = encode_fns[name] if encode_fns is not None else None
            owns_fn = encode_fn is None
            if owns_fn:
                encode_fn = _load_real_encode_fn(name, args, group[0])
            assert encode_fn is not None
            try:
                for variant in group:
                    slot = prepared[variant]
                    print(
                        f"[analyze] encoding {name} {variant} "
                        f"({len(slot['texts'])} strings)...",
                        flush=True,
                    )
                    emb, n_truncated = _encode_and_cache(
                        encode_fn,
                        count_truncation=owns_fn,
                        args=args,
                        name=name,
                        variant=variant,
                        molecules=molecules,
                        texts=slot["texts"],
                        smiles=smiles,
                        labels=labels,
                        ctag=slot["ctag"],
                    )
                    slot["embeddings"][name] = emb
                    slot["truncated"][name] = n_truncated
            finally:
                if owns_fn:
                    _close_encode_fn(encode_fn)

    unique_clusters = sorted(set(labels))
    style_map = mtv.build_cluster_style_map(unique_clusters)
    reports: list[dict] = []
    for variant in variants:
        slot = prepared[variant]
        titles = _VARIANT_TITLES[variant]
        scores: dict[str, dict] = {}
        embeddings: dict[str, np.ndarray] = {}
        for name in encoders:
            emb = slot["embeddings"][name]
            embeddings[name] = emb
            scores[name] = mtv.score_variant(emb, labels, args.seed)
            scores[name]["n_truncated"] = int(slot["truncated"][name])
            print(
                f"  {variant} {name}: knn@5={scores[name]['knn_purity_at_5']:.3f} "
                f"kmeans={scores[name]['kmeans_purity']:.3f} "
                f"sil={scores[name]['silhouette_cosine']:.3f}",
                flush=True,
            )
            mtv.plot_variant_pca(
                emb,
                labels,
                title=(
                    f"{name}  —  {variant} 2-D PCA, "
                    f"coloured by {args.label_field}"
                ),
                save_path=artifact_path(
                    args.out_dir, f"encoder_cluster_{name}_pca", slot["tag"], "png"
                ),
                style_map=style_map,
                unique_clusters=unique_clusters,
            )
        print_encoder_report(
            scores,
            labels,
            slot["example"],
            heading=titles["heading"],
            variant=variant,
        )
        tag = slot["tag"]
        report = {
            "definitions_path": os.path.abspath(args.definitions),
            "rdkit_definitions_path": os.path.abspath(args.rdkit_definitions),
            "n_requested": args.n,
            "n": len(molecules),
            "seed": args.seed,
            "variant": variant,
            "example": slot["example"],
            "prompt": slot["example"],
            "label_field": args.label_field,
            "label_counts": mtv.label_counts(labels),
            "pooling": args.pooling,
            "llama_pooling": encoder_pooling(args, "llama", variant),
            "max_length": args.max_length,
            "qwen_model": args.qwen_model,
            "qwen_llm_model": args.qwen_llm_model,
            "llama_model": args.llama_model,
            "mist_model": encoder_model_id(args, "mist"),
            "mist_checkpoint": getattr(args, "mist_resolved", None),
            "cache_dir": args.cache_dir or None,
            "encoders": encoders,
            "reused_encoders": [name for name in encoders if name in slot["reused"]],
            "artifact_tag": tag,
            "cache_tag": slot["ctag"],
            "reuse_previous": bool(args.reuse_previous),
            "scores": mtv.strip_arrays(scores),
            "smiles": smiles,
            "labels": labels,
        }
        json_path = artifact_path(args.out_dir, "encoder_cluster", tag, "json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[analyze] wrote {json_path}")
        written = [
            plot_encoder_purity(
                scores,
                artifact_path(args.out_dir, "encoder_cluster_purity", tag, "png"),
                title=titles["purity"],
            ),
            plot_encoder_metrics(
                scores,
                artifact_path(args.out_dir, "encoder_cluster_metrics", tag, "png"),
                title=titles["metrics"],
            ),
            plot_encoder_pca_grid(
                embeddings,
                labels,
                unique_clusters,
                style_map,
                artifact_path(args.out_dir, "encoder_cluster_pca", tag, "png"),
                suptitle=titles["pca"],
            ),
        ]
        for path in written:
            if path:
                print(f"[analyze] wrote {path}")
        reports.append(report)
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.definitions):
        print(
            f"ERROR: {args.definitions} not found — run "
            f"scripts/prepare_molopt_chebi20.py --download first",
            file=sys.stderr,
        )
        return 1
    if not os.path.isfile(args.rdkit_definitions):
        print(f"ERROR: {args.rdkit_definitions} not found", file=sys.stderr)
        return 1
    reports = run(args)
    for report in reports:
        maybe_log_encoder_cluster(
            args.out_dir,
            report,
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            name=args.wandb_run_name,
            wandb_dir=args.wandb_dir,
            no_wandb=args.no_wandb,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
