"""Search-time subspace expansion: absorb BO observations into the ReFT map.

Every ``expand.every`` acquisition batches, selected decoded observations are
continue-trained with the checkpoint recipe. Replay is the original train
vocabulary plus already-absorbed search identities (not merely the warmstart
prefix). Learned bias vectors replace ``observation.point`` so the next GP fit
lives in the updated latent geometry.

This reuses :func:`boreft.learn_bias.learn_biases_batched` and the coreset
selectors shared with ``extend_subspace``. It does not write a new training
checkpoint; search decode already injects raw bias vectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Callable, ClassVar, Literal, Mapping, Optional, Sequence

import numpy as np

from boreft.bias_tables import stack_bias_vectors, training_search_region
from boreft.bo.acquisition import LatentEllipsoid, latent_bounds, latent_ellipsoid, union_latent_bounds
from boreft.bo.state import Observation, RunState
from boreft.coreset import SAMPLE_STRATEGIES, select_subset_indices
from boreft.eval.eval_suite import target_normalizer
from boreft.interactive import generate_base, strip_thinking_text
from boreft.learn_bias import (
    BatchLearnConfig,
    BatchLearnResult,
    GroupStopSpec,
    learn_biases_batched,
    predict_bias_network_rows,
)
from boreft.task_config import (
    definition_generation_field_label,
    definition_generation_instruction,
    definition_generation_query_label,
    task_supports_definition_generation,
    task_supports_sdpo,
)
from boreft.text_similarity import (
    default_definitions_path,
    load_raw_definitions,
)

ExpandStrategy = Literal["all", "random", "herding", "k-center-greedy"]
ExpandInit = Literal["predicted", "random"]

EXPANSION_STATE_NAME = "expansion_state.json"
EXPANSIONS_JSONL = "expansions.jsonl"
EXPANSION_WEIGHTS_NAME = "expansion_intervention.pt"
EXPAND_GROUP_REPLAY = "replay"
EXPAND_GROUP_NEW = "new"
EXPAND_GROUP_COMBINED = "combined"


@dataclass
class SearchExpandConfig:
    """CLI subtree ``--expand.*`` for search-time subspace expansion."""

    every: int = 0
    new_strategy: ExpandStrategy = "all"
    new_n: Optional[int] = None
    new_prop: Optional[float] = None
    replay_strategy: ExpandStrategy = "all"
    replay_n: Optional[int] = None
    replay_prop: Optional[float] = None
    init: ExpandInit = "predicted"
    new_only_epochs: Optional[int] = None
    definition_icl_k: int = 8
    definition_max_new_tokens: int = 96
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    grad_acc_steps: Optional[int] = None
    lr: Optional[float] = None
    kl_beta: Optional[float] = None
    lambda_ce: Optional[float] = None
    lambda_sdpo: Optional[float] = None
    linear_annealing_map: Optional[str] = None
    max_grad_norm: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    warmup_ratio: Optional[float] = None
    weight_decay_mode: Optional[str] = None
    wd_W: Optional[float] = None
    wd_b: Optional[float] = None
    eval_epochs: Optional[int] = None
    eval_batch_size: Optional[int] = None
    eval_max_new_tokens: Optional[int] = None
    eval_selection_metric: Optional[str] = None
    learn_w: bool = True
    learn_r: bool = True
    learn_bias_network: bool = True
    stop_threshold: Optional[float] = None
    stop_threshold_min: Optional[float] = None
    stop_threshold_frac: Optional[float] = None
    stop_replay_threshold: Optional[float] = None
    stop_replay_threshold_min: Optional[float] = None
    stop_replay_threshold_frac: Optional[float] = None
    stop_new_threshold: Optional[float] = None
    stop_new_threshold_min: Optional[float] = None
    stop_new_threshold_frac: Optional[float] = None

    # Timing/selection/`learn_*` stay frozen on --resume. Recipe, eval cadence,
    # and stop bars may change; they only affect future expansion rounds.
    RESUME_MUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "epochs",
            "new_only_epochs",
            "batch_size",
            "grad_acc_steps",
            "lr",
            "kl_beta",
            "lambda_ce",
            "lambda_sdpo",
            "linear_annealing_map",
            "max_grad_norm",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay_mode",
            "wd_W",
            "wd_b",
            "eval_epochs",
            "eval_batch_size",
            "eval_max_new_tokens",
            "eval_selection_metric",
            "stop_threshold",
            "stop_threshold_min",
            "stop_threshold_frac",
            "stop_replay_threshold",
            "stop_replay_threshold_min",
            "stop_replay_threshold_frac",
            "stop_new_threshold",
            "stop_new_threshold_min",
            "stop_new_threshold_frac",
        }
    )

    def enabled(self) -> bool:
        return int(self.every) > 0

    def mutates_checkpoint(self) -> bool:
        """True when expansion updates shared W / R / the bias network."""
        return bool(self.learn_w or self.learn_r or self.learn_bias_network)

    def validate(self) -> None:
        if self.every < 0:
            raise ValueError("expand.every must be >= 0")
        if self.new_strategy not in SAMPLE_STRATEGIES:
            raise ValueError(f"unknown expand.new_strategy {self.new_strategy!r}")
        if self.replay_strategy not in SAMPLE_STRATEGIES:
            raise ValueError(
                f"unknown expand.replay_strategy {self.replay_strategy!r}"
            )
        if self.init not in ("predicted", "random"):
            raise ValueError(f"unknown expand.init {self.init!r}")
        if self.new_only_epochs is not None and self.new_only_epochs < 0:
            raise ValueError("expand.new_only_epochs must be >= 0")
        if self.definition_icl_k < 0:
            raise ValueError("expand.definition_icl_k must be >= 0")
        if self.definition_max_new_tokens < 1:
            raise ValueError("expand.definition_max_new_tokens must be positive")
        _validate_count_pair(
            self.new_n,
            self.new_prop,
            n_name="expand.new_n",
            prop_name="expand.new_prop",
            strategy=self.new_strategy,
        )
        _validate_count_pair(
            self.replay_n,
            self.replay_prop,
            n_name="expand.replay_n",
            prop_name="expand.replay_prop",
            strategy=self.replay_strategy,
        )
        for name, value in (
            ("expand.stop_threshold_frac", self.stop_threshold_frac),
            ("expand.stop_replay_threshold_frac", self.stop_replay_threshold_frac),
            ("expand.stop_new_threshold_frac", self.stop_new_threshold_frac),
        ):
            if value is None:
                continue
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        _validate_frac_requires_min(
            self.stop_threshold_frac,
            self.stop_threshold_min,
            frac_name="expand.stop_threshold_frac",
            min_name="expand.stop_threshold_min",
        )
        _validate_frac_requires_min(
            self.stop_replay_threshold_frac,
            self.stop_replay_threshold_min,
            frac_name="expand.stop_replay_threshold_frac",
            min_name="expand.stop_replay_threshold_min",
        )
        _validate_frac_requires_min(
            self.stop_new_threshold_frac,
            self.stop_new_threshold_min,
            frac_name="expand.stop_new_threshold_frac",
            min_name="expand.stop_new_threshold_min",
        )


def _validate_count_pair(
    n: Optional[int],
    prop: Optional[float],
    *,
    n_name: str,
    prop_name: str,
    strategy: str,
) -> None:
    if strategy == "all" and (n is not None or prop is not None):
        raise ValueError(f"strategy='all' cannot be combined with {n_name} / {prop_name}")
    if n is not None and prop is not None:
        raise ValueError(f"specify at most one of {n_name} / {prop_name}")
    if n is not None and n < 0:
        raise ValueError(f"{n_name} must be >= 0")
    if prop is not None and not (0.0 <= float(prop) <= 1.0):
        raise ValueError(f"{prop_name} must be in [0, 1]")


def _validate_frac_requires_min(
    frac: Optional[float],
    minimum: Optional[float],
    *,
    frac_name: str,
    min_name: str,
) -> None:
    if frac is None:
        return
    if float(frac) < 1.0 - 1e-12 and minimum is None:
        raise ValueError(f"{frac_name} < 1 requires {min_name}")


@dataclass(frozen=True)
class UniqueTarget:
    text: str
    key: str
    observation_indices: tuple[int, ...]
    point: tuple[float, ...]
    origin: Literal["train", "search"] = "search"


@dataclass
class ExpansionDiskState:
    n_rounds: int = 0
    last_expanded_batches: int = 0
    absorbed_through_index: int = 0


def acquisition_batches_completed(observations: Sequence[Observation]) -> int:
    """1-based count of acquisition batches from ``bo_batch`` on the last acq."""
    acquired = [item for item in observations if item.source == "acquisition"]
    if not acquired:
        return 0
    return int(acquired[-1].components.get("bo_batch", 0)) + 1


def should_expand(
    *,
    expand_every: int,
    n_batches: int,
    last_expanded_batches: int,
) -> bool:
    if expand_every <= 0 or n_batches <= 0:
        return False
    if n_batches % expand_every != 0:
        return False
    return last_expanded_batches != n_batches


def replay_window_end(
    observations: Sequence[Observation],
    absorbed_through_index: int,
) -> int:
    """Exclusive end index of the replay window.

    ``absorbed_through_index == 0`` means no expansion has finished yet, so
    replay is the warmstart prefix. After a round (including a skip) the
    persisted cutoff is used as-is.
    """
    n = len(observations)
    absorbed = max(0, int(absorbed_through_index))
    if absorbed > 0:
        return min(absorbed, n)
    return int(sum(item.source == "warmstart" for item in observations))


def observation_target_key(observation: Observation, task: str) -> str:
    text = str(observation.decoded or "").strip()
    if not text:
        return ""
    return target_normalizer(task)(text)


def unique_targets_from_observations(
    observations: Sequence[Observation],
    *,
    task: str,
    indices: Sequence[int] | None = None,
) -> list[UniqueTarget]:
    """Deduplicate observations by task-normalized decode, preserving first spelling."""
    chosen = list(indices) if indices is not None else list(range(len(observations)))
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index in chosen:
        if index < 0 or index >= len(observations):
            raise IndexError(f"observation index {index} out of range")
        key = observation_target_key(observations[index], task)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(int(index))
    out: list[UniqueTarget] = []
    for key in order:
        idxs = grouped[key]
        first = observations[idxs[0]]
        out.append(
            UniqueTarget(
                text=str(first.decoded).strip(),
                key=key,
                observation_indices=tuple(idxs),
                point=tuple(float(v) for v in first.point),
                origin="search",
            )
        )
    return out


def unique_targets_from_checkpoint(ckpt, *, task: str) -> list[UniqueTarget]:
    """Original train vocabulary as replay candidates, with current table mus."""
    words = getattr(ckpt, "words", None)
    items = getattr(ckpt, "items", None)
    if not isinstance(words, (list, tuple)):
        words = []
    if not isinstance(items, (list, tuple)):
        items = []
    texts: list[str] = []
    ids: list[int] = []
    for i, word in enumerate(words):
        text = str(word).strip()
        word_id = i
        if i < len(items) and isinstance(items[i], Mapping):
            row = items[i]
            text = str(row.get("word", row.get("target", text))).strip()
            try:
                word_id = int(row.get("id", i))
            except (TypeError, ValueError):
                word_id = i
        if not text:
            continue
        texts.append(text)
        ids.append(word_id)
    if not texts:
        return []
    mus = np.asarray(stack_bias_vectors(ckpt.reft_model, ids), dtype=np.float32)
    if mus.ndim != 2 or mus.shape[0] != len(texts):
        raise ValueError("checkpoint bias rows do not match the train vocabulary")
    norm = target_normalizer(task)
    out: list[UniqueTarget] = []
    seen: set[str] = set()
    for text, row in zip(texts, mus):
        key = norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            UniqueTarget(
                text=text,
                key=key,
                observation_indices=(),
                point=tuple(float(v) for v in np.asarray(row, dtype=float).reshape(-1)),
                origin="train",
            )
        )
    return out


def merge_replay_targets(
    train_targets: Sequence[UniqueTarget],
    absorbed_targets: Sequence[UniqueTarget],
) -> list[UniqueTarget]:
    """Train vocabulary first, then absorbed search identities not already in it."""
    by_key: dict[str, UniqueTarget] = {}
    order: list[str] = []
    for item in train_targets:
        if item.key not in by_key:
            order.append(item.key)
        by_key[item.key] = replace(item, origin="train")
    extras: list[UniqueTarget] = []
    for item in absorbed_targets:
        previous = by_key.get(item.key)
        if previous is None:
            extras.append(replace(item, origin="search"))
            continue
        by_key[item.key] = UniqueTarget(
            text=previous.text,
            key=previous.key,
            observation_indices=item.observation_indices,
            point=item.point,
            origin="train",
        )
    for item in extras:
        if item.key in by_key:
            continue
        order.append(item.key)
        by_key[item.key] = item
    return [by_key[key] for key in order]


def partition_new_and_replay(
    observations: Sequence[Observation],
    *,
    task: str,
    absorbed_through_index: int,
    train_targets: Sequence[UniqueTarget] | None = None,
) -> tuple[list[UniqueTarget], list[UniqueTarget]]:
    """Split unique targets into the new window vs train + absorbed replay."""
    absorbed = max(0, min(int(absorbed_through_index), len(observations)))
    new_idxs = list(range(absorbed, len(observations)))
    replay_idxs = list(range(absorbed))
    new_unique = unique_targets_from_observations(
        observations, task=task, indices=new_idxs
    )
    absorbed_unique = unique_targets_from_observations(
        observations, task=task, indices=replay_idxs
    )
    replay_unique = merge_replay_targets(train_targets or (), absorbed_unique)
    replay_keys = {item.key for item in replay_unique}
    still_new = [item for item in new_unique if item.key not in replay_keys]
    return still_new, replay_unique


def select_unique_subset(
    items: Sequence[UniqueTarget],
    *,
    strategy: str,
    n: Optional[int],
    prop: Optional[float],
    seed: int,
    n_name: str,
    prop_name: str,
    strategy_name: str,
) -> list[UniqueTarget]:
    if not items:
        return []
    features = np.asarray([item.point for item in items], dtype=np.float32)
    idxs = select_subset_indices(
        len(items),
        n=n,
        prop=prop,
        strategy=strategy,
        features=features,
        seed=seed,
        allowed=SAMPLE_STRATEGIES,
        n_name=n_name,
        prop_name=prop_name,
        strategy_name=strategy_name,
    )
    return [items[i] for i in idxs]


def points_for_targets(items: Sequence[UniqueTarget]) -> np.ndarray:
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray([item.point for item in items], dtype=np.float32)


def random_bias_init(
    n: int,
    dimension: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if n < 1:
        return np.zeros((0, dimension), dtype=np.float32)
    lo = np.asarray(bounds[0], dtype=np.float32)
    hi = np.asarray(bounds[1], dtype=np.float32)
    unit = rng.random((n, dimension)).astype(np.float32)
    return lo + unit * (hi - lo)


def remap_observation_points(
    observations: Sequence[Observation],
    *,
    task: str,
    key_to_mu: Mapping[str, np.ndarray],
) -> list[Observation]:
    updated: list[Observation] = []
    for item in observations:
        key = observation_target_key(item, task)
        mu = key_to_mu.get(key)
        if mu is None:
            updated.append(item)
            continue
        point = np.asarray(mu, dtype=float).reshape(-1)
        updated.append(replace(item, point=[float(v) for v in point]))
    return updated


def load_expansion_state(seed_dir: str | Path) -> ExpansionDiskState:
    path = Path(seed_dir) / EXPANSION_STATE_NAME
    if not path.is_file():
        return ExpansionDiskState()
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExpansionDiskState(
        n_rounds=int(data.get("n_rounds") or 0),
        last_expanded_batches=int(data.get("last_expanded_batches") or 0),
        absorbed_through_index=int(data.get("absorbed_through_index") or 0),
    )


def write_expansion_state(seed_dir: str | Path, state: ExpansionDiskState) -> None:
    path = Path(seed_dir) / EXPANSION_STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ckpt_intervention(ckpt):
    intervention = list(ckpt.reft_model.interventions.values())[0]
    return intervention[0] if isinstance(intervention, (list, tuple)) else intervention


def _clone_state_dict(module, *, device: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in module.state_dict().items():
        tensor = value.detach().clone()
        if device is not None:
            tensor = tensor.to(device)
        out[key] = tensor
    return out


def snapshot_search_intervention(ckpt, *, device: str | None = None) -> dict[str, Any]:
    """Clone shared intervention tensors so a seed or failed round can roll back."""
    intervention = _ckpt_intervention(ckpt)
    payload: dict[str, Any] = {
        "learned_source": _clone_state_dict(
            intervention.learned_source, device=device
        ),
        "rotate_layer": _clone_state_dict(intervention.rotate_layer, device=device),
    }
    if getattr(intervention, "bias_network", None) is not None:
        payload["bias_network"] = _clone_state_dict(
            intervention.bias_network, device=device
        )
    encoder = (
        intervention.get_semantic_encoder()
        if hasattr(intervention, "get_semantic_encoder")
        else None
    )
    if encoder is not None:
        payload["semantic_encoder"] = _clone_state_dict(encoder, device=device)
    return payload


def restore_search_intervention(ckpt, payload: Mapping[str, Any]) -> None:
    intervention = _ckpt_intervention(ckpt)
    if "learned_source" in payload:
        intervention.learned_source.load_state_dict(payload["learned_source"])
    if "rotate_layer" in payload:
        intervention.rotate_layer.load_state_dict(payload["rotate_layer"])
    if "bias_network" in payload and getattr(intervention, "bias_network", None) is not None:
        intervention.bias_network.load_state_dict(payload["bias_network"])
    encoder = (
        intervention.get_semantic_encoder()
        if hasattr(intervention, "get_semantic_encoder")
        else None
    )
    if encoder is not None and "semantic_encoder" in payload:
        encoder.load_state_dict(payload["semantic_encoder"])


def save_expansion_weights(seed_dir: str | Path, ckpt) -> None:
    import torch

    path = Path(seed_dir) / EXPANSION_WEIGHTS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(snapshot_search_intervention(ckpt, device="cpu"), temporary)
    os.replace(temporary, path)


def load_expansion_weights(seed_dir: str | Path, ckpt) -> bool:
    import torch

    path = Path(seed_dir) / EXPANSION_WEIGHTS_NAME
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    restore_search_intervention(ckpt, payload)
    return True


def append_expansion_record(seed_dir: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(seed_dir) / EXPANSIONS_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_expansion_records(seed_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(seed_dir) / EXPANSIONS_JSONL
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_run_definitions_path(saved_cfg: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "definitions_path",
        "sdpo_definitions_path",
        "bias_encoder_definitions_path",
    ):
        path = saved_cfg.get(key)
        if path and os.path.isfile(str(path)):
            return str(path)
    fallback = default_definitions_path(str(saved_cfg.get("task", "semantle")))
    return fallback if os.path.isfile(fallback) else None


def load_run_raw_definitions(saved_cfg: Mapping[str, Any]) -> dict[str, str]:
    path = resolve_run_definitions_path(saved_cfg)
    if not path:
        return {}
    try:
        return load_raw_definitions(path)
    except (OSError, ValueError, KeyError):
        return {}


def definition_lookup_by_key(
    raw_defs: Mapping[str, str],
    *,
    task: str,
) -> dict[str, tuple[str, str]]:
    """Map normalized target -> (original text, definition)."""
    norm = target_normalizer(task)
    out: dict[str, tuple[str, str]] = {}
    for text, definition in raw_defs.items():
        word = str(text).strip()
        gloss = str(definition).strip()
        if not word or not gloss:
            continue
        out.setdefault(norm(word), (word, gloss))
    return out


def lookup_definition(
    keyed: Mapping[str, tuple[str, str]],
    text: str,
    *,
    task: str,
) -> Optional[str]:
    key = target_normalizer(task)(str(text).strip())
    if not key:
        return None
    hit = keyed.get(key)
    return hit[1] if hit else None


def sample_icl_examples(
    keyed: Mapping[str, tuple[str, str]],
    *,
    exclude: str,
    k: int,
    seed: int,
    task: str,
) -> list[tuple[str, str]]:
    if k <= 0 or not keyed:
        return []
    exclude_key = target_normalizer(task)(exclude)
    candidates = [
        (text, definition)
        for key, (text, definition) in keyed.items()
        if key != exclude_key
    ]
    if len(candidates) <= k:
        return candidates
    return random.Random(int(seed)).sample(candidates, k)


_GENERIC_DEFINITION_LABELS = (
    "definition",
    "description",
    "research objective",
)


def strip_generated_definition(text: str, *, task: str = "semantle") -> str:
    """Keep the first gloss; drop ICL labels and any continued example list."""
    cleaned = strip_thinking_text(str(text or "")).strip()
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    labels: list[str] = []
    seen: set[str] = set()
    for label in (definition_generation_field_label(task), *_GENERIC_DEFINITION_LABELS):
        key = str(label).strip().lower()
        if key and key not in seen:
            seen.add(key)
            labels.append(re.escape(key))
    if labels:
        label_re = "|".join(labels)
        parts = re.split(
            rf"(?im)^(?:\*\*)?(?:{label_re})\s*:?\s*(?:\*\*)?",
            cleaned,
        )
        if len(parts) > 1:
            cleaned = next((part.strip() for part in parts[1:] if part.strip()), "")
        else:
            cleaned = re.sub(
                rf"(?is)^(?:\*\*)?(?:{label_re})\s*:?\s*(?:\*\*)?",
                "",
                cleaned,
                count=1,
            ).strip()
    query_label = definition_generation_query_label(task)
    if query_label:
        cleaned = re.split(
            rf"(?im)\n(?:\*\*)?{re.escape(query_label)}\s*(?::|\s*$)",
            cleaned,
            maxsplit=1,
        )[0].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def generate_target_definition(
    ckpt,
    text: str,
    *,
    task: str,
    examples: Sequence[tuple[str, str]],
    max_new_tokens: int,
) -> str:
    if not task_supports_definition_generation(task):
        raise ValueError(
            f"task {task!r} has no definition-generation prompt in task_config"
        )
    use_chat = bool(getattr(ckpt, "from_chat_template", False))
    prompt = definition_generation_instruction(
        task,
        text,
        examples=examples,
        use_chat_template=use_chat,
    )
    if use_chat:
        from boreft.data_utils import chat_prompt

        prompt = chat_prompt(ckpt.tokenizer, prompt)
        prompt = f"{prompt}{definition_generation_field_label(task)}: "
    raw = generate_base(
        ckpt.reft_model.model,
        ckpt.tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        from_chat_template=use_chat,
        assistant_suffix=getattr(ckpt, "assistant_suffix", None),
    )
    return strip_generated_definition(raw, task=task)


def collect_definitions(
    ckpt,
    targets: Sequence[str],
    *,
    task: str,
    raw_defs: Mapping[str, str],
    icl_k: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``{target: definition}`` plus a generated-only subset."""
    keyed = definition_lookup_by_key(raw_defs, task=task)
    found: dict[str, str] = {}
    generated: dict[str, str] = {}
    for offset, target in enumerate(targets):
        existing = lookup_definition(keyed, target, task=task)
        if existing:
            found[target] = existing
            continue
        examples = sample_icl_examples(
            keyed,
            exclude=target,
            k=icl_k,
            seed=int(seed) + offset,
            task=task,
        )
        definition = generate_target_definition(
            ckpt,
            target,
            task=task,
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
        if not definition:
            definition = str(target).strip()
            print(
                f"[expand] definition generation empty for {target!r}; "
                "falling back to the target string",
                flush=True,
            )
        found[target] = definition
        generated[target] = definition
    return found, generated


def definitions_required(ckpt, config: SearchExpandConfig) -> bool:
    saved = getattr(ckpt, "saved_cfg", None) or {}
    task = str(saved.get("task") or "semantle")
    lambda_sdpo = (
        float(config.lambda_sdpo)
        if config.lambda_sdpo is not None
        else float(saved.get("lambda_sdpo") or 0.0)
    )
    if lambda_sdpo > 0.0 and task_supports_sdpo(task):
        return True
    if saved.get("bias_input_source") == "llm_encoder":
        return True
    return bool(saved.get("use_definition_embeds"))


def _shared_stop_frac(config: SearchExpandConfig) -> float:
    if config.stop_threshold_frac is not None:
        return float(config.stop_threshold_frac)
    return 1.0


def _group_stop_spec(
    *,
    threshold: Optional[float],
    threshold_min: Optional[float],
    threshold_frac: Optional[float],
    default_frac: float,
) -> Optional[GroupStopSpec]:
    if threshold is None and threshold_min is None:
        return None
    frac = float(threshold_frac) if threshold_frac is not None else default_frac
    spec = GroupStopSpec(
        stop_threshold=threshold,
        stop_threshold_min=threshold_min,
        stop_threshold_frac=frac,
    )
    spec.validate()
    return spec


def _batch_learn_config(
    config: SearchExpandConfig,
    *,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> BatchLearnConfig:
    return BatchLearnConfig(
        epochs=config.epochs if epochs is None else epochs,
        batch_size=config.batch_size,
        grad_acc_steps=config.grad_acc_steps,
        lr=config.lr,
        kl_beta=config.kl_beta,
        lambda_ce=config.lambda_ce,
        lambda_sdpo=config.lambda_sdpo,
        linear_annealing_map=config.linear_annealing_map,
        seed=seed,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay_mode=config.weight_decay_mode,
        wd_W=config.wd_W,
        wd_b=config.wd_b,
        eval_epochs=config.eval_epochs,
        eval_batch_size=config.eval_batch_size,
        eval_max_new_tokens=config.eval_max_new_tokens,
        eval_selection_metric=config.eval_selection_metric,
        stop_threshold=config.stop_threshold,
        stop_threshold_min=config.stop_threshold_min,
        stop_threshold_frac=config.stop_threshold_frac,
        inherit_stop_threshold=False,
        learn_W=config.learn_w,
        learn_R=config.learn_r,
        learn_bias_network=config.learn_bias_network,
    )


def _eval_layout(
    n_replay: int,
    n_new: int,
) -> tuple[dict[str, list[int]], list[int]]:
    groups: dict[str, list[int]] = {}
    if n_replay:
        groups[EXPAND_GROUP_REPLAY] = list(range(n_replay))
    if n_new:
        groups[EXPAND_GROUP_NEW] = list(range(n_replay, n_replay + n_new))
    groups[EXPAND_GROUP_COMBINED] = list(range(n_replay + n_new))
    return groups, list(range(n_replay + n_new))


def _stop_groups_for(
    config: SearchExpandConfig,
    *,
    has_replay: bool,
    has_new: bool,
    require_missing: bool = True,
) -> Optional[dict[str, GroupStopSpec]]:
    default_frac = _shared_stop_frac(config)
    specs: dict[str, GroupStopSpec] = {}
    replay_spec = _group_stop_spec(
        threshold=config.stop_replay_threshold,
        threshold_min=config.stop_replay_threshold_min,
        threshold_frac=config.stop_replay_threshold_frac,
        default_frac=default_frac,
    )
    new_spec = _group_stop_spec(
        threshold=config.stop_new_threshold,
        threshold_min=config.stop_new_threshold_min,
        threshold_frac=config.stop_new_threshold_frac,
        default_frac=default_frac,
    )
    if replay_spec is not None:
        if has_replay:
            specs[EXPAND_GROUP_REPLAY] = replay_spec
        elif require_missing:
            raise ValueError("expand.stop_replay_* requires a non-empty replay set")
    if new_spec is not None:
        if has_new:
            specs[EXPAND_GROUP_NEW] = new_spec
        elif require_missing:
            raise ValueError("expand.stop_new_* requires a non-empty new set")
    return specs or None


def _result_payload(result: BatchLearnResult, *, phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "targets": list(result.targets),
        "steps": int(result.steps),
        "mean_train_loss": float(result.mean_train_loss),
        "stopped_early": bool(result.stopped_early),
        "eval_history": list(result.eval_history or []),
        "resolved_train_config": dict(result.resolved_train_config or {}),
    }


def _key_to_mu(result: BatchLearnResult, *, task: str) -> dict[str, np.ndarray]:
    norm = target_normalizer(task)
    mu = np.asarray(result.mu, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for i, target in enumerate(result.targets):
        out[norm(target)] = mu[i]
    return out


def expand_search_subspace_window(
    *,
    ckpt,
    state: RunState,
    config: SearchExpandConfig,
    bounds: np.ndarray,
    task: str,
    seed: int,
    round_index: int,
    absorbed_through_index: int,
    bounds_padding: float = 0.0,
    aabb_std_k: float = 0.0,
    search_domain: str = "aabb",
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    eval_result_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    progress_factory: Optional[
        Callable[..., Optional[Callable[[dict[str, Any]], None]]]
    ] = None,
    raw_defs: Optional[Mapping[str, str]] = None,
) -> tuple[RunState, np.ndarray, dict[str, Any]]:
    """Absorb the current new-observation window into the live subspace."""
    new_all, replay_all = partition_new_and_replay(
        state.observations,
        task=task,
        absorbed_through_index=absorbed_through_index,
        train_targets=unique_targets_from_checkpoint(ckpt, task=task),
    )
    new_items = select_unique_subset(
        new_all,
        strategy=config.new_strategy,
        n=config.new_n,
        prop=config.new_prop,
        seed=seed + 17 * round_index,
        n_name="expand.new_n",
        prop_name="expand.new_prop",
        strategy_name="expand.new_strategy",
    )
    replay_items = select_unique_subset(
        replay_all,
        strategy=config.replay_strategy,
        n=config.replay_n,
        prop=config.replay_prop,
        seed=seed + 31 * round_index,
        n_name="expand.replay_n",
        prop_name="expand.replay_prop",
        strategy_name="expand.replay_strategy",
    )
    if not new_items:
        print(
            "[expand] skipping round: no new unique observations to absorb",
            flush=True,
        )
        return state, np.asarray(bounds, dtype=np.float32), {
            "round": round_index,
            "skipped": True,
            "reason": "no_new_targets",
            "n_new": 0,
            "n_replay": len(replay_items),
        }

    new_targets = [item.text for item in new_items]
    replay_targets = [item.text for item in replay_items]
    rank = int(np.asarray(state.observations[0].point).shape[0])
    bound_array = np.asarray(bounds, dtype=np.float32)
    rng = np.random.default_rng(seed + 53 * round_index)

    defs_raw = dict(raw_defs or {})
    need_defs = definitions_required(ckpt, config)
    train_defs: dict[str, str] = {}
    generated_defs: dict[str, str] = {}
    all_targets_for_defs = replay_targets + new_targets
    if need_defs:
        train_defs, generated_defs = collect_definitions(
            ckpt,
            all_targets_for_defs,
            task=task,
            raw_defs=defs_raw,
            icl_k=config.definition_icl_k,
            max_new_tokens=config.definition_max_new_tokens,
            seed=seed + 71 * round_index,
        )

    if config.init == "predicted":
        pred_mu, _pred_logvar = predict_bias_network_rows(
            ckpt,
            new_targets,
            definitions=train_defs if need_defs else None,
        )
        new_warm = np.asarray(pred_mu, dtype=np.float32)
    else:
        new_warm = random_bias_init(len(new_targets), rank, bound_array, rng)
    replay_warm = points_for_targets(replay_items)
    if replay_warm.size and replay_warm.shape[1] != rank:
        raise ValueError("replay bias dimension does not match observations")

    def _factory_callback(phase: str) -> Optional[Callable[[dict[str, Any]], None]]:
        if progress_factory is None:
            return None
        try:
            return progress_factory(round_index, phase)
        except TypeError:
            return progress_factory(round_index)

    snapshot = snapshot_search_intervention(ckpt)
    phases: list[dict[str, Any]] = []
    new_only_epochs = int(config.new_only_epochs or 0)
    learned: Optional[BatchLearnResult] = None
    try:
        if new_only_epochs > 0:
            print(
                f"[expand] round {round_index}: new-only phase "
                f"({len(new_targets)} targets, {new_only_epochs} epochs)",
                flush=True,
            )
            new_groups, new_eval = _eval_layout(0, len(new_targets))
            new_stop = _stop_groups_for(
                config, has_replay=False, has_new=True, require_missing=False
            )
            new_cb = _factory_callback("new_only")
            learned = learn_biases_batched(
                ckpt,
                targets=new_targets,
                warm_start_mu=new_warm,
                definitions=train_defs if need_defs else None,
                config=_batch_learn_config(config, epochs=new_only_epochs, seed=seed),
                eval_groups=new_groups,
                stop_eval_indices=new_eval,
                stop_groups=new_stop,
                show_progress=True,
                progress_callback=new_cb if new_cb is not None else progress_callback,
                eval_result_callback=(
                    new_cb if new_cb is not None else eval_result_callback
                ),
            )
            phases.append(_result_payload(learned, phase="new_only"))
            new_warm = np.asarray(learned.mu, dtype=np.float32)

        joint_targets = replay_targets + new_targets
        if replay_warm.size:
            joint_warm = np.concatenate([replay_warm, new_warm], axis=0)
        else:
            joint_warm = new_warm
        joint_groups, joint_eval = _eval_layout(len(replay_targets), len(new_targets))
        joint_stop = _stop_groups_for(
            config,
            has_replay=bool(replay_targets),
            has_new=True,
            require_missing=True,
        )
        print(
            f"[expand] round {round_index}: joint phase "
            f"(new={len(new_targets)}, replay={len(replay_targets)})",
            flush=True,
        )
        joint_cb = _factory_callback("joint")
        learned = learn_biases_batched(
            ckpt,
            targets=joint_targets,
            warm_start_mu=joint_warm,
            definitions=train_defs if need_defs else None,
            config=_batch_learn_config(config, seed=seed),
            eval_groups=joint_groups,
            stop_eval_indices=joint_eval,
            stop_groups=joint_stop,
            show_progress=True,
            progress_callback=joint_cb if joint_cb is not None else progress_callback,
            eval_result_callback=(
                joint_cb if joint_cb is not None else eval_result_callback
            ),
        )
        phases.append(_result_payload(learned, phase="joint"))
    except Exception:
        restore_search_intervention(ckpt, snapshot)
        raise

    assert learned is not None
    key_to_mu = _key_to_mu(learned, task=task)
    if config.learn_bias_network:
        missing: list[str] = []
        missing_keys: list[str] = []
        seen: set[str] = set()
        for item in state.observations:
            key = observation_target_key(item, task)
            if not key or key in key_to_mu or key in seen:
                continue
            seen.add(key)
            missing.append(str(item.decoded).strip())
            missing_keys.append(key)
        if missing:
            miss_defs = None
            if need_defs:
                extra, extra_gen = collect_definitions(
                    ckpt,
                    missing,
                    task=task,
                    raw_defs={**defs_raw, **train_defs},
                    icl_k=config.definition_icl_k,
                    max_new_tokens=config.definition_max_new_tokens,
                    seed=seed + 97 * round_index,
                )
                miss_defs = extra
                generated_defs.update(extra_gen)
            pred_mu, _ = predict_bias_network_rows(
                ckpt, missing, definitions=miss_defs
            )
            for key, row in zip(missing_keys, np.asarray(pred_mu, dtype=np.float32)):
                key_to_mu[key] = row

    remapped = remap_observation_points(
        state.observations, task=task, key_to_mu=key_to_mu
    )
    state.rewrite(remapped)
    obs_points = np.asarray(state.points, dtype=np.float32)
    obs_bounds = latent_bounds(
        obs_points,
        padding=bounds_padding,
    ).astype(np.float32)
    words = getattr(ckpt, "words", None) or []
    new_ellipsoid = None
    if search_domain == "ellipsoid":
        if words:
            _, _, train_ell = training_search_region(
                ckpt.reft_model,
                list(range(len(words))),
                padding=bounds_padding,
                aabb_std_k=aabb_std_k,
                search_domain="ellipsoid",
            )
            if train_ell is None:
                raise RuntimeError("ellipsoid search domain returned no ellipsoid")
            extra_radius = float(np.max(train_ell.mahalanobis(obs_points), initial=0.0))
            # Radial inflation of the train covariance ellipsoid keeps that
            # orientation and is a superset; this is not a refit MVEE.
            scale = max(extra_radius, 1.0)
            new_ellipsoid = LatentEllipsoid(
                center=train_ell.center,
                chol=train_ell.chol * scale,
            )
        else:
            new_ellipsoid = latent_ellipsoid(obs_points, padding=bounds_padding)
        new_bounds = new_ellipsoid.aabb().astype(np.float32)
    elif aabb_std_k > 0 and words:
        _, train_bounds, _ = training_search_region(
            ckpt.reft_model,
            list(range(len(words))),
            padding=bounds_padding,
            aabb_std_k=aabb_std_k,
            search_domain="aabb",
        )
        new_bounds = union_latent_bounds(train_bounds, obs_bounds).astype(np.float32)
    else:
        new_bounds = obs_bounds
    record = {
        "round": round_index,
        "skipped": False,
        "n_new": len(new_targets),
        "n_replay": len(replay_targets),
        "n_replay_train": sum(item.origin == "train" for item in replay_items),
        "n_replay_search": sum(item.origin == "search" for item in replay_items),
        "new_targets": new_targets,
        "replay_targets": replay_targets,
        "generated_definitions": generated_defs,
        "n_observations": len(state.observations),
        "n_verifications": sum(item.sample_count for item in state.observations),
        "n_point_updates": sum(
            observation_target_key(item, task) in key_to_mu
            for item in state.observations
        ),
        "phases": phases,
        "init": config.init,
        "learn_w": config.learn_w,
        "learn_r": config.learn_r,
        "learn_bias_network": config.learn_bias_network,
        "search_domain": search_domain,
    }
    if new_ellipsoid is not None:
        record["ellipsoid_center"] = new_ellipsoid.center.tolist()
        record["ellipsoid_chol"] = new_ellipsoid.chol.tolist()
    return state, new_bounds, record


def make_expansion_hook(
    *,
    ckpt,
    config: SearchExpandConfig,
    task: str,
    seed: int,
    bounds: np.ndarray,
    bounds_padding: float,
    seed_dir: str | Path,
    aabb_std_k: float = 0.0,
    search_domain: str = "aabb",
    raw_defs: Optional[Mapping[str, str]] = None,
    progress_factory: Optional[
        Callable[..., Optional[Callable[[dict[str, Any]], None]]]
    ] = None,
) -> Callable[[RunState], Optional[np.ndarray | tuple[np.ndarray, LatentEllipsoid]]]:
    """Return ``run_bo(..., on_after_batch=...)`` that expands every k batches."""

    disk = load_expansion_state(seed_dir)
    current_bounds = np.asarray(bounds, dtype=np.float32)

    def _on_after_batch(
        state: RunState,
    ) -> Optional[np.ndarray | tuple[np.ndarray, LatentEllipsoid]]:
        nonlocal disk, current_bounds
        n_batches = acquisition_batches_completed(state.observations)
        if not should_expand(
            expand_every=config.every,
            n_batches=n_batches,
            last_expanded_batches=disk.last_expanded_batches,
        ):
            return None
        round_index = disk.n_rounds + 1
        print(
            f"[expand] starting round {round_index} "
            f"(acquisition batches={n_batches}, every={config.every})",
            flush=True,
        )
        absorbed = replay_window_end(
            state.observations, disk.absorbed_through_index
        )
        _state, new_bounds, record = expand_search_subspace_window(
            ckpt=ckpt,
            state=state,
            config=config,
            bounds=current_bounds,
            task=task,
            seed=seed,
            round_index=round_index,
            absorbed_through_index=absorbed,
            bounds_padding=bounds_padding,
            aabb_std_k=aabb_std_k,
            search_domain=search_domain,
            progress_factory=progress_factory,
            raw_defs=raw_defs,
        )
        if not record.get("skipped"):
            append_expansion_record(seed_dir, record)
            disk = ExpansionDiskState(
                n_rounds=round_index,
                last_expanded_batches=n_batches,
                absorbed_through_index=len(state.observations),
            )
            current_bounds = np.asarray(new_bounds, dtype=np.float32)
            if config.mutates_checkpoint():
                save_expansion_weights(seed_dir, ckpt)
        else:
            disk = ExpansionDiskState(
                n_rounds=disk.n_rounds,
                last_expanded_batches=n_batches,
                absorbed_through_index=len(state.observations),
            )
        write_expansion_state(seed_dir, disk)
        print(
            f"[expand] round {round_index} done "
            f"(skipped={bool(record.get('skipped'))}, "
            f"new={record.get('n_new')}, replay={record.get('n_replay')})",
            flush=True,
        )
        if record.get("skipped"):
            return None
        if record.get("ellipsoid_center") is not None:
            return (
                current_bounds,
                LatentEllipsoid(
                    record["ellipsoid_center"], record["ellipsoid_chol"]
                ),
            )
        return current_bounds

    return _on_after_batch
