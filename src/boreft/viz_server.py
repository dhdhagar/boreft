"""HTTP runtime for exploring and activating a BOReFT bias subspace."""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from typing import Any, Callable, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from boreft.bias_tables import (
    attach_materialized_bias_tables,
    bias_predict_kwargs,
    encoder_definitions_path,
    predict_bias_vectors_for_words,
    predict_bias_vectors_from_raw_texts,
    save_bias_tables,
    stack_bias_vectors,
)
from boreft.data_utils import load_merged_run_config, system_prompt_from_cfg
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_TEST_N_SAMPLES,
    build_test_sets,
    target_normalizer,
    task_csv_paths,
)
from boreft.eval.plot_cluster_pca import resolve_word_cluster_labels
from boreft.eval.semantle import (
    DEFAULT_MAX_NEW_TOKENS,
    _get_intervention,
    generate_text,
    generate_texts_batch,
    load_eval_checkpoint,
)
from boreft.interactive import (
    GenerationMode,
    build_checkpoint_prompt,
    generate_base,
    strip_thinking_text,
    target_intervention_flags,
)
from boreft.learn_bias import (
    BatchLearnControl,
    BatchLearnConfig,
    delete_learned_biases,
    get_train_embeddings,
    learn_biases_batched,
    load_learned_biases,
    predict_bias_network_rows,
    save_learned_bias,
)
from boreft.pyreft.losses import (
    annealing_map_from_saved_cfg,
    format_linear_annealing_map_cli,
    resolve_linear_annealing_map,
    serialize_linear_annealing_map,
)
from boreft.subspace_viz import (
    Projection2D,
    bias_l2_norm,
    fit_projection,
    mean_bias_l2_norm,
    plot_bounds,
    projected_point,
)
from boreft.task_config import (
    definition_embedding_text,
    task_supports_fingerprints,
    task_supports_sdpo,
)
from boreft.text_similarity import (
    default_definitions_path,
    definition_lookup_for_cfg,
    definition_text_for_cfg,
    embedding_sim_per_text,
    load_raw_definitions,
    rdkit_map_path_for_cfg,
)

_PARAM_DELTA_EPS = 1e-12


def snapshot_module_state_dict(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """CPU clone of a module ``state_dict`` for later delta comparisons."""
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def snapshot_recovery_parameters(intervention: Any) -> dict[str, Any]:
    """Snapshot shared recovery params: W, R, bias network, and encoder LoRA."""
    snapshot: dict[str, Any] = {
        "W": None,
        "R": None,
        "bias_network": None,
        "encoder": None,
    }
    learned_source = getattr(intervention, "learned_source", None)
    if isinstance(learned_source, torch.nn.Module):
        snapshot["W"] = snapshot_module_state_dict(learned_source)
    rotate_layer = getattr(intervention, "rotate_layer", None)
    if isinstance(rotate_layer, torch.nn.Module):
        snapshot["R"] = snapshot_module_state_dict(rotate_layer)
    bias_network = getattr(intervention, "bias_network", None)
    if isinstance(bias_network, torch.nn.Module):
        snapshot["bias_network"] = snapshot_module_state_dict(bias_network)
    encoder = None
    getter = getattr(intervention, "get_semantic_encoder", None)
    if callable(getter):
        encoder = getter()
    if isinstance(encoder, torch.nn.Module):
        snapshot["encoder"] = {
            name: param.detach().cpu().clone()
            for name, param in encoder.named_parameters()
        }
    return snapshot


def state_dict_l2_delta(
    current: dict[str, torch.Tensor] | None,
    reference: dict[str, torch.Tensor] | None,
) -> dict[str, float | None]:
    """Relative and absolute Frobenius/L2 change between two state dicts."""
    if current is None or reference is None:
        return {"relative_l2": None, "absolute_l2": None}
    if set(current) != set(reference):
        return {"relative_l2": None, "absolute_l2": None}
    delta_sq = 0.0
    ref_sq = 0.0
    for key, ref_tensor in reference.items():
        cur_tensor = current[key]
        if cur_tensor.shape != ref_tensor.shape:
            return {"relative_l2": None, "absolute_l2": None}
        diff = cur_tensor.to(dtype=torch.float64) - ref_tensor.to(
            dtype=torch.float64
        )
        delta_sq += float(diff.pow(2).sum().item())
        ref_sq += float(ref_tensor.to(dtype=torch.float64).pow(2).sum().item())
    absolute = float(delta_sq**0.5)
    relative = absolute / (float(ref_sq**0.5) + _PARAM_DELTA_EPS)
    return {"relative_l2": relative, "absolute_l2": absolute}


def _component_change_metrics(
    *,
    current: dict[str, torch.Tensor] | None,
    reference: dict[str, torch.Tensor] | None,
    trained: bool,
) -> dict[str, Any]:
    if not trained:
        return {
            "trained": False,
            "status": "unchanged",
            "relative_l2": None,
            "absolute_l2": None,
        }
    delta = state_dict_l2_delta(current, reference)
    if delta["relative_l2"] is None:
        return {
            "trained": True,
            "status": "n/a",
            "relative_l2": None,
            "absolute_l2": None,
        }
    return {
        "trained": True,
        "status": "changed",
        "relative_l2": delta["relative_l2"],
        "absolute_l2": delta["absolute_l2"],
    }


def recovery_parameter_changes(
    *,
    current: dict[str, Any],
    previous: dict[str, Any],
    original: dict[str, Any],
    learn_W: bool,
    learn_R: bool,
    learn_bias_network: bool,
) -> dict[str, Any]:
    """Compare post-train params vs pre-run and original training snapshots."""
    trained = {
        "W": learn_W,
        "R": learn_R,
        "bias_network": learn_bias_network,
        "encoder": learn_bias_network,
    }

    def against(reference: dict[str, Any]) -> dict[str, Any]:
        return {
            name: _component_change_metrics(
                current=current.get(name),
                reference=reference.get(name),
                trained=trained[name],
            )
            for name in ("W", "R", "bias_network", "encoder")
        }

    return {
        "vs_previous": against(previous),
        "vs_original": against(original),
    }


class ActivateRequest(BaseModel):
    session_id: str
    kind: Literal["train", "test", "learned", "coordinate", "base", "zero"]
    point_id: str | None = None
    x: float | None = None
    y: float | None = None


class TextActivationRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=4096)


class ChatRequest(BaseModel):
    session_id: str
    text: str
    multi_turn: bool = False
    use_checkpoint_prompt: bool = False
    enable_thinking: bool = False
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0.0, le=10.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    compute_similarity: bool = False


class SessionRequest(BaseModel):
    session_id: str


class RecoveryScanRequest(BaseModel):
    scope: Literal["train", "test", "both"] = "test"
    n_samples: int = Field(default=25, ge=1, le=100)
    temperature: float = Field(default=1.0, gt=0.0, le=10.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=1024
    )
    seed: int | None = None


_POINT_SET_KEYS = frozenset(
    {
        "train_seen",
        "train_unseen",
        "test_seen",
        "test_unseen",
        "additional_seen",
        "additional_unseen",
    }
)


class ProjectionRequest(BaseModel):
    sets: list[str] = Field(min_length=1, max_length=6)


class AdditionalLearningTarget(BaseModel):
    target: str = Field(min_length=1, max_length=512)
    definition: str = Field(min_length=1, max_length=4096)


class RecoveryTrainRequest(BaseModel):
    point_ids: list[str] = Field(default_factory=list, max_length=1000)
    training_sets: list[str] = Field(default_factory=list, max_length=6)
    additional_targets: list[AdditionalLearningTarget] = Field(
        default_factory=list, max_length=1000
    )
    new_target: str | None = Field(default=None, max_length=512)
    new_definition: str | None = Field(default=None, max_length=4096)
    epochs: int | None = Field(default=None, ge=1, le=10_000)
    batch_size: int | None = Field(default=None, ge=1)
    lr: float | None = Field(default=None, gt=0.0)
    kl_beta: float | None = Field(default=None, ge=0.0)
    lambda_ce: float | None = Field(default=None, ge=0.0)
    lambda_sdpo: float | None = Field(default=None, ge=0.0)
    # None = inherit checkpoint map; "" = disable annealing for this job.
    linear_annealing_map: str | None = Field(default=None, max_length=2048)
    weight_decay_mode: Literal["none", "W_only", "b_only", "W_and_b"] | None = None
    wd_W: float | None = Field(default=None, ge=0.0)
    wd_b: float | None = Field(default=None, ge=0.0)
    lr_scheduler_type: str | None = Field(default=None, max_length=64)
    warmup_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    eval_epochs: int | None = Field(default=None, ge=1)
    stop_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    stop_threshold_min: float | None = Field(default=None, ge=0.0, le=1.0)
    stop_threshold_frac: float | None = Field(default=None, gt=0.0, le=1.0)
    # None inherits the checkpoint's metric (embed_sim unless it was trained
    # with a molecular one).
    eval_selection_metric: Literal["embed_sim", "rdkit_sim", "tfs"] | None = None
    learn_W: bool = False
    learn_R: bool = False
    learn_bias_network: bool = False
    include_previous_targets: bool = True
    force: bool = False


_RECOVERY_SCHEDULER_TYPES = frozenset(
    {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    }
)


class DeleteLearnedRequest(BaseModel):
    point_id: str | None = Field(default=None, max_length=256)
    target: str | None = Field(default=None, max_length=512)


@dataclass
class BrowserSession:
    mode: GenerationMode = "base"
    subspace: int | np.ndarray | None = None
    label: str = "Base model"
    target: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class VizRuntime:
    """Long-lived checkpoint, PCA state, and browser session registry."""

    @staticmethod
    def _item_original_split(item: dict[str, Any]) -> str:
        split = str(item.get("original_split") or "")
        if split in {"train", "test", "additional"}:
            return split
        # Backward compatibility with checkpoints written before durable
        # original_split metadata was introduced. A legacy "new" row is
        # ambiguous until the original test set is reconstructed.
        return "additional" if item.get("origin") == "new" else "train"

    @staticmethod
    def _set_key(original_split: str, seen: bool) -> str:
        split = (
            original_split
            if original_split in {"train", "test", "additional"}
            else "additional"
        )
        return f"{split}_{'seen' if seen else 'unseen'}"

    def __init__(
        self,
        checkpoint_dir: str,
        ckpt,
        *,
        model_name: str,
        low_rank_dim: int,
        semantle_dir: str | None = None,
        top_k: int = 100,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.ckpt = ckpt
        self.model_name = model_name
        self.low_rank_dim = low_rank_dim
        self.lock = threading.RLock()
        self.sessions: dict[str, BrowserSession] = {}
        self.jobs_lock = threading.Lock()
        self.recovery_jobs: dict[str, dict[str, Any]] = {}
        self.recovery_controls: dict[str, BatchLearnControl] = {}
        self.active_recovery_job_id: str | None = None
        self.recovery_status: dict[str, dict[str, Any]] = {}
        self.original_param_snapshot = self._capture_param_snapshot(ckpt)
        cfg = ckpt.saved_cfg or {}
        self.task = str(cfg.get("task", "semantle"))
        self.supports_text_activation = bool(cfg.get("add_bias_network"))
        self.raw_text_cache: dict[str, np.ndarray] = {}

        self.raw_definition_lookup: dict[str, str] = {}
        definition_paths = [
            cfg.get("definitions_path"),
            encoder_definitions_path(cfg),
            default_definitions_path(self.task),
        ]
        for path in definition_paths:
            if path and os.path.isfile(str(path)):
                # Prefer the checkpoint's configured definitions when files overlap.
                loaded = load_raw_definitions(str(path))
                self.raw_definition_lookup = {
                    **loaded,
                    **self.raw_definition_lookup,
                }
        self.raw_definition_by_normalized = {
            self._target_key(word): definition
            for word, definition in self.raw_definition_lookup.items()
        }

        self.word_ids = [int(item["id"]) for item in ckpt.items]
        self.words = [
            str(item.get("word", item.get("target", ckpt.words[index])))
            for index, item in enumerate(ckpt.items)
        ]
        self.item_original_splits = [
            self._item_original_split(item) for item in ckpt.items
        ]
        self.word_to_id = dict(zip(self.words, self.word_ids))
        self.train_biases = stack_bias_vectors(ckpt.reft_model, self.word_ids)
        self.projection_sets = sorted(
            {
                self._set_key(split, True)
                for split in self.item_original_splits
            }
        )
        self.projection = fit_projection(self.train_biases)

        try:
            word_to_cluster, self.label_source = resolve_word_cluster_labels(
                self.words, checkpoint_dir, semantle_dir, top_k
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[viz] cluster labels unavailable ({exc}); using one group", flush=True)
            word_to_cluster = {word: "all" for word in self.words}
            self.label_source = "none"

        self.word_clusters = {
            word: word_to_cluster.get(word, "unknown") for word in self.words
        }
        self.train_points: list[dict[str, Any]] = []
        self.additional_points: list[dict[str, Any]] = []
        self._rebuild_item_points()
        self.learned_records: list[dict[str, Any]] = []
        self.display_learned_records: list[dict[str, Any]] = []
        self.test_words: list[str] = []
        self.test_biases = np.zeros(
            (0, self.projection.bias_dim), dtype=np.float32
        )
        self.test_base_biases = self.test_biases.copy()
        self.test_item_ids: list[int | None] = []
        self.test_seen: list[bool] = []
        self.test_points: list[dict[str, Any]] = []
        self.test_error: str | None = None
        if self.supports_text_activation:
            self._initialize_test_points(cfg)
            self.projection_sets = sorted(
                {
                    self._set_key(split, True)
                    for split in self.item_original_splits
                }
            )
            self._rebuild_item_points()
            self._reproject_test_points()

    @cached_property
    def _target_key(self) -> Callable[[str], str]:
        """Key every target lookup in this server goes through.

        Lower-cased text for Semantle, RDKit canonical SMILES for molopt, where
        lower-casing would merge distinct molecules (cyclohexane into benzene).
        """
        return target_normalizer(getattr(self, "task", "semantle"))

    def _definition_for(self, word: str) -> str | None:
        return self.raw_definition_by_normalized.get(self._target_key(word))

    def _initialize_test_points(self, cfg: dict[str, Any]) -> None:
        """Restore the original held-out set and mark absorbed rows as seen."""
        try:
            original_train_words = [
                word
                for word, split in zip(self.words, self.item_original_splits)
                if split == "train"
            ]
            interp, extrap, _meta = build_test_sets(
                train_targets=original_train_words,
                csv_paths=task_csv_paths(cfg, task=self.task),
                test_n_samples=int(
                    cfg.get("test_n_samples", DEFAULT_TEST_N_SAMPLES)
                ),
                seed=int(cfg.get("seed", 42)),
                pca_var=float(
                    cfg.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR)
                ),
                task=self.task,
                train_embeddings=None,
            )
            split_by_word = {
                **{word: "interp" for word in interp},
                **{word: "extrap" for word in extrap},
            }
            test_words = [*interp, *extrap]
            test_norms = {self._target_key(word) for word in test_words}
            known_train_norms = {
                self._target_key(word)
                for word, split in zip(
                    self.words, self.item_original_splits
                )
                if split == "train"
            }
            for index, (item, word) in enumerate(
                zip(self.ckpt.items, self.words)
            ):
                if (
                    item.get("original_split") is not None
                    or item.get("origin") != "new"
                ):
                    continue
                word_norm = self._target_key(word)
                self.item_original_splits[index] = (
                    "train"
                    if word_norm in known_train_norms
                    else "test"
                    if word_norm in test_norms
                    else "additional"
                )
            for word, original_split in zip(
                self.words, self.item_original_splits
            ):
                if (
                    original_split == "test"
                    and self._target_key(word) not in test_norms
                ):
                    test_words.append(word)
                    test_norms.add(self._target_key(word))
                    split_by_word[word] = "absorbed"
            for word in test_words:
                definition = self._definition_for(word)
                if definition is not None and word not in self.raw_definition_lookup:
                    self.raw_definition_lookup[word] = definition
            extra_kwargs = bias_predict_kwargs(
                cfg,
                tokenizer=self.ckpt.tokenizer,
                raw_definition_lookup=self.raw_definition_lookup,
            )
            if cfg.get("bias_input_source") == "llm_encoder":
                test_words = [
                    word for word in test_words if word in self.raw_definition_lookup
                ]
            if not test_words:
                return
            item_index_by_norm = {
                self._target_key(word): index
                for index, (word, split) in enumerate(
                    zip(self.words, self.item_original_splits)
                )
                if split == "test"
            }
            unseen_words = [
                word
                for word in test_words
                if self._target_key(word) not in item_index_by_norm
            ]
            definition_lookup = definition_lookup_for_cfg(cfg)
            unseen_vectors = np.asarray(
                predict_bias_vectors_for_words(
                    self.ckpt.reft_model,
                    unseen_words,
                    definition_lookup=definition_lookup,
                    task=self.task,
                    **extra_kwargs,
                )
                if unseen_words
                else np.zeros((0, self.projection.bias_dim)),
                dtype=np.float32,
            )
            unseen_by_norm = {
                self._target_key(word): unseen_vectors[index]
                for index, word in enumerate(unseen_words)
            }
            rows: list[np.ndarray] = []
            item_ids: list[int | None] = []
            seen_flags: list[bool] = []
            for word in test_words:
                item_index = item_index_by_norm.get(self._target_key(word))
                if item_index is not None:
                    rows.append(
                        np.asarray(self.train_biases[item_index], dtype=np.float32)
                    )
                    item_ids.append(self.word_ids[item_index])
                    seen_flags.append(True)
                else:
                    rows.append(unseen_by_norm[self._target_key(word)])
                    item_ids.append(None)
                    seen_flags.append(False)
            self.test_words = test_words
            self.test_item_ids = item_ids
            self.test_seen = seen_flags
            self.test_biases = np.stack(rows, axis=0).astype(np.float32)
            self.test_base_biases = self.test_biases.copy()
            coordinates = self.projection.project(self.test_biases)
            self.test_points = [
                projected_point(
                    point_id=(
                        f"train:{item_id}"
                        if item_id is not None
                        else f"test:{index}"
                    ),
                    kind="test",
                    label=word,
                    coordinates=coordinate,
                    cluster=f"test-{split_by_word.get(word, 'extrap')}",
                    bias_norm=bias_l2_norm(bias),
                    metadata={
                        "split": split_by_word.get(word, "extrap"),
                        "original_split": "test",
                        "seen": seen,
                        "set_key": self._set_key("test", seen),
                        "vector_index": index,
                        **(
                            {"word_id": item_id}
                            if item_id is not None
                            else {}
                        ),
                        **(
                            {"definition": self._definition_for(word)}
                            if self._definition_for(word) is not None
                            else {}
                        ),
                    },
                )
                for index, (word, coordinate, bias, item_id, seen) in enumerate(
                    zip(
                        self.test_words,
                        coordinates,
                        self.test_biases,
                        self.test_item_ids,
                        self.test_seen,
                    )
                )
            ]
        except Exception as exc:
            # The map and chat remain useful when optional held-out data is absent.
            self.test_error = str(exc)
            print(f"[viz] test overlay unavailable: {exc}", flush=True)

    def _refresh_unseen_test_base_biases(self) -> None:
        """Re-predict unseen test rows after the bias network changes."""
        indices = [
            index
            for index, item_id in enumerate(self.test_item_ids)
            if item_id is None
        ]
        if not indices:
            return
        words = [self.test_words[index] for index in indices]
        definitions = {
            word: definition
            for word in words
            if (definition := self._definition_for(word)) is not None
        }
        refreshed, _ = predict_bias_network_rows(
            self.ckpt,
            words,
            definitions=definitions or None,
        )
        refreshed = np.asarray(refreshed, dtype=np.float32)
        if refreshed.shape != (len(indices), self.projection.bias_dim):
            raise ValueError(
                "bias-network refresh returned an unexpected test-bias shape"
            )
        base = np.asarray(self.test_base_biases, dtype=np.float32).copy()
        for row, index in zip(refreshed, indices):
            base[index] = row
        self.test_base_biases = base

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        *,
        model_name: str | None = None,
        cache_dir: str | None = None,
        layer: int | None = None,
        low_rank_dim: int | None = None,
        torch_dtype: str | None = None,
        semantle_dir: str | None = None,
        top_k: int | None = None,
        load_latest: bool = False,
    ) -> "VizRuntime":
        cfg = load_merged_run_config(checkpoint_dir)
        resolved_model = str(
            model_name or cfg.get("model_name") or "meta-llama/Llama-3.2-1B"
        )
        resolved_cache = cache_dir if cache_dir is not None else cfg.get("cache_dir")
        resolved_layer = int(layer if layer is not None else cfg.get("layer", 13))
        resolved_rank = int(
            low_rank_dim
            if low_rank_dim is not None
            else cfg.get("low_rank_dim", 64)
        )
        resolved_semantle_dir = (
            semantle_dir
            if semantle_dir is not None
            else cfg.get("semantle_dir")
        )
        resolved_top_k = int(
            top_k if top_k is not None else cfg.get("train_top_k", 100)
        )
        ckpt = load_eval_checkpoint(
            checkpoint_dir,
            resolved_model,
            resolved_layer,
            resolved_rank,
            resolved_cache,
            torch_dtype=torch_dtype,
            load_latest=load_latest,
        )
        return cls(
            checkpoint_dir,
            ckpt,
            model_name=resolved_model,
            low_rank_dim=resolved_rank,
            semantle_dir=resolved_semantle_dir,
            top_k=resolved_top_k,
        )

    def _learned_points(self) -> list[dict[str, Any]]:
        self.learned_records = load_learned_biases(self.checkpoint_dir)
        original_targets = {
            self._target_key(word) for word in [*self.words, *self.test_words]
        }
        latest: dict[str, dict[str, Any]] = {}
        for record in self.learned_records:
            key = self._target_key(str(record.get("target", "")))
            if key:
                latest[key] = record
        points: list[dict[str, Any]] = []
        self.display_learned_records = []
        for record in latest.values():
            if self._target_key(str(record.get("target", ""))) in original_targets:
                continue
            mu = np.asarray(record.get("mu"), dtype=np.float64)
            if mu.shape != (self.projection.bias_dim,) or not np.isfinite(mu).all():
                continue
            self.display_learned_records.append(record)
            coordinate = self.projection.project(mu)
            metadata = {
                key: record[key]
                for key in (
                    "definition",
                    "decode",
                    "final_sim",
                    "classification",
                    "closest_train_word",
                    "learned_at",
                )
                if key in record
            }
            metadata.update(
                {
                    "original_split": "additional",
                    "seen": True,
                    "set_key": "additional_seen",
                }
            )
            points.append(
                projected_point(
                    point_id=f"learned:{len(points)}",
                    kind="learned",
                    label=str(
                        record.get("target") or f"learned {len(points) + 1}"
                    ),
                    coordinates=coordinate,
                    cluster="learned",
                    bias_norm=bias_l2_norm(mu),
                    metadata=metadata,
                )
            )
        return points

    def _latest_learned_record(self, target: str) -> dict[str, Any] | None:
        target_norm = self._target_key(target)
        latest: dict[str, Any] | None = None
        for record in load_learned_biases(self.checkpoint_dir):
            if self._target_key(str(record.get("target", ""))) == target_norm:
                latest = record
        return latest

    def _resolve_learned_delete_target(
        self, *, point_id: str | None, target: str | None
    ) -> str:
        """Resolve a map point or explicit target into a learned-bias word."""
        explicit = (target or "").strip()
        if explicit:
            if self._latest_learned_record(explicit) is None:
                raise ValueError(f"no learned bias for target {explicit!r}")
            return explicit
        if not point_id:
            raise ValueError("provide point_id or target")
        if point_id.startswith("learned:"):
            self._learned_points()
            index = int(point_id.partition(":")[2])
            if index < 0 or index >= len(self.display_learned_records):
                raise ValueError(f"unknown learned point index {index}")
            record = self.display_learned_records[index]
            resolved = str(record.get("target") or "").strip()
            if not resolved:
                raise ValueError("learned point has no target")
            return resolved
        if point_id.startswith("item:"):
            spec = self._recovery_point(point_id)
            resolved = str(spec["target"]).strip()
            if self._latest_learned_record(resolved) is None:
                raise ValueError(
                    f"no learned bias overlay for {point_id} ({resolved!r})"
                )
            return resolved
        if point_id.startswith("train:") or point_id.startswith("test:"):
            spec = self._recovery_point(point_id)
            resolved = str(spec["target"]).strip()
            if self._latest_learned_record(resolved) is None:
                raise ValueError(
                    f"no learned bias overlay for {point_id} ({resolved!r})"
                )
            return resolved
        raise ValueError(
            "delete supports learned / train / test point ids, or an explicit target"
        )

    def delete_learned_target(
        self,
        *,
        point_id: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Remove persisted learned-bias rows for one target and refresh the map."""
        with self.jobs_lock:
            active = (
                self.recovery_jobs.get(self.active_recovery_job_id)
                if self.active_recovery_job_id is not None
                else None
            )
            if active and active["status"] in {"queued", "running"}:
                raise RuntimeError(
                    f"recovery job {self.active_recovery_job_id} is already running"
                )
        with self.lock:
            resolved = self._resolve_learned_delete_target(
                point_id=point_id, target=target
            )
            target_norm = self._target_key(resolved)
            is_train_overlay = target_norm in {
                self._target_key(word) for word in self.words
            }
            is_test_overlay = target_norm in {
                self._target_key(word) for word in self.test_words
            }
            selected_sets = set(self.projection_sets)
            removes_selected_row = (
                not is_train_overlay
                and (
                    (
                        is_test_overlay
                        and "test_seen" in selected_sets
                        and "test_unseen" not in selected_sets
                    )
                    or (
                        not is_test_overlay
                        and "additional_seen" in selected_sets
                    )
                )
            )
            if (
                removes_selected_row
                and len(self._projection_rows_for_sets(selected_sets)) <= 2
            ):
                raise ValueError(
                    "deleting this bias would leave the selected PCA sets "
                    "with fewer than two points; choose different PCA sets first"
                )
            removed = delete_learned_biases(
                self.checkpoint_dir, [resolved], task=self.task
            )
            if removed <= 0:
                raise ValueError(f"no learned bias for target {resolved!r}")
            for point_key, status in list(self.recovery_status.items()):
                label = None
                if point_key.startswith("train:"):
                    try:
                        word_id = int(point_key.partition(":")[2])
                        index = self.word_ids.index(word_id)
                        label = self.words[index]
                    except (ValueError, IndexError):
                        label = None
                elif point_key.startswith("test:"):
                    try:
                        index = int(point_key.partition(":")[2])
                        label = self.test_words[index]
                    except (ValueError, IndexError):
                        label = None
                if label is not None and self._target_key(label) == target_norm:
                    for key in (
                        "learned_converged",
                        "learned_decode",
                        "learned_steps",
                        "batch_mean_train_loss",
                        "learned_recall_n_samples",
                        "learned_temperature",
                        "recovery_evaluated",
                    ):
                        status.pop(key, None)
                    status["effective_converged"] = bool(
                        status.get("baseline_converged")
                    )
            for session in self.sessions.values():
                if (
                    session.target is not None
                    and self._target_key(session.target) == target_norm
                ):
                    session.mode = "base"
                    session.subspace = None
                    session.label = "Base model"
                    session.target = None
            affected_sets: set[str] = set()
            for word, split in zip(self.words, self.item_original_splits):
                if self._target_key(word) == target_norm:
                    affected_sets.add(self._set_key(split, True))
            if is_test_overlay:
                affected_sets.update({"test_seen", "test_unseen"})
            if not is_train_overlay and not is_test_overlay:
                affected_sets.add("additional_seen")
            geometry_refreshed = False
            if is_train_overlay or affected_sets & set(self.projection_sets):
                geometry_refreshed = self._refresh_subspace_geometry(force=True)
            elif is_test_overlay:
                self._reproject_test_points()
            self.raw_text_cache.clear()
            return {
                "target": resolved,
                "removed_records": removed,
                "geometry_refreshed": geometry_refreshed,
            }

    def _learned_bias_for_target(self, target: str) -> np.ndarray | None:
        record = self._latest_learned_record(target)
        if record is None:
            return None
        mu = np.asarray(record.get("mu"), dtype=np.float32)
        if mu.shape != (self.projection.bias_dim,) or not np.isfinite(mu).all():
            return None
        return mu

    def _effective_train_biases(self) -> np.ndarray:
        """Current train μ rows: learned overlay when present, else live stack."""
        stacked: np.ndarray | None = None
        rows: list[np.ndarray] = []
        for index, word in enumerate(self.words):
            recovered = self._learned_bias_for_target(word)
            if recovered is not None:
                rows.append(recovered)
                continue
            if stacked is None:
                stacked = np.asarray(
                    stack_bias_vectors(self.ckpt.reft_model, self.word_ids),
                    dtype=np.float32,
                )
            rows.append(np.asarray(stacked[index], dtype=np.float32))
        if not rows:
            return np.zeros((0, self.projection.bias_dim), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def _rebuild_item_points(self) -> None:
        """Rebuild exact checkpoint rows, preserving their original split."""
        coordinates = self.projection.project(self.train_biases)
        train_points: list[dict[str, Any]] = []
        additional_points: list[dict[str, Any]] = []
        for word_id, word, coordinate, bias, original_split in zip(
            self.word_ids,
            self.words,
            coordinates,
            self.train_biases,
            self.item_original_splits,
        ):
            if original_split == "test":
                # Original-test checkpoint rows are emitted by
                # _initialize_test_points / _reproject_test_points so they keep
                # the test marker style and do not appear twice.
                continue
            kind = "train" if original_split == "train" else "learned"
            point_id = (
                f"train:{word_id}"
                if original_split == "train"
                else f"item:{word_id}"
            )
            point = projected_point(
                point_id=f"train:{word_id}",
                kind=kind,
                label=word,
                coordinates=coordinate,
                cluster=(
                    self.word_clusters.get(word, "unknown")
                    if original_split == "train"
                    else "learned"
                ),
                bias_norm=bias_l2_norm(bias),
                metadata={
                    "word_id": word_id,
                    "original_split": original_split,
                    "seen": True,
                    "set_key": self._set_key(original_split, True),
                    **(
                        {"definition": self._definition_for(word)}
                        if self._definition_for(word) is not None
                        else {}
                    ),
                },
            )
            point["id"] = point_id
            if original_split == "train":
                train_points.append(point)
            else:
                additional_points.append(point)
        self.train_points = train_points
        self.additional_points = additional_points

    def _reproject_test_points(self) -> None:
        """Reproject test markers; prefer learned overlays over predicted biases."""
        if not self.test_words:
            self.test_points = []
            return
        meta_by_index = {
            int(point.get("metadata", {}).get("vector_index", index)): point
            for index, point in enumerate(self.test_points)
        }
        rows: list[np.ndarray] = []
        seen_flags: list[bool] = []
        for index, word in enumerate(self.test_words):
            recovered = self._learned_bias_for_target(word)
            if recovered is not None:
                rows.append(recovered)
                seen_flags.append(True)
            elif (
                index < len(self.test_item_ids)
                and self.test_item_ids[index] is not None
                and self.test_item_ids[index] in self.word_ids
            ):
                item_index = self.word_ids.index(self.test_item_ids[index])
                rows.append(
                    np.asarray(self.train_biases[item_index], dtype=np.float32)
                )
                seen_flags.append(True)
            else:
                rows.append(
                    np.asarray(
                        getattr(
                            self,
                            "test_base_biases",
                            self.test_biases,
                        )[index],
                        dtype=np.float32,
                    )
                )
                seen_flags.append(False)
        self.test_seen = seen_flags
        self.test_biases = np.stack(rows, axis=0).astype(np.float32)
        coordinates = self.projection.project(self.test_biases)
        self.test_points = [
            projected_point(
                point_id=meta_by_index.get(index, {}).get(
                    "id",
                    f"test:{index}",
                ),
                kind="test",
                label=word,
                coordinates=coordinate,
                cluster=meta_by_index.get(index, {}).get(
                    "cluster", "test-extrap"
                ),
                bias_norm=bias_l2_norm(bias),
                metadata={
                    **dict(
                        meta_by_index.get(index, {}).get(
                        "metadata",
                        {
                            "vector_index": index,
                            "original_split": "test",
                        },
                        )
                    ),
                    "seen": seen_flags[index],
                    "set_key": self._set_key(
                        "test", seen_flags[index]
                    ),
                },
            )
            for index, (word, coordinate, bias) in enumerate(
                zip(self.test_words, coordinates, self.test_biases)
            )
        ]

    def _projection_rows_for_sets(
        self, set_keys: set[str]
    ) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for bias, split in zip(self.train_biases, self.item_original_splits):
            if split == "test":
                continue
            if self._set_key(split, True) in set_keys:
                rows.append(np.asarray(bias, dtype=np.float32))
        for index, bias in enumerate(self.test_biases):
            seen = (
                self.test_seen[index]
                if index < len(self.test_seen)
                else False
            )
            if self._set_key("test", seen) in set_keys:
                rows.append(np.asarray(bias, dtype=np.float32))

        represented = {
            self._target_key(word) for word in [*self.words, *self.test_words]
        }
        if "additional_seen" in set_keys:
            for record in load_learned_biases(self.checkpoint_dir):
                target = str(record.get("target", ""))
                if self._target_key(target) in represented:
                    continue
                mu = np.asarray(record.get("mu"), dtype=np.float32)
                if (
                    mu.shape == (self.projection.bias_dim,)
                    and np.isfinite(mu).all()
                ):
                    rows.append(mu)
                    represented.add(self._target_key(target))
        return rows

    def set_projection_sets(self, sets: list[str]) -> dict[str, Any]:
        selected = set(sets)
        invalid = selected - _POINT_SET_KEYS
        if invalid:
            raise ValueError(
                f"unknown PCA sets: {', '.join(sorted(invalid))}"
            )
        with self.lock:
            rows = self._projection_rows_for_sets(selected)
            if len(rows) < 2:
                raise ValueError(
                    "PCA projection requires at least two available points"
                )
            self.projection_sets = sorted(selected)
            self.projection = fit_projection(np.stack(rows, axis=0))
            self._rebuild_item_points()
            self._reproject_test_points()
            self.raw_text_cache.clear()
        return {
            "sets": self.projection_sets,
            "n_points": len(rows),
        }

    def _refresh_subspace_geometry(self, *, force: bool = False) -> bool:
        """Refit PCA and rebuild map points when effective train biases changed.

        Returns ``True`` when the PCA plane was refit. When train biases are
        unchanged, test overlays are still reprojected onto the current plane so
        recovered test points move to their learned μ.
        """
        previous = np.asarray(self.train_biases, dtype=np.float32)
        try:
            updated = self._effective_train_biases()
        except Exception as exc:
            print(
                f"[viz] subspace geometry refresh skipped ({exc})",
                flush=True,
            )
            return False
        if (
            not force
            and updated.shape == previous.shape
            and np.allclose(updated, previous, rtol=1e-5, atol=1e-6)
        ):
            self._reproject_test_points()
            self.raw_text_cache.clear()
            return False
        self.train_biases = updated
        # Refresh exact/recovered test rows before selecting vectors for PCA;
        # their temporary coordinates are rebuilt again after the fit.
        self._reproject_test_points()
        rows = self._projection_rows_for_sets(set(self.projection_sets))
        if len(rows) >= 2:
            self.projection = fit_projection(np.stack(rows, axis=0))
        self._rebuild_item_points()
        self._reproject_test_points()
        self.raw_text_cache.clear()
        if len(rows) < 2:
            print(
                "[viz] PCA refit skipped: selected sets now contain fewer "
                "than two points",
                flush=True,
            )
            return False
        print(
            f"[viz] refit PCA on {len(rows)} selected biases "
            "(geometry refreshed)",
            flush=True,
        )
        return True

    def _prior_training_specs(self) -> list[dict[str, Any]]:
        """Return one latest bias row for every target trained so far."""
        latest: dict[str, dict[str, Any]] = {}
        for record in load_learned_biases(self.checkpoint_dir):
            key = self._target_key(str(record.get("target", "")))
            if key:
                latest[key] = record

        specs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, target in enumerate(self.words):
            key = self._target_key(target)
            record = latest.get(key)
            learned_mu = (
                np.asarray(record.get("mu"), dtype=np.float32)
                if record is not None
                else None
            )
            warm_start = (
                learned_mu
                if learned_mu is not None
                and learned_mu.shape == (self.projection.bias_dim,)
                and np.isfinite(learned_mu).all()
                else self.train_biases[index].copy()
            )
            specs.append(
                {
                    "id": None,
                    "target": target,
                    "definition": (
                        str(record.get("definition"))
                        if record is not None and record.get("definition")
                        else self._definition_for(target)
                    ),
                    "warm_start": warm_start,
                    "source": "original",
                    "previous_record": record,
                }
            )
            seen.add(key)

        for key, record in latest.items():
            if key in seen:
                continue
            mu = np.asarray(record.get("mu"), dtype=np.float32)
            if (
                mu.shape != (self.projection.bias_dim,)
                or not np.isfinite(mu).all()
            ):
                continue
            specs.append(
                {
                    "id": None,
                    "target": str(record["target"]),
                    "definition": record.get("definition")
                    or self._definition_for(str(record["target"])),
                    "warm_start": mu,
                    "source": (
                        "new"
                        if record.get("new_target")
                        else "recovery"
                    ),
                    "previous_record": record,
                }
            )
            seen.add(key)
        return specs

    def _new_target_spec(self, target: str, definition: str) -> dict[str, Any]:
        target = target.strip()
        definition = definition.strip()
        if not target or not definition:
            raise ValueError("new target and definition must both be non-empty")
        prior_targets = {
            self._target_key(spec["target"]) for spec in self._prior_training_specs()
        }
        if self._target_key(target) in prior_targets:
            raise ValueError(f"target {target!r} has already been trained")

        cfg = self.ckpt.saved_cfg or {}
        definition_lookup = definition_lookup_for_cfg(cfg)
        if cfg.get("use_definition_embeds"):
            effective_definition = definition_text_for_cfg(
                cfg, target, definition
            )
            definition_lookup = {
                **(definition_lookup or {}),
                target: definition_embedding_text(
                    self.task, target, effective_definition
                ),
            }
        raw_lookup = {**self.raw_definition_lookup, target: definition}
        extra_kwargs = bias_predict_kwargs(
            cfg,
            tokenizer=self.ckpt.tokenizer,
            raw_definition_lookup=raw_lookup,
        )
        predicted = predict_bias_vectors_for_words(
            self.ckpt.reft_model,
            [target],
            definition_lookup=definition_lookup,
            task=self.task,
            **extra_kwargs,
        )[0]
        return {
            "id": f"new:{uuid.uuid4().hex}",
            "target": target,
            "definition": definition,
            "warm_start": np.asarray(predicted, dtype=np.float32),
            "subspace": np.asarray(predicted, dtype=np.float32),
            "source": "new",
            "previous_record": None,
        }

    @staticmethod
    def _capture_param_snapshot(ckpt: Any) -> dict[str, Any]:
        """Snapshot W / R / bias network at load time for vs-original deltas."""
        try:
            intervention = _get_intervention(ckpt.reft_model)
        except Exception:
            return {
                "W": None,
                "R": None,
                "bias_network": None,
                "encoder": None,
            }
        return snapshot_recovery_parameters(intervention)

    def _save_derived_checkpoint(
        self,
        *,
        learn_W: bool,
        learn_R: bool,
        learn_bias_network: bool,
        include_previous_targets: bool,
        n_targets: int,
        parameter_changes: dict[str, Any] | None = None,
    ) -> str:
        """Copy the active checkpoint and replace its intervention weights."""
        source = os.path.abspath(self.checkpoint_dir)
        parent = os.path.dirname(source)
        stem = os.path.basename(source.rstrip(os.sep))
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        destination = os.path.join(
            parent, f"{stem}-recovery-{stamp}-{uuid.uuid4().hex[:6]}"
        )
        shutil.copytree(source, destination)

        intervention_dir = os.path.join(destination, "intervenable_model")
        if os.path.isdir(intervention_dir):
            shutil.rmtree(intervention_dir)
        self.ckpt.reft_model.save_intervention(
            save_directory=intervention_dir,
            include_model=False,
        )

        copied_tables = os.path.join(destination, "bias_tables.pt")
        for config_name in (
            "intervention_config.json",
            "training_config.json",
        ):
            config_path = os.path.join(destination, config_name)
            if not os.path.isfile(config_path):
                continue
            with open(config_path, encoding="utf-8") as file:
                config = json.load(file)
            for key, value in list(config.items()):
                if not isinstance(value, str) or not os.path.isabs(value):
                    continue
                try:
                    inside_source = os.path.commonpath(
                        [source, value]
                    ) == source
                except ValueError:
                    inside_source = False
                if inside_source:
                    config[key] = os.path.join(
                        destination, os.path.relpath(value, source)
                    )
            config["output_dir"] = destination
            if os.path.isfile(copied_tables):
                config["bias_tables_path"] = copied_tables
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
                file.write("\n")

        recovery_info: dict[str, Any] = {
            "parent_checkpoint": source,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "learn_W": learn_W,
            "learn_R": learn_R,
            "learn_bias_network": learn_bias_network,
            "include_previous_targets": include_previous_targets,
            "n_targets": n_targets,
        }
        if parameter_changes is not None:
            recovery_info["parameter_changes"] = parameter_changes
        with open(
            os.path.join(destination, "recovery_info.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(recovery_info, file, indent=2)
            file.write("\n")
        return destination

    def _persist_recovery_resolved_config(
        self,
        destination: str,
        resolved_train_config: dict[str, Any] | None,
    ) -> None:
        """Write recovery-resolved settings into derived configs + saved_cfg.

        Ensures the next recovery inherits the annealing map (or disable),
        coefficient overrides, and stop criteria actually used for this job.
        """
        if not resolved_train_config:
            return
        persist_keys = (
            "linear_annealing_map",
            "lambda_ce",
            "lambda_sdpo",
            "kl_beta",
            "stop_threshold",
            "stop_threshold_min",
            "stop_threshold_frac",
            "eval_selection_metric",
            "eval_epochs",
        )
        updates = {
            key: resolved_train_config[key]
            for key in persist_keys
            if key in resolved_train_config
        }
        if not updates:
            return
        for config_name in ("intervention_config.json", "training_config.json"):
            config_path = os.path.join(destination, config_name)
            if not os.path.isfile(config_path):
                continue
            with open(config_path, encoding="utf-8") as file:
                config = json.load(file)
            config.update(updates)
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
                file.write("\n")
        if self.ckpt.saved_cfg is not None:
            self.ckpt.saved_cfg.update(updates)

    def _points_with_recovery(
        self, points: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        latest_learned: dict[str, dict[str, Any]] = {}
        for record in self.learned_records:
            key = self._target_key(str(record.get("target", "")))
            if key and record.get("learned_at", "") >= latest_learned.get(
                key, {}
            ).get("learned_at", ""):
                latest_learned[key] = record

        enriched: list[dict[str, Any]] = []
        for point in points:
            status = dict(self.recovery_status.get(point["id"], {}))
            learned = latest_learned.get(self._target_key(point["label"]))
            if learned is not None:
                if "converged" in learned:
                    status["optimizer_converged"] = bool(
                        learned.get("converged")
                    )
                if "recovery_recall_converged" in learned:
                    status["recovery_evaluated"] = True
                    status["learned_converged"] = bool(
                        learned["recovery_recall_converged"]
                    )
                status["learned_decode"] = learned.get("decode")
                status["learned_at"] = learned.get("learned_at")
            if status:
                status["effective_converged"] = bool(
                    status.get("baseline_converged")
                    or status.get("learned_converged")
                )
                metadata = {**point["metadata"], "recovery": status}
                enriched.append({**point, "metadata": metadata})
            else:
                enriched.append(point)
        return enriched

    def map_payload(self) -> dict[str, Any]:
        with self.lock:
            primary_points = [
                *self.train_points,
                *self.additional_points,
                *self._learned_points(),
            ]
            points = self._points_with_recovery(
                [*primary_points, *self.test_points]
            )
            ratios = np.zeros(2, dtype=np.float64)
            ratios[: self.projection.projected_dim] = (
                self.projection.explained_variance_ratio
            )
            original_train_rows = [
                bias
                for bias, split in zip(
                    self.train_biases, self.item_original_splits
                )
                if split == "train"
            ]
            avg_train_bias_norm = mean_bias_l2_norm(
                original_train_rows or self.train_biases
            )
            set_counts = {
                key: sum(
                    1
                    for point in points
                    if point.get("metadata", {}).get("set_key") == key
                )
                for key in sorted(_POINT_SET_KEYS)
            }
            ckpt_anneal_map = annealing_map_from_saved_cfg(
                self.ckpt.saved_cfg or {}
            )
            return {
                "points": points,
                "bounds": plot_bounds(points),
                "primary_bounds": plot_bounds(primary_points),
                "axes": {
                    "x": f"PC1 ({ratios[0] * 100:.1f}%)",
                    "y": f"PC2 ({ratios[1] * 100:.1f}%)",
                    "avg_train_bias_norm": avg_train_bias_norm,
                    "projection_sets": list(self.projection_sets),
                    "set_counts": set_counts,
                },
                "metadata": {
                    "checkpoint_dir": self.checkpoint_dir,
                    "model_name": self.model_name,
                    "bias_dim": self.projection.bias_dim,
                    "avg_train_bias_norm": avg_train_bias_norm,
                    "projection_sets": list(self.projection_sets),
                    "set_counts": set_counts,
                    "bias_type": str(
                        (self.ckpt.saved_cfg or {}).get("bias_type", "vae")
                    ),
                    "kl_beta": float(
                        (self.ckpt.saved_cfg or {}).get("kl_beta", 0.0) or 0.0
                    ),
                    "lambda_ce": float(
                        (self.ckpt.saved_cfg or {}).get("lambda_ce", 1.0) or 0.0
                    ),
                    "lambda_sdpo": float(
                        (self.ckpt.saved_cfg or {}).get("lambda_sdpo", 0.0) or 0.0
                    ),
                    "linear_annealing_map": (
                        serialize_linear_annealing_map(ckpt_anneal_map)
                        if ckpt_anneal_map
                        else None
                    ),
                    "linear_annealing_map_cli": format_linear_annealing_map_cli(
                        ckpt_anneal_map
                    ),
                    "supports_sdpo": task_supports_sdpo(self.task),
                    "task": self.task,
                    "supports_fingerprints": task_supports_fingerprints(
                        self.task
                    ),
                    "weight_decay_mode": str(
                        (self.ckpt.saved_cfg or {}).get("weight_decay_mode", "none")
                        or "none"
                    ),
                    "wd_W": float(
                        (self.ckpt.saved_cfg or {}).get("wd_W", 0.0) or 0.0
                    ),
                    "wd_b": float(
                        (self.ckpt.saved_cfg or {}).get("wd_b", 0.0) or 0.0
                    ),
                    "lr_scheduler_type": str(
                        (self.ckpt.saved_cfg or {}).get(
                            "lr_scheduler_type", "linear"
                        )
                        or "linear"
                    ),
                    "warmup_ratio": float(
                        (self.ckpt.saved_cfg or {}).get("warmup_ratio", 0.0)
                        or 0.0
                    ),
                    "train_points": len(self.train_points),
                    "learned_points": (
                        len(primary_points) - len(self.train_points)
                    ),
                    "test_points": len(self.test_points),
                    "test_error": self.test_error,
                    "supports_text_activation": self.supports_text_activation,
                    "supports_recovery": self.supports_text_activation,
                    "cluster_source": self.label_source,
                    "click_note": (
                        "Named points activate their exact full-dimensional bias; "
                        "empty-space clicks use the inverse PCA plane."
                    ),
                },
            }

    def _session(self, session_id: str) -> BrowserSession:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        return self.sessions.setdefault(session_id, BrowserSession())

    @staticmethod
    def _selection_payload(session: BrowserSession) -> dict[str, Any]:
        return {
            "mode": session.mode,
            "label": session.label,
            "target": session.target,
        }

    def activate(
        self,
        session_id: str,
        *,
        kind: Literal[
            "train", "test", "learned", "coordinate", "base", "zero"
        ],
        point_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            session = self._session(session_id)
            if kind == "base":
                session.mode = "base"
                session.subspace = None
                session.label = "Base model"
                session.target = None
            elif kind == "zero":
                session.mode = "zero_bias"
                session.subspace = np.zeros(
                    self.projection.bias_dim, dtype=np.float32
                )
                session.label = "Zero bias"
                session.target = None
            elif kind == "coordinate":
                if x is None or y is None or not np.isfinite([x, y]).all():
                    raise ValueError("coordinate activation requires finite x and y")
                session.mode = "intervention"
                session.subspace = self.projection.inverse(x, y).astype(np.float32)
                session.label = f"PC coordinate ({x:.3f}, {y:.3f})"
                session.target = None
            elif kind == "train":
                if not point_id or not point_id.startswith("train:"):
                    raise ValueError("train activation requires a train point_id")
                word_id = int(point_id.partition(":")[2])
                if word_id not in self.word_ids:
                    raise ValueError(f"unknown training word id {word_id}")
                index = self.word_ids.index(word_id)
                target = self.words[index]
                recovered = self._learned_bias_for_target(target)
                session.mode = "intervention"
                session.subspace = recovered if recovered is not None else word_id
                session.label = (
                    f"{target} (recovered)" if recovered is not None else target
                )
                session.target = target
            elif kind == "test":
                if not point_id:
                    raise ValueError("test activation requires a point_id")
                if point_id.startswith("train:"):
                    word_id = int(point_id.partition(":")[2])
                    if word_id not in self.word_ids:
                        raise ValueError(f"unknown seen test word id {word_id}")
                    item_index = self.word_ids.index(word_id)
                    target = self.words[item_index]
                    recovered = self._learned_bias_for_target(target)
                    session.mode = "intervention"
                    session.subspace = (
                        recovered if recovered is not None else word_id
                    )
                    session.label = target
                    session.target = target
                    return self._selection_payload(session)
                if not point_id.startswith("test:"):
                    raise ValueError("test activation requires a test point_id")
                index = int(point_id.partition(":")[2])
                if index < 0 or index >= len(self.test_words):
                    raise ValueError(f"unknown test point index {index}")
                target = self.test_words[index]
                recovered = self._learned_bias_for_target(target)
                session.mode = "intervention"
                session.subspace = (
                    recovered
                    if recovered is not None
                    else self.test_biases[index].copy()
                )
                session.label = (
                    f"{target} (recovered)" if recovered is not None else target
                )
                session.target = target
            elif kind == "learned":
                if point_id and point_id.startswith("item:"):
                    word_id = int(point_id.partition(":")[2])
                    if word_id not in self.word_ids:
                        raise ValueError(f"unknown additional word id {word_id}")
                    item_index = self.word_ids.index(word_id)
                    target = self.words[item_index]
                    recovered = self._learned_bias_for_target(target)
                    session.mode = "intervention"
                    session.subspace = (
                        recovered if recovered is not None else word_id
                    )
                    session.label = target
                    session.target = target
                    return self._selection_payload(session)
                if not point_id or not point_id.startswith("learned:"):
                    raise ValueError("learned activation requires a learned point_id")
                self._learned_points()
                index = int(point_id.partition(":")[2])
                if index < 0 or index >= len(self.display_learned_records):
                    raise ValueError(f"unknown learned point index {index}")
                record = self.display_learned_records[index]
                mu = np.asarray(record.get("mu"), dtype=np.float32)
                if mu.shape != (self.projection.bias_dim,) or not np.isfinite(mu).all():
                    raise ValueError("learned point has an invalid bias vector")
                session.mode = "intervention"
                session.subspace = mu
                session.label = str(record.get("target") or f"learned {index + 1}")
                session.target = str(record.get("target") or "") or None
            return self._selection_payload(session)

    def activate_text(self, session_id: str, text: str) -> dict[str, Any]:
        """Predict and activate a bias from arbitrary text, like `/target-text`."""
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("text must not be empty")
        if not self.supports_text_activation:
            raise ValueError(
                "arbitrary text activation requires a bias-network checkpoint"
            )
        with self.lock:
            bias = self.raw_text_cache.get(raw_text)
            if bias is None:
                cfg = self.ckpt.saved_cfg or {}
                encoder_kwargs = bias_predict_kwargs(
                    cfg,
                    tokenizer=self.ckpt.tokenizer,
                    raw_definition_lookup=self.raw_definition_lookup,
                )
                # Raw text has no definition to look up, so drop that key and keep
                # only what predict_bias_vectors_from_raw_texts accepts.
                raw_kwargs = {
                    key: encoder_kwargs[key]
                    for key in (
                        "tokenizer",
                        "encoder_max_length",
                        "encoder_layer_index",
                        "embed_model",
                    )
                    if key in encoder_kwargs
                }
                bias = np.asarray(
                    predict_bias_vectors_from_raw_texts(
                        self.ckpt.reft_model,
                        [raw_text],
                        task=self.task,
                        **raw_kwargs,
                    )[0],
                    dtype=np.float32,
                )
                self.raw_text_cache[raw_text] = bias
            session = self._session(session_id)
            session.mode = "intervention"
            session.subspace = bias.copy()
            session.label = f"text: {raw_text}"
            session.target = None
            x, y = self.projection.project(bias)
            return {
                **self._selection_payload(session),
                "point": {"x": float(x), "y": float(y)},
            }

    def _recovery_point(self, point_id: str) -> dict[str, Any]:
        if point_id.startswith("train:"):
            word_id = int(point_id.partition(":")[2])
            if word_id not in self.word_ids:
                raise ValueError(f"unknown training word id {word_id}")
            index = self.word_ids.index(word_id)
            target = self.words[index]
            original_split = self.item_original_splits[index]
            previous = self._latest_learned_record(target)
            recovered = self._learned_bias_for_target(target)
            return {
                "id": point_id,
                "kind": (
                    "test" if original_split == "test" else "train"
                ),
                "target": target,
                "definition": (
                    previous.get("definition")
                    if previous is not None and previous.get("definition")
                    else self._definition_for(target)
                ),
                "subspace": recovered if recovered is not None else word_id,
                "warm_start": (
                    recovered
                    if recovered is not None
                    else self.train_biases[index].copy()
                ),
                "source": (
                    "original"
                    if original_split == "train"
                    else "recovery"
                ),
                "previous_record": previous,
            }
        if point_id.startswith("item:"):
            word_id = int(point_id.partition(":")[2])
            if word_id not in self.word_ids:
                raise ValueError(f"unknown additional word id {word_id}")
            index = self.word_ids.index(word_id)
            target = self.words[index]
            previous = self._latest_learned_record(target)
            recovered = self._learned_bias_for_target(target)
            return {
                "id": point_id,
                "kind": "learned",
                "target": target,
                "definition": (
                    previous.get("definition")
                    if previous is not None and previous.get("definition")
                    else self._definition_for(target)
                ),
                "subspace": recovered if recovered is not None else word_id,
                "warm_start": (
                    recovered
                    if recovered is not None
                    else self.train_biases[index].copy()
                ),
                "source": "additional",
                "previous_record": previous,
            }
        if point_id.startswith("learned:"):
            self._learned_points()
            index = int(point_id.partition(":")[2])
            if index < 0 or index >= len(self.display_learned_records):
                raise ValueError(f"unknown learned point index {index}")
            record = self.display_learned_records[index]
            target = str(record.get("target") or "").strip()
            mu = np.asarray(record.get("mu"), dtype=np.float32)
            if (
                not target
                or mu.shape != (self.projection.bias_dim,)
                or not np.isfinite(mu).all()
            ):
                raise ValueError("learned point has invalid target or bias")
            return {
                "id": point_id,
                "kind": "learned",
                "target": target,
                "definition": (
                    record.get("definition")
                    or self._definition_for(target)
                ),
                "subspace": mu.copy(),
                "warm_start": mu.copy(),
                "source": "additional",
                "previous_record": record,
            }
        if point_id.startswith("test:"):
            index = int(point_id.partition(":")[2])
            if index < 0 or index >= len(self.test_words):
                raise ValueError(f"unknown test point index {index}")
            target = self.test_words[index]
            previous = self._latest_learned_record(target)
            recovered = self._learned_bias_for_target(target)
            return {
                "id": point_id,
                "kind": "test",
                "target": target,
                "definition": (
                    previous.get("definition")
                    if previous is not None and previous.get("definition")
                    else self._definition_for(target)
                ),
                "subspace": (
                    recovered
                    if recovered is not None
                    else self.test_biases[index].copy()
                ),
                "warm_start": (
                    recovered
                    if recovered is not None
                    else self.test_biases[index].copy()
                ),
                "source": "recovery",
                "previous_record": previous,
            }
        raise ValueError(
            "recovery supports train, test, and learned point ids only"
        )

    def _point_ids_for_sets(self, set_keys: set[str]) -> list[str]:
        points = [
            *self.train_points,
            *self.test_points,
            *self.additional_points,
            *self._learned_points(),
        ]
        ids: list[str] = []
        seen_ids: set[str] = set()
        for point in points:
            if (
                point.get("metadata", {}).get("set_key") in set_keys
                and point["id"] not in seen_ids
            ):
                ids.append(str(point["id"]))
                seen_ids.add(str(point["id"]))
        return ids

    def _new_recovery_job(
        self,
        *,
        job_type: str,
        total: int,
        worker,
    ) -> dict[str, Any]:
        with self.jobs_lock:
            if self.active_recovery_job_id is not None:
                active = self.recovery_jobs.get(self.active_recovery_job_id)
                if active and active["status"] in {"queued", "running"}:
                    raise RuntimeError(
                        f"recovery job {self.active_recovery_job_id} is already running"
                    )
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "type": job_type,
                "status": "queued",
                "progress": 0,
                "total": total,
                "message": "Queued",
                "created_at": time.time(),
                "updated_at": time.time(),
                "result": None,
                "error": None,
            }
            self.recovery_jobs[job_id] = job
            self.active_recovery_job_id = job_id

        def run() -> None:
            self._update_recovery_job(
                job_id, status="running", message="Starting"
            )
            try:
                result = worker(job_id)
                self._update_recovery_job(
                    job_id,
                    status="completed",
                    message="Completed",
                    result=result,
                )
            except Exception as exc:
                self._update_recovery_job(
                    job_id,
                    status="failed",
                    message="Failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                with self.jobs_lock:
                    if self.active_recovery_job_id == job_id:
                        self.active_recovery_job_id = None

        threading.Thread(
            target=run,
            name=f"boreft-{job_type}-{job_id[:8]}",
            daemon=True,
        ).start()
        return dict(job)

    def _update_recovery_job(self, job_id: str, **updates: Any) -> None:
        with self.jobs_lock:
            job = self.recovery_jobs[job_id]
            job.update(updates)
            job["updated_at"] = time.time()

    def recovery_job(self, job_id: str) -> dict[str, Any]:
        with self.jobs_lock:
            if job_id not in self.recovery_jobs:
                raise ValueError(f"unknown recovery job {job_id}")
            return dict(self.recovery_jobs[job_id])

    def control_recovery_training(
        self, job_id: str, action: Literal["pause", "resume", "stop"]
    ) -> dict[str, Any]:
        with self.jobs_lock:
            job = self.recovery_jobs.get(job_id)
            control = self.recovery_controls.get(job_id)
            if job is None:
                raise ValueError(f"unknown recovery job {job_id}")
            if job.get("type") != "train":
                raise ValueError("only training jobs can be controlled")
            if job.get("status") not in {"queued", "running"}:
                raise ValueError("training job is no longer running")
            if control is None or job.get("phase") != "training":
                raise ValueError("training controls are not active yet")

        if action == "pause":
            control.pause()
            updates = {
                "training_state": "pausing",
                "message": "Pausing after the current step",
            }
        elif action == "resume":
            control.resume()
            updates = {
                "training_state": "running",
                "message": "Resuming training",
            }
        else:
            control.stop()
            updates = {
                "training_state": "stopping",
                "message": "Stopping after the current step",
            }
        self._update_recovery_job(job_id, **updates)
        return self.recovery_job(job_id)

    def start_recovery_scan(
        self,
        *,
        scope: Literal["train", "test", "both"],
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        if not self.supports_text_activation:
            raise ValueError("recovery requires a bias-network checkpoint")
        point_ids: list[str] = []
        if scope in {"train", "both"}:
            point_ids.extend(point["id"] for point in self.train_points)
        if scope in {"test", "both"}:
            point_ids.extend(point["id"] for point in self.test_points)
        point_ids = list(dict.fromkeys(point_ids))
        if not point_ids:
            raise ValueError(f"no {scope} points are available to scan")
        specs = [self._recovery_point(point_id) for point_id in point_ids]
        scan_seed = int(
            seed if seed is not None else (self.ckpt.saved_cfg or {}).get("seed", 42)
        )

        def worker(job_id: str) -> dict[str, Any]:
            misses: list[str] = []
            with self.lock:
                random.seed(scan_seed)
                np.random.seed(scan_seed)
                torch.manual_seed(scan_seed)
                common = {
                    "max_new_tokens": max_new_tokens,
                    "position": (self.ckpt.saved_cfg or {}).get("position", "l1"),
                    "assistant_suffix": self.ckpt.assistant_suffix,
                    "from_chat_template": self.ckpt.from_chat_template,
                    "intervention_token_id": self.ckpt.intervention_token_id,
                    "content_span": self.ckpt.content_span,
                }
                for index, spec in enumerate(specs, start=1):
                    self._update_recovery_job(
                        job_id,
                        progress=index - 1,
                        message=f"Decoding {spec['target']} ({index}/{len(specs)})",
                    )
                    greedy = generate_text(
                        self.ckpt.reft_model,
                        self.ckpt.tokenizer,
                        self.ckpt.prompt,
                        spec["subspace"],
                        use_sample=False,
                        **common,
                    )
                    samples = generate_texts_batch(
                        self.ckpt.reft_model,
                        self.ckpt.tokenizer,
                        self.ckpt.prompt,
                        spec["subspace"],
                        n_samples=n_samples,
                        use_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        **common,
                    )
                    target_norm = self._target_key(spec["target"])
                    greedy_hit = self._target_key(greedy) == target_norm
                    recall_hit = target_norm in {
                        self._target_key(sample) for sample in samples
                    }
                    self.recovery_status[spec["id"]] = {
                        "baseline_scanned": True,
                        "baseline_converged": recall_hit,
                        "baseline_greedy_hit": greedy_hit,
                        "baseline_decode": greedy,
                        "n_samples": n_samples,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "seed": scan_seed,
                    }
                    if not recall_hit:
                        misses.append(spec["id"])
                    self._update_recovery_job(job_id, progress=index)
            return {
                "scope": scope,
                "n_scanned": len(specs),
                "n_misses": len(misses),
                "miss_point_ids": misses,
                "n_samples": n_samples,
                "temperature": temperature,
            }

        return self._new_recovery_job(
            job_type="scan", total=len(specs), worker=worker
        )

    def start_recovery_training(
        self,
        point_ids: list[str],
        *,
        training_sets: list[str] | None = None,
        additional_targets: list[dict[str, str]] | None = None,
        new_target: str | None,
        new_definition: str | None,
        epochs: int | None,
        batch_size: int | None,
        lr: float | None,
        kl_beta: float | None,
        eval_epochs: int | None,
        learn_W: bool,
        learn_R: bool,
        learn_bias_network: bool,
        include_previous_targets: bool,
        force: bool,
        stop_threshold: float | None = None,
        stop_threshold_min: float | None = None,
        stop_threshold_frac: float | None = None,
        eval_selection_metric: str | None = None,
        lambda_ce: float | None = None,
        lambda_sdpo: float | None = None,
        linear_annealing_map: str | None = None,
        weight_decay_mode: str | None = None,
        wd_W: float | None = None,
        wd_b: float | None = None,
        lr_scheduler_type: str | None = None,
        warmup_ratio: float | None = None,
    ) -> dict[str, Any]:
        if not self.supports_text_activation:
            raise ValueError("recovery requires a bias-network checkpoint")
        selected_sets = set(training_sets or [])
        invalid_sets = selected_sets - _POINT_SET_KEYS
        if invalid_sets:
            raise ValueError(
                f"unknown training sets: {', '.join(sorted(invalid_sets))}"
            )
        extra_targets = list(additional_targets or [])
        if (
            not point_ids
            and not selected_sets
            and not extra_targets
            and not (new_target or "").strip()
        ):
            raise ValueError(
                "select at least one point/set or provide a new target"
            )
        if bool((new_target or "").strip()) != bool(
            (new_definition or "").strip()
        ):
            raise ValueError(
                "new target and definition must be provided together"
            )
        if weight_decay_mode is not None and weight_decay_mode not in {
            "none",
            "W_only",
            "b_only",
            "W_and_b",
        }:
            raise ValueError(
                "weight_decay_mode must be one of "
                "none, W_only, b_only, W_and_b"
            )
        if (
            lr_scheduler_type is not None
            and lr_scheduler_type not in _RECOVERY_SCHEDULER_TYPES
        ):
            raise ValueError(
                "lr_scheduler_type must be one of "
                + ", ".join(sorted(_RECOVERY_SCHEDULER_TYPES))
            )
        if linear_annealing_map is not None and str(linear_annealing_map).strip():
            saved_cfg = self.ckpt.saved_cfg or {}
            try:
                resolve_linear_annealing_map(
                    raw_map=linear_annealing_map,
                    defaults={
                        "kl_beta": float(
                            kl_beta
                            if kl_beta is not None
                            else saved_cfg.get("kl_beta", 0.0) or 0.0
                        ),
                        "lambda_ce": float(
                            lambda_ce
                            if lambda_ce is not None
                            else saved_cfg.get("lambda_ce", 1.0) or 0.0
                        ),
                        "lambda_sdpo": float(
                            lambda_sdpo
                            if lambda_sdpo is not None
                            else saved_cfg.get("lambda_sdpo", 0.0) or 0.0
                        ),
                    },
                )
            except ValueError as exc:
                raise ValueError(f"invalid linear_annealing_map: {exc}") from exc
        has_stop = stop_threshold is not None or stop_threshold_min is not None
        if (
            stop_threshold_frac is not None
            and stop_threshold_frac < 1.0 - 1e-12
            and stop_threshold_min is None
        ):
            raise ValueError(
                "stop_threshold_frac < 1 requires stop_threshold_min "
                "(per-target embed_sim tau)"
            )
        if stop_threshold_frac is not None and stop_threshold_frac < 1.0 - 1e-12:
            has_stop = True
        if (
            eval_selection_metric is not None
            and eval_selection_metric != "embed_sim"
            and not task_supports_fingerprints(self.task)
        ):
            raise ValueError(
                f"eval_selection_metric={eval_selection_metric!r} requires a "
                f"task whose targets are molecules (task={self.task!r})"
            )
        inherited_eval = (self.ckpt.saved_cfg or {}).get("eval_epochs")
        effective_eval = eval_epochs
        if effective_eval is None and inherited_eval:
            try:
                effective_eval = int(inherited_eval)
            except (TypeError, ValueError):
                effective_eval = None
        if has_stop and not effective_eval:
            raise ValueError(
                "stop embed-sim thresholds require eval every N epochs "
                "(set it in the recovery panel or inherit a positive "
                "eval_epochs from the checkpoint)"
            )

        set_point_ids = self._point_ids_for_sets(selected_sets)
        requested_ids = list(dict.fromkeys([*point_ids, *set_point_ids]))
        requested_specs = [
            self._recovery_point(point_id) for point_id in requested_ids
        ]
        set_target_norms = {
            self._target_key(self._recovery_point(point_id)["target"])
            for point_id in set_point_ids
        }
        records = load_learned_biases(self.checkpoint_dir)
        learned_converged = {
            self._target_key(str(record.get("target", "")))
            for record in records
            if record.get("recovery_recall_converged")
        }
        if not force:
            requested_specs = [
                spec
                for spec in requested_specs
                if (
                    self._target_key(spec["target"]) in set_target_norms
                    or (
                        not self.recovery_status.get(spec["id"], {}).get(
                            "baseline_converged"
                        )
                        and self._target_key(spec["target"])
                        not in learned_converged
                    )
                )
            ]
        for extra in extra_targets:
            target = str(extra.get("target", "")).strip()
            definition = str(extra.get("definition", "")).strip()
            if not target or not definition:
                raise ValueError(
                    "every additional target requires target and definition"
                )
            with self.lock:
                requested_specs.append(
                    self._new_target_spec(target, definition)
                )
        if (new_target or "").strip():
            with self.lock:
                requested_specs.append(
                    self._new_target_spec(
                        str(new_target), str(new_definition)
                    )
                )
        if not requested_specs:
            raise ValueError("all selected points are already converged")

        training_by_target: dict[str, dict[str, Any]] = {}
        if include_previous_targets:
            for spec in self._prior_training_specs():
                training_by_target[self._target_key(spec["target"])] = spec
        for spec in requested_specs:
            training_by_target[self._target_key(spec["target"])] = spec
        training_specs = list(training_by_target.values())

        saved_cfg = self.ckpt.saved_cfg or {}
        effective_sdpo = float(
            lambda_sdpo
            if lambda_sdpo is not None
            else saved_cfg.get("lambda_sdpo", 0.0) or 0.0
        )
        sdpo_active = effective_sdpo > 0.0 and task_supports_sdpo(self.task)
        missing_definition = (
            [
                spec["target"]
                for spec in training_specs
                if not spec["definition"]
            ]
            if sdpo_active
            else []
        )
        if missing_definition:
            preview = ", ".join(missing_definition[:5])
            raise ValueError(
                f"{len(missing_definition)} training targets lack definitions "
                f"(e.g. {preview})"
            )

        def worker(job_id: str) -> dict[str, Any]:
            outcomes: list[dict[str, Any]] = []
            training_control = BatchLearnControl()
            with self.jobs_lock:
                self.recovery_controls[job_id] = training_control
            self._update_recovery_job(
                job_id,
                phase="training",
                training_state="running",
                eval_history=[],
                loss_history=[],
            )
            with self.lock:
                targets = [spec["target"] for spec in training_specs]
                warm_start = np.stack(
                    [
                        np.asarray(spec["warm_start"], dtype=np.float32)
                        for spec in training_specs
                    ]
                )
                definitions = {
                    spec["target"]: spec["definition"]
                    for spec in training_specs
                    if spec["definition"]
                }

                def progress(metrics: dict[str, Any]) -> None:
                    step = int(metrics.get("step", 0))
                    max_steps = max(1, int(metrics.get("max_steps", 1)))
                    event = str(metrics.get("event", "log"))
                    if event == "paused":
                        self._update_recovery_job(
                            job_id,
                            progress=step,
                            total=max_steps,
                            message=f"Paused at step {step}/{max_steps}",
                            metrics=metrics,
                            training_state="paused",
                        )
                        return
                    if event == "resumed":
                        self._update_recovery_job(
                            job_id,
                            progress=step,
                            total=max_steps,
                            message=f"Resumed at step {step}/{max_steps}",
                            metrics=metrics,
                            training_state="running",
                        )
                        return
                    if event == "stopping":
                        self._update_recovery_job(
                            job_id,
                            progress=step,
                            total=max_steps,
                            message=f"Stopping at step {step}/{max_steps}",
                            metrics=metrics,
                            training_state="stopping",
                        )
                        return
                    parts = [f"Training step {step}/{max_steps}"]
                    if "loss" in metrics:
                        parts.append(f"loss={metrics['loss']:.4f}")
                    if "learning_rate" in metrics:
                        parts.append(f"lr={metrics['learning_rate']:.2e}")
                    if "grad_norm" in metrics:
                        parts.append(f"grad={metrics['grad_norm']:.3f}")
                    if "eval_avg_embed_sim" in metrics:
                        parts.append(
                            f"eval_sim={metrics['eval_avg_embed_sim']:.3f}"
                        )
                    if "eval_embed_sim_gte_stop_min" in metrics:
                        parts.append(
                            f"eval_frac={metrics['eval_embed_sim_gte_stop_min']:.3f}"
                        )
                    if "eval_n_recovered" in metrics:
                        parts.append(
                            f"eval_recovered={int(metrics['eval_n_recovered'])}"
                        )
                    training_state = "running"
                    if training_control.stop_requested:
                        training_state = "stopping"
                    elif training_control.paused:
                        training_state = "pausing"
                    updates: dict[str, Any] = {
                        "progress": step,
                        "total": max_steps,
                        "message": " · ".join(parts),
                        "metrics": metrics,
                        "phase": "training",
                        "training_state": training_state,
                    }
                    with self.jobs_lock:
                        job = self.recovery_jobs[job_id]
                        if event == "log" and "loss" in metrics:
                            loss_history = list(job.get("loss_history") or [])
                            entry: dict[str, Any] = {
                                "step": step,
                                "loss": float(metrics["loss"]),
                                "epoch": float(metrics.get("epoch", 0.0)),
                            }
                            if "learning_rate" in metrics:
                                entry["learning_rate"] = float(
                                    metrics["learning_rate"]
                                )
                            if "grad_norm" in metrics:
                                entry["grad_norm"] = float(
                                    metrics["grad_norm"]
                                )
                            loss_history.append(entry)
                            updates["loss_history"] = loss_history
                    if "eval_avg_embed_sim" in metrics:
                        history = list(job.get("eval_history") or [])
                        entry = {
                            "epoch": float(metrics.get("epoch", 0.0)),
                            "step": step,
                            "avg_embed_sim": float(
                                metrics["eval_avg_embed_sim"]
                            ),
                            "min_embed_sim": (
                                float(metrics["eval_min_embed_sim"])
                                if "eval_min_embed_sim" in metrics
                                else None
                            ),
                            "n_recovered": (
                                int(metrics["eval_n_recovered"])
                                if "eval_n_recovered" in metrics
                                else None
                            ),
                        }
                        if "eval_embed_sim_gte_stop_min" in metrics:
                            entry["embed_sim_gte_stop_min"] = float(
                                metrics["eval_embed_sim_gte_stop_min"]
                            )
                        # The metric that actually drives early stopping; equal
                        # to the embed_sim pair unless a molecular metric was
                        # selected.
                        for metric_key, entry_key in (
                            ("eval_selection_sim", "selection_sim"),
                            ("eval_selection_sim_min", "selection_sim_min"),
                            (
                                "eval_selection_gte_stop_min",
                                "selection_gte_stop_min",
                            ),
                        ):
                            if metric_key in metrics:
                                entry[entry_key] = float(metrics[metric_key])
                        history.append(entry)
                        updates["eval_history"] = history
                        job.update(updates)
                        job["updated_at"] = time.time()

                empty_snapshot = {
                    "W": None,
                    "R": None,
                    "bias_network": None,
                    "encoder": None,
                }
                try:
                    intervention = _get_intervention(self.ckpt.reft_model)
                    pre_param_snapshot = snapshot_recovery_parameters(
                        intervention
                    )
                except Exception:
                    intervention = None
                    pre_param_snapshot = dict(empty_snapshot)
                original_snapshot = getattr(
                    self, "original_param_snapshot", None
                )
                if original_snapshot is None:
                    original_snapshot = dict(empty_snapshot)
                    self.original_param_snapshot = original_snapshot
                if not any(original_snapshot.values()):
                    self.original_param_snapshot = {
                        key: (
                            {
                                name: tensor.detach().cpu().clone()
                                for name, tensor in value.items()
                            }
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in pre_param_snapshot.items()
                    }

                learned = learn_biases_batched(
                    self.ckpt,
                    targets=targets,
                    warm_start_mu=warm_start,
                    definitions=definitions,
                    config=BatchLearnConfig(
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        kl_beta=kl_beta,
                        lambda_ce=lambda_ce,
                        lambda_sdpo=lambda_sdpo,
                        linear_annealing_map=linear_annealing_map,
                        weight_decay_mode=weight_decay_mode,
                        wd_W=wd_W,
                        wd_b=wd_b,
                        lr_scheduler_type=lr_scheduler_type,
                        warmup_ratio=warmup_ratio,
                        eval_epochs=eval_epochs,
                        stop_threshold=stop_threshold,
                        stop_threshold_min=stop_threshold_min,
                        stop_threshold_frac=stop_threshold_frac,
                        eval_selection_metric=eval_selection_metric,
                        inherit_stop_threshold=False,
                        eval_batch_size=int(
                            (self.ckpt.saved_cfg or {}).get(
                                "full_eval_batch_size", 32
                            )
                            or 32
                        ),
                        eval_max_new_tokens=int(
                            (self.ckpt.saved_cfg or {}).get(
                                "full_eval_max_new_tokens",
                                DEFAULT_MAX_NEW_TOKENS,
                            )
                            or DEFAULT_MAX_NEW_TOKENS
                        ),
                        learn_W=learn_W,
                        learn_R=learn_R,
                        learn_bias_network=learn_bias_network,
                    ),
                    show_progress=False,
                    progress_callback=progress,
                    training_control=training_control,
                )
                if intervention is not None:
                    post_param_snapshot = snapshot_recovery_parameters(
                        intervention
                    )
                else:
                    post_param_snapshot = dict(empty_snapshot)
                parameter_changes = recovery_parameter_changes(
                    current=post_param_snapshot,
                    previous=pre_param_snapshot,
                    original=self.original_param_snapshot,
                    learn_W=learn_W,
                    learn_R=learn_R,
                    learn_bias_network=learn_bias_network,
                )
                self._update_recovery_job(
                    job_id,
                    progress=0,
                    total=len(requested_specs),
                    message=(
                        "Evaluating requested biases "
                        f"(0/{len(requested_specs)})"
                    ),
                    metrics={
                        "mean_train_loss": learned.mean_train_loss,
                        "steps": learned.steps,
                    },
                    phase="evaluating",
                    training_state=(
                        "stopped" if learned.stopped_early else "completed"
                    ),
                    eval_history=list(learned.eval_history),
                )
                learned_index = {
                    self._target_key(target): index
                    for index, target in enumerate(learned.targets)
                }
                records_to_save: dict[str, dict[str, Any]] = {}
                learned_at = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                for spec in training_specs:
                    key = self._target_key(spec["target"])
                    index = learned_index[key]
                    previous = dict(spec.get("previous_record") or {})
                    records_to_save[key] = {
                        **previous,
                        "target": spec["target"],
                        "definition": spec["definition"],
                        "target_norm": key,
                        "mu": np.asarray(learned.mu[index]).tolist(),
                        "mu_pred": np.asarray(
                            learned.mu_pred[index]
                        ).tolist(),
                        "logvar": (
                            np.asarray(learned.logvar[index]).tolist()
                            if learned.logvar is not None
                            else None
                        ),
                        "bias_dim": int(learned.bias_dim),
                        "steps": learned.steps,
                        "batch_training": True,
                        "mean_train_loss": learned.mean_train_loss,
                        "loss_config": learned.loss_config,
                        "resolved_train_config": (
                            learned.resolved_train_config
                        ),
                        "training_source": spec["source"],
                        "new_target": spec["source"] == "new",
                        "learned_at": learned_at,
                    }

                for eval_index, spec in enumerate(
                    requested_specs, start=1
                ):
                    bias_index = learned_index[
                        self._target_key(spec["target"])
                    ]
                    self._update_recovery_job(
                        job_id,
                        progress=eval_index - 1,
                        total=len(requested_specs),
                        message=(
                            f"Evaluating {spec['target']} "
                            f"({eval_index}/{len(requested_specs)})"
                        ),
                    )
                    status = self.recovery_status.setdefault(spec["id"], {})
                    n_samples = int(status.get("n_samples", 25))
                    temperature = float(status.get("temperature", 1.0))
                    top_p = float(status.get("top_p", 1.0))
                    max_new_tokens = int(
                        status.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
                    )
                    post_seed = (
                        int(
                            status.get(
                                "seed",
                                (self.ckpt.saved_cfg or {}).get("seed", 42),
                            )
                        )
                        + 20_000
                        + eval_index
                    )
                    random.seed(post_seed)
                    np.random.seed(post_seed)
                    torch.manual_seed(post_seed)
                    common = {
                        "max_new_tokens": max_new_tokens,
                        "position": (self.ckpt.saved_cfg or {}).get(
                            "position", "l1"
                        ),
                        "assistant_suffix": self.ckpt.assistant_suffix,
                        "from_chat_template": self.ckpt.from_chat_template,
                        "intervention_token_id": self.ckpt.intervention_token_id,
                        "content_span": self.ckpt.content_span,
                    }
                    post_greedy = generate_text(
                        self.ckpt.reft_model,
                        self.ckpt.tokenizer,
                        self.ckpt.prompt,
                        learned.mu[bias_index],
                        use_sample=False,
                        **common,
                    )
                    post_samples = generate_texts_batch(
                        self.ckpt.reft_model,
                        self.ckpt.tokenizer,
                        self.ckpt.prompt,
                        learned.mu[bias_index],
                        n_samples=n_samples,
                        use_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        **common,
                    )
                    target_norm = self._target_key(spec["target"])
                    recall_converged = target_norm in {
                        self._target_key(sample) for sample in post_samples
                    }
                    greedy_hit = self._target_key(post_greedy) == target_norm
                    records_to_save[target_norm].update(
                        {
                            "batch_greedy_hit": greedy_hit,
                            "decode": post_greedy,
                            "recovery_recall_converged": recall_converged,
                            "recovery_greedy_hit": greedy_hit,
                            "recovery_n_samples": n_samples,
                            "recovery_temperature": temperature,
                            "recovery_top_p": top_p,
                            "recovery_post_seed": post_seed,
                        }
                    )
                    status.update(
                        {
                            "learned_converged": recall_converged,
                            "learned_decode": post_greedy,
                            "learned_steps": learned.steps,
                            "batch_mean_train_loss": learned.mean_train_loss,
                            "learned_recall_n_samples": n_samples,
                            "learned_temperature": temperature,
                        }
                    )
                    outcomes.append(
                        {
                            "point_id": spec["id"],
                            "target": spec["target"],
                            "converged": recall_converged,
                            "decode": post_greedy,
                            "steps": learned.steps,
                        }
                    )
                    self._update_recovery_job(
                        job_id, progress=eval_index
                    )

                self._update_recovery_job(
                    job_id,
                    progress=0,
                    total=1,
                    message="Saving derived checkpoint",
                )
                derived_checkpoint = self._save_derived_checkpoint(
                    learn_W=learn_W,
                    learn_R=learn_R,
                    learn_bias_network=learn_bias_network,
                    include_previous_targets=include_previous_targets,
                    n_targets=len(training_specs),
                    parameter_changes=parameter_changes,
                )
                if learn_bias_network:
                    latest_existing = {
                        self._target_key(str(record.get("target", ""))): record
                        for record in records
                        if record.get("target")
                    }
                    original_definitions = {
                        target: (
                            self._definition_for(target)
                            or (
                                latest_existing.get(
                                    self._target_key(target), {}
                                ).get("definition")
                            )
                        )
                        for target in self.words
                    }
                    refreshed_mu, refreshed_logvar = (
                        predict_bias_network_rows(
                            self.ckpt,
                            self.words,
                            {
                                target: definition
                                for target, definition in (
                                    original_definitions.items()
                                )
                                if definition
                            },
                        )
                    )
                    original_mu = torch.as_tensor(
                        refreshed_mu, dtype=torch.float32
                    )
                    original_logvar = (
                        torch.as_tensor(
                            refreshed_logvar,
                            dtype=torch.float32,
                        )
                        if refreshed_logvar is not None
                        else None
                    )
                    refreshed_tables_path = save_bias_tables(
                        os.path.join(derived_checkpoint, "bias_tables.pt"),
                        original_mu,
                        original_logvar,
                        metadata={
                            "recovery_refresh": True,
                            "num_words": len(self.words),
                        },
                    )
                    for config_name in (
                        "intervention_config.json",
                        "training_config.json",
                    ):
                        config_path = os.path.join(
                            derived_checkpoint, config_name
                        )
                        if not os.path.isfile(config_path):
                            continue
                        with open(config_path, encoding="utf-8") as file:
                            checkpoint_config = json.load(file)
                        checkpoint_config[
                            "bias_tables_path"
                        ] = refreshed_tables_path
                        with open(
                            config_path, "w", encoding="utf-8"
                        ) as file:
                            json.dump(checkpoint_config, file, indent=2)
                            file.write("\n")
                    intervention = list(
                        self.ckpt.reft_model.interventions.values()
                    )[0]
                    if isinstance(intervention, (list, tuple)):
                        intervention = intervention[0]
                    attach_materialized_bias_tables(
                        intervention, original_mu, original_logvar
                    )
                    for index, target in enumerate(self.words):
                        key = self._target_key(target)
                        record = {
                            **latest_existing.get(key, {}),
                            **records_to_save.get(key, {}),
                        }
                        record.update(
                            {
                                "target": target,
                                "definition": original_definitions[target],
                                "target_norm": key,
                                "mu": refreshed_mu[index].tolist(),
                                "mu_pred": record.get(
                                    "mu_pred", refreshed_mu[index].tolist()
                                ),
                                "logvar": (
                                    refreshed_logvar[index].tolist()
                                    if refreshed_logvar is not None
                                    else None
                                ),
                                "bias_dim": int(learned.bias_dim),
                                "steps": learned.steps,
                                "batch_training": True,
                                "mean_train_loss": learned.mean_train_loss,
                                "loss_config": learned.loss_config,
                                "resolved_train_config": (
                                    learned.resolved_train_config
                                ),
                                "training_source": "original",
                                "bias_network_refresh": True,
                                "new_target": False,
                                "learned_at": learned_at,
                            }
                        )
                        records_to_save[key] = record
                for record in records_to_save.values():
                    save_learned_bias(derived_checkpoint, record)
                self.checkpoint_dir = derived_checkpoint
                self._persist_recovery_resolved_config(
                    derived_checkpoint, learned.resolved_train_config
                )
                if self.ckpt.saved_cfg is not None:
                    self.ckpt.saved_cfg["output_dir"] = derived_checkpoint
                    copied_tables = os.path.join(
                        derived_checkpoint, "bias_tables.pt"
                    )
                    if os.path.isfile(copied_tables):
                        self.ckpt.saved_cfg[
                            "bias_tables_path"
                        ] = copied_tables
                for spec in requested_specs:
                    if spec["source"] == "new":
                        self.raw_definition_lookup[spec["target"]] = spec[
                            "definition"
                        ]
                        self.raw_definition_by_normalized[
                            self._target_key(spec["target"])
                        ] = spec["definition"]
                if learn_bias_network:
                    self._refresh_unseen_test_base_biases()
                train_biases_may_change = (
                    learn_bias_network
                    or include_previous_targets
                    or any(
                        spec["source"] == "original" for spec in requested_specs
                    )
                    or (
                        bool(
                            {"test_seen", "test_unseen"}
                            & set(self.projection_sets)
                        )
                        and any(
                            spec.get("kind") == "test"
                            for spec in requested_specs
                        )
                    )
                    or (
                        "additional_seen" in self.projection_sets
                        and any(
                            spec["source"] in {"new", "additional"}
                            for spec in requested_specs
                        )
                    )
                )
                if train_biases_may_change:
                    self._update_recovery_job(
                        job_id,
                        phase="refitting_pca",
                        progress=0,
                        total=1,
                        message="Refitting bias PCA…",
                        training_state=(
                            "stopped"
                            if learned.stopped_early
                            else "completed"
                        ),
                    )
                    geometry_refreshed = self._refresh_subspace_geometry(
                        force=True
                    )
                else:
                    # Recovered test / new overlays still move on the current plane.
                    self._reproject_test_points()
                    self.raw_text_cache.clear()
                    geometry_refreshed = False
                self._update_recovery_job(
                    job_id,
                    phase="finalizing",
                    progress=1,
                    total=1,
                    message=(
                        "Derived checkpoint saved"
                        + (
                            " · PCA refit"
                            if geometry_refreshed
                            else ""
                        )
                    ),
                )
            return {
                "n_trained": len(training_specs),
                "n_evaluated": len(outcomes),
                "n_converged": sum(item["converged"] for item in outcomes),
                "steps": learned.steps,
                "mean_train_loss": learned.mean_train_loss,
                "eval_history": learned.eval_history,
                "loss_history": (
                    list(
                        (
                            self.recovery_jobs.get(job_id) or {}
                        ).get("loss_history")
                        or []
                    )
                ),
                "resolved_train_config": learned.resolved_train_config,
                "stopped_early": learned.stopped_early,
                "learn_W": learn_W,
                "learn_R": learn_R,
                "learn_bias_network": learn_bias_network,
                "include_previous_targets": include_previous_targets,
                "training_sets": sorted(selected_sets),
                "checkpoint_dir": derived_checkpoint,
                "geometry_refreshed": geometry_refreshed,
                "parameter_changes": parameter_changes,
                "outcomes": outcomes,
            }

        return self._new_recovery_job(
            job_type="train", total=1, worker=worker
        )

    def clear_history(self, session_id: str) -> dict[str, int]:
        with self.lock:
            session = self._session(session_id)
            cleared = len(session.history)
            session.history.clear()
            return {"cleared_messages": cleared}

    def similarity_vs_target(
        self, generated: str, target: str
    ) -> tuple[dict[str, float], str | None]:
        """Score a decode against a known target using task metrics.

        Metrics are computed independently so a RDKit/map failure still returns
        any scores that succeeded (typically ``embed_sim``, and often ``tfs``).
        Returns ``(scores, error)`` where ``error`` joins per-metric failures.
        """
        gen = strip_thinking_text(generated)
        scores: dict[str, float] = {}
        errors: list[str] = []

        try:
            embed_arr = embedding_sim_per_text(
                [target], [gen], task=self.task
            )
            scores["embed_sim"] = float(embed_arr[0])
        except Exception as exc:
            errors.append(f"embed_sim: {exc}")

        if task_supports_fingerprints(self.task):
            from boreft.chem import rdkit_sim_per_text, tanimoto_sim_per_text

            try:
                tfs_arr = tanimoto_sim_per_text([target], [gen])
                scores["tfs"] = float(tfs_arr[0])
            except Exception as exc:
                errors.append(f"tfs: {exc}")

            try:
                rdkit_arr = rdkit_sim_per_text(
                    [target],
                    [gen],
                    map_path=rdkit_map_path_for_cfg(
                        self.ckpt.saved_cfg or {}
                    ),
                )
                scores["rdkit_sim"] = float(rdkit_arr[0])
            except Exception as exc:
                errors.append(f"rdkit_sim: {exc}")

        return scores, ("; ".join(errors) if errors else None)

    def chat(
        self,
        session_id: str,
        *,
        text: str,
        multi_turn: bool,
        use_checkpoint_prompt: bool,
        enable_thinking: bool,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        compute_similarity: bool = False,
    ) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text must not be empty")
        # Snapshot target inside the lock; score outside so embedding/chem work
        # does not block map/activate for other browser sessions.
        score_target: str | None = None
        with self.lock:
            session = self._session(session_id)
            prompt, from_chat, content_span = build_checkpoint_prompt(
                self.ckpt,
                user_text=text,
                history=session.history,
                accumulate_history=multi_turn,
                use_checkpoint_prompt=use_checkpoint_prompt,
                generation_mode=session.mode,
                model_name=self.model_name,
                enable_thinking=enable_thinking,
                system_prompt=system_prompt_from_cfg(self.ckpt.saved_cfg),
            )
            if session.mode == "base":
                generated = generate_base(
                    self.ckpt.reft_model.model,
                    self.ckpt.tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    from_chat_template=from_chat,
                    assistant_suffix=self.ckpt.assistant_suffix,
                )
            else:
                if session.subspace is None:
                    raise ValueError("select a bias point before using intervention mode")
                generated = generate_text(
                    self.ckpt.reft_model,
                    self.ckpt.tokenizer,
                    prompt,
                    session.subspace,
                    max_new_tokens=max_new_tokens,
                    use_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    position=(self.ckpt.saved_cfg or {}).get("position", "l1"),
                    assistant_suffix=self.ckpt.assistant_suffix,
                    from_chat_template=from_chat,
                    intervention_token_id=self.ckpt.intervention_token_id,
                    content_span=content_span,
                )

            if multi_turn and not use_checkpoint_prompt:
                session.history.extend(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": generated},
                    ]
                )
            seen, is_target = target_intervention_flags(
                generated, session.target, self.word_to_id
            )
            if compute_similarity and session.target:
                score_target = session.target
            payload: dict[str, Any] = {
                "text": generated,
                "seen": seen,
                "is_target": is_target,
                "selection": self._selection_payload(session),
                "history_messages": len(session.history),
            }
        if score_target is not None:
            # Keep the successful decode visible even when some/all metrics fail
            # (missing embed model, bad RDKit map, etc.).
            scores, similarity_error = self.similarity_vs_target(
                generated, score_target
            )
            if scores:
                payload["similarity"] = scores
            if similarity_error:
                payload["similarity_error"] = similarity_error
        return payload


def create_app(runtime: VizRuntime):
    """Create the FastAPI application around an initialized runtime."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ImportError(
            "The visualization server requires fastapi and uvicorn."
        ) from exc

    app = FastAPI(title="BOReFT Subspace Explorer")

    def _bad_request(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (
            resources.files("boreft")
            .joinpath("static/subspace_viz.html")
            .read_text(encoding="utf-8")
        )
        return HTMLResponse(html)

    @app.get("/api/map")
    def get_map():
        return runtime.map_payload()

    @app.post("/api/activate")
    def activate(request: ActivateRequest):
        try:
            return runtime.activate(
                request.session_id,
                kind=request.kind,
                point_id=request.point_id,
                x=request.x,
                y=request.y,
            )
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/activate-text")
    def activate_text(request: TextActivationRequest):
        try:
            return runtime.activate_text(request.session_id, request.text)
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/learned/delete")
    def delete_learned(request: DeleteLearnedRequest):
        try:
            return runtime.delete_learned_target(
                point_id=request.point_id,
                target=request.target,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/recovery/scan")
    def start_recovery_scan(request: RecoveryScanRequest):
        try:
            return runtime.start_recovery_scan(
                scope=request.scope,
                n_samples=request.n_samples,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                seed=request.seed,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/recovery/train")
    def start_recovery_training(request: RecoveryTrainRequest):
        try:
            return runtime.start_recovery_training(
                request.point_ids,
                training_sets=request.training_sets,
                additional_targets=[
                    {
                        "target": target.target,
                        "definition": target.definition,
                    }
                    for target in request.additional_targets
                ],
                new_target=request.new_target,
                new_definition=request.new_definition,
                epochs=request.epochs,
                batch_size=request.batch_size,
                lr=request.lr,
                kl_beta=request.kl_beta,
                lambda_ce=request.lambda_ce,
                lambda_sdpo=request.lambda_sdpo,
                linear_annealing_map=request.linear_annealing_map,
                weight_decay_mode=request.weight_decay_mode,
                wd_W=request.wd_W,
                wd_b=request.wd_b,
                lr_scheduler_type=request.lr_scheduler_type,
                warmup_ratio=request.warmup_ratio,
                eval_epochs=request.eval_epochs,
                stop_threshold=request.stop_threshold,
                stop_threshold_min=request.stop_threshold_min,
                stop_threshold_frac=request.stop_threshold_frac,
                eval_selection_metric=request.eval_selection_metric,
                learn_W=request.learn_W,
                learn_R=request.learn_R,
                learn_bias_network=request.learn_bias_network,
                include_previous_targets=request.include_previous_targets,
                force=request.force,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/projection")
    def set_projection(request: ProjectionRequest):
        try:
            return runtime.set_projection_sets(request.sets)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, IndexError) as exc:
            raise _bad_request(exc) from exc

    @app.get("/api/recovery/jobs/{job_id}")
    def recovery_job(job_id: str):
        try:
            return runtime.recovery_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/recovery/jobs/{job_id}/pause")
    def pause_recovery_training(job_id: str):
        try:
            return runtime.control_recovery_training(job_id, "pause")
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/recovery/jobs/{job_id}/resume")
    def resume_recovery_training(job_id: str):
        try:
            return runtime.control_recovery_training(job_id, "resume")
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/recovery/jobs/{job_id}/stop")
    def stop_recovery_training(job_id: str):
        try:
            return runtime.control_recovery_training(job_id, "stop")
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        try:
            return runtime.chat(
                request.session_id,
                text=request.text,
                multi_turn=request.multi_turn,
                use_checkpoint_prompt=request.use_checkpoint_prompt,
                enable_thinking=request.enable_thinking,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                compute_similarity=request.compute_similarity,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/clear")
    def clear(request: SessionRequest):
        try:
            return runtime.clear_history(request.session_id)
        except ValueError as exc:
            raise _bad_request(exc) from exc

    return app
