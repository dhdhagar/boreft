"""During-training reconstruction eval: generate each target at b=mu and score it.

Task-neutral: the similarity metric comes from the task's embedding model (see
:mod:`boreft.text_similarity`), and exact-match / validity / fingerprint
similarity come from the task's target kind, so the same callback serves Semantle
words and molopt SMILES.
"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import wandb
from transformers import TrainerCallback

from boreft.eval.eval_suite import (
    DEFAULT_EMBED_SIM_TAU,
    DEFAULT_RDKIT_SIM_TAU,
    DEFAULT_TFS_TAU,
)
from boreft.task_config import (
    task_instruction,
    task_supports_fingerprints,
    task_supports_validity,
)
from boreft.text_display import (
    EVAL_LOG_GENERATED_W,
    EVAL_LOG_SIM_W,
    EVAL_LOG_TARGET_W,
    EVAL_LOG_WIDTH_MAX,
    ellipsis_middle_fit,
    print_log_table,
)
from boreft.text_similarity import embedding_sim_per_text, rdkit_map_path_for_cfg

from .base import item_target, sample_eval_subset

# NOTE: ``generate_text`` is imported lazily inside ``_run_periodic_eval`` to avoid a
# module-load import cycle (data_utils → data → data.eval_callback → eval.semantle →
# data_utils). Keeping it lazy lets eval/run_full_eval.py and eval/semantle.py be
# imported as standalone entry points.


#: Metrics that may drive early stopping and threshold checkpoints. ``embed_sim``
#: works for every recon-eval task; the other two need molecular targets.
EVAL_SELECTION_METRICS = ("embed_sim", "rdkit_sim", "tfs")


def embed_sim_frac_at_or_above(sims, tau: float) -> float:
    """Fraction of per-target sims that are ``>= tau``."""
    arr = np.asarray(sims, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= float(tau)))


def embed_sim_stop_triggered(
    mean_sim: float,
    min_sim: float,
    *,
    stop_threshold: Optional[float] = None,
    stop_threshold_min: Optional[float] = None,
    stop_threshold_frac: float = 1.0,
    sims: Optional[Sequence[float]] = None,
) -> bool:
    """Return True when all configured early-stop criteria are satisfied.

    The sims are whichever metric the callback selects (embed_sim by default);
    the names below say embed_sim because that is the usual case.

    ``stop_threshold`` compares against the mean per-target sim.

    ``stop_threshold_min`` is a per-target bar (tau). With ``stop_threshold_frac``
    (default ``1.0``), stop when the fraction of eval targets with
    ``embed_sim >= tau`` is at least that value. ``frac=1`` is the legacy
    "every target / min >= tau" behavior. When ``sims`` is provided it is used
    for the frac check; otherwise ``min_sim >= tau`` is used (valid only for
    ``frac == 1``).

    When both mean and min/frac criteria are set, both must pass.

    Mean embed_sim of 1.0 always stops training (perfect reconstruction), even when
    no explicit thresholds are configured.
    """
    if mean_sim >= 1.0 - 1e-12:
        return True
    if stop_threshold is None and stop_threshold_min is None:
        return False
    mean_ok = stop_threshold is None or mean_sim >= stop_threshold
    if stop_threshold_min is None:
        min_ok = True
    elif sims is not None:
        frac = embed_sim_frac_at_or_above(sims, stop_threshold_min)
        min_ok = frac >= float(stop_threshold_frac)
    else:
        # Without per-target sims, only the all-pass (frac=1) case is expressible.
        if float(stop_threshold_frac) < 1.0 - 1e-12:
            raise ValueError(
                "sims is required when stop_threshold_frac < 1 "
                "(cannot evaluate a partial pass rate from min_sim alone)"
            )
        min_ok = min_sim >= stop_threshold_min
    return mean_ok and min_ok


def embed_sim_checkpoint_thresholds_to_save(
    metric: float,
    thresholds_sorted: List[float],
    done: set,
) -> List[float]:
    """Return embed_sim thresholds newly crossed by ``metric`` (not already in ``done``)."""
    return [t for t in thresholds_sorted if t not in done and metric >= t]


# TODO: structure this code better, move to another file for train time evals
class TargetReconEval(TrainerCallback):
    """Periodic greedy eval: target vs generated, scored by every available metric.

    Trigger via ``eval_steps`` (every N optimizer steps) or ``eval_every_epochs``
    (every N epochs). Exactly one must be set.

    selection_metric: which per-target metric drives early stopping and threshold
        checkpoints — ``embed_sim`` (default), ``rdkit_sim`` or ``tfs``. The two
        molecular metrics require a task with molecular targets. Reporting is
        unaffected: all available metrics are printed and logged either way, and
        the selected one is mirrored to ``eval/selection_sim``.
    stop_threshold: stop when the mean selection metric >= this value. None =
        disabled, except a mean of 1.0 always stops.
    stop_threshold_min: per-target selection-metric bar (tau). Combined with
        stop_threshold_frac (default 1.0): stop when at least that fraction of
        eval targets score >= tau. frac=1 is "every target / min >= tau".
        When both mean and min/frac criteria are set, both must pass.
    stop_threshold_frac: minimum fraction of eval targets that must clear
        stop_threshold_min (in (0, 1]; default 1.0).
    embed_sim_tau: RECON threshold used for wandb ``eval/embed_sim_gte_tau``
        (independent of early-stop bars).

    checkpoint_embed_sim_thresholds: optional list (e.g. 0.6, 0.7, 0.8, 0.9, 1.0). The first time
    the mean selection metric reaches each threshold, saves a full eval-ready dir under
    output_dir/checkpoints_by_embed_sim/embed_sim_<tag>/ (intervenable_model, items.json,
    intervention_config.json, tokenizer, checkpoint_info.json). The paths keep the
    embed_sim name for compatibility; checkpoint_info.json records which metric
    actually triggered the save.

    checkpoint_embed_sim_thresholds_min: same for the min over the eval subset, saved under
    output_dir/checkpoints_by_embed_sim_min/embed_sim_min_<tag>/.

    For tasks whose targets have a validity notion (molopt SMILES), the fraction of
    parseable generations is also logged as ``eval/validity``, mean Morgan Tanimoto
    similarity as ``eval/tfs`` / ``eval/tfs_gte_tau``, descriptor similarity as
    ``eval/rdkit_sim`` / ``eval/rdkit_sim_gte_tau``, and canonical-SMILES
    Levenshtein distance as ``eval/edit_dist`` (lower is better).
    """

    def __init__(
        self,
        intervenable,
        tokenizer,
        items,
        eval_steps: Optional[int] = None,
        eval_every_epochs: Optional[int] = None,
        eval_n_samples: Optional[int] = None,
        eval_sample_seed: int = 42,
        stop_threshold: Optional[float] = None,
        stop_threshold_min: Optional[float] = None,
        stop_threshold_frac: float = 1.0,
        embed_sim_tau: float = DEFAULT_EMBED_SIM_TAU,
        selection_metric: str = "embed_sim",
        save_best_params: bool = True,
        use_stochastic_intervention: bool = False,
        output_dir: Optional[str] = None,
        intervention_config: Optional[Dict] = None,
        checkpoint_embed_sim_thresholds: Optional[List[float]] = None,
        checkpoint_embed_sim_thresholds_min: Optional[List[float]] = None,
        position: str = "l1",
        assistant_suffix: Optional[str] = None,
        from_chat_template: bool = False,
        eval_prompt: Optional[str] = None,
        intervention_token_id: Optional[int] = None,
        content_span: Optional[tuple[int, int]] = None,
        task: str = "semantle",
    ):
        if (eval_steps is None) == (eval_every_epochs is None):
            raise ValueError(
                "exactly one of eval_steps or eval_every_epochs must be set"
            )
        self.intervenable = intervenable
        self.tokenizer = tokenizer
        self.items = list(items)
        self.task = str(task or (intervention_config or {}).get("task", "semantle"))
        self.targets = [item_target(it) for it in items]
        self.prompt = (
            eval_prompt
            if eval_prompt is not None
            else (
                items[0].prompt
                if items
                else task_instruction(self.task, use_chat_template=False)
            )
        )
        self.intervention_token_id = intervention_token_id
        self.content_span = content_span
        self.assistant_suffix = assistant_suffix
        self.from_chat_template = from_chat_template
        self.eval_steps = eval_steps
        self.eval_every_epochs = eval_every_epochs
        self._eval_items = sample_eval_subset(
            self.items, eval_n_samples, eval_sample_seed
        )
        self.stop_threshold = stop_threshold
        self.stop_threshold_min = stop_threshold_min
        self.stop_threshold_frac = float(stop_threshold_frac)
        if not (0.0 < self.stop_threshold_frac <= 1.0):
            raise ValueError("stop_threshold_frac must be in (0, 1]")
        if (
            self.stop_threshold_frac < 1.0 - 1e-12
            and self.stop_threshold_min is None
        ):
            raise ValueError(
                "stop_threshold_frac < 1 requires stop_threshold_min"
            )
        self.embed_sim_tau = float(embed_sim_tau)
        if not (0.0 <= self.embed_sim_tau <= 1.0):
            raise ValueError("embed_sim_tau must be in [0, 1]")
        self.use_stochastic_intervention = use_stochastic_intervention
        self.log_validity = task_supports_validity(self.task)
        self.log_tfs = task_supports_fingerprints(self.task)
        self.log_rdkit_sim = task_supports_fingerprints(self.task)
        self.log_edit_dist = task_supports_validity(self.task)
        self.selection_metric = str(selection_metric)
        if self.selection_metric not in EVAL_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {EVAL_SELECTION_METRICS}, "
                f"got {selection_metric!r}"
            )
        if self.selection_metric == "tfs" and not self.log_tfs:
            raise ValueError(
                f"selection_metric='tfs' is unavailable for task={self.task!r}"
            )
        if self.selection_metric == "rdkit_sim" and not self.log_rdkit_sim:
            raise ValueError(
                f"selection_metric='rdkit_sim' is unavailable for task={self.task!r}"
            )
        # Reporting tau for the selected metric: the fraction of eval targets at or
        # above it is printed and logged next to its mean.
        self.selection_tau = {
            "embed_sim": self.embed_sim_tau,
            "tfs": DEFAULT_TFS_TAU,
            "rdkit_sim": DEFAULT_RDKIT_SIM_TAU,
        }[self.selection_metric]
        cfg = intervention_config or {}
        self.rdkit_map_path = rdkit_map_path_for_cfg(cfg)

        # Widen the log columns for long targets (SMILES) without letting one
        # outlier blow up the table.
        longest = max((len(t) for t in self.targets), default=0)
        self._target_w = min(EVAL_LOG_WIDTH_MAX, max(EVAL_LOG_TARGET_W, longest))
        self._generated_w = min(
            EVAL_LOG_WIDTH_MAX, max(EVAL_LOG_GENERATED_W, self._target_w + 2)
        )

        self.output_dir = output_dir
        self.intervention_config = cfg
        th = (
            [float(x) for x in checkpoint_embed_sim_thresholds]
            if checkpoint_embed_sim_thresholds
            else []
        )
        self._checkpoint_thresholds_sorted = sorted(set(th))
        self._embed_sim_checkpoint_done: set = set()
        th_min = (
            [float(x) for x in checkpoint_embed_sim_thresholds_min]
            if checkpoint_embed_sim_thresholds_min
            else []
        )
        self._checkpoint_min_thresholds_sorted = sorted(set(th_min))
        self._embed_sim_min_checkpoint_done: set = set()
        self.save_best_params = bool(save_best_params)
        self._best_selection_mean: Optional[float] = None

        self.position = position

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            eval_subset_targets = [item_target(it) for it in self._eval_items]
            manifest = {
                "task": self.task,
                "n_training_targets": len(self.targets),
                "training_targets": self.targets,
                "eval_n_samples": eval_n_samples,
                "eval_sample_seed": eval_sample_seed,
                "eval_subset_targets": eval_subset_targets,
                "n_eval_subset_targets": len(eval_subset_targets),
            }
            with open(
                os.path.join(self.output_dir, "training_target_manifest.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(manifest, f, indent=2)

    def _print_eval_table(
        self,
        rows: list[tuple[str, str]],
        *,
        sims: Optional[List[float]] = None,
    ) -> None:
        display_rows = [
            (
                ellipsis_middle_fit(target, self._target_w),
                repr(ellipsis_middle_fit(gen, self._generated_w)),
            )
            for target, gen in rows
        ]
        if sims is None:
            print_log_table(
                ("target", "generated"),
                display_rows,
                widths=(self._target_w, self._generated_w),
            )
            return

        table_rows = [
            (target, generated, f"{sim:.4f}")
            for (target, generated), sim in zip(display_rows, sims)
        ]
        print_log_table(
            ("target", "generated", "sim"),
            table_rows,
            widths=(self._target_w, self._generated_w, EVAL_LOG_SIM_W),
            align=("<", "<", ">"),
        )

    def _save_embed_sim_checkpoint(
        self, threshold: float, mean_sim: float, global_step: int
    ) -> None:
        """Write eval-ready tree: tokenizer + items + config + intervenable weights."""
        from boreft.data_utils import save_reft_checkpoint_dir

        assert self.output_dir is not None
        base = os.path.join(self.output_dir, "checkpoints_by_embed_sim")
        tag = f"embed_sim_{threshold:g}".replace(".", "p")
        sub = os.path.join(base, tag)
        save_reft_checkpoint_dir(
            sub,
            reft_model=self.intervenable,
            tokenizer=self.tokenizer,
            items=self.items,
            intervention_config=self.intervention_config,
            checkpoint_info={
                "embed_sim_threshold": threshold,
                "mean_sim_at_save": mean_sim,
                "selection_metric": self.selection_metric,
                "global_step": global_step,
            },
        )
        print(
            f"  [checkpoint] {self.selection_metric} reached >= {threshold:g} "
            f"(mean_sim={mean_sim:.4f}) -> saved {sub}"
        )

    def _save_embed_sim_min_checkpoint(
        self,
        threshold: float,
        min_sim: float,
        mean_sim: float,
        global_step: int,
    ) -> None:
        """Write eval-ready tree when the min selection metric crosses a threshold."""
        from boreft.data_utils import save_reft_checkpoint_dir

        assert self.output_dir is not None
        base = os.path.join(self.output_dir, "checkpoints_by_embed_sim_min")
        tag = f"embed_sim_min_{threshold:g}".replace(".", "p")
        sub = os.path.join(base, tag)
        save_reft_checkpoint_dir(
            sub,
            reft_model=self.intervenable,
            tokenizer=self.tokenizer,
            items=self.items,
            intervention_config=self.intervention_config,
            checkpoint_info={
                "embed_sim_min_threshold": threshold,
                "min_sim_at_save": min_sim,
                "mean_sim_at_save": mean_sim,
                "selection_metric": self.selection_metric,
                "global_step": global_step,
            },
        )
        print(
            f"  [checkpoint] {self.selection_metric}_min reached >= {threshold:g} "
            f"(min_sim={min_sim:.4f}, mean_sim={mean_sim:.4f}) -> saved {sub}"
        )

    def _maybe_save_best_checkpoint(
        self, mean_sim: float, *, global_step: int, epoch: Optional[float]
    ) -> None:
        """Overwrite ``output_dir/best/`` when selection mean improves."""
        if not self.save_best_params or self.output_dir is None:
            return
        if not math.isfinite(mean_sim):
            return
        if (
            self._best_selection_mean is not None
            and mean_sim <= self._best_selection_mean
        ):
            return
        from boreft.data_utils import BEST_CHECKPOINT_DIRNAME, save_reft_checkpoint_dir

        sub = os.path.join(self.output_dir, BEST_CHECKPOINT_DIRNAME)
        info = {
            "selection_metric": self.selection_metric,
            "mean_sim_at_save": float(mean_sim),
            "global_step": int(global_step),
            "epoch": float(epoch) if epoch is not None else None,
            "is_best": True,
        }
        save_reft_checkpoint_dir(
            sub,
            reft_model=self.intervenable,
            tokenizer=self.tokenizer,
            items=self.items,
            intervention_config=self.intervention_config,
            checkpoint_info=info,
        )
        prev = self._best_selection_mean
        self._best_selection_mean = float(mean_sim)
        if prev is None:
            print(
                f"  [best] {self.selection_metric} mean={mean_sim:.4f} "
                f"-> saved {sub}"
            )
        else:
            print(
                f"  [best] {self.selection_metric} mean={mean_sim:.4f} "
                f"(prev {prev:.4f}) -> saved {sub}"
            )

    def _run_periodic_eval(self, state, control, label: str):
        eval_items = self._eval_items
        eval_targets = [item_target(it) for it in eval_items]
        eval_ids = [it.id for it in eval_items]

        print(
            f"\n[eval] {self.task} · {label}  "
            f"({len(eval_targets)}/{len(self.targets)} targets)"
        )

        from boreft.eval.semantle import generate_text

        generated_list = [
            generate_text(
                self.intervenable,
                self.tokenizer,
                self.prompt,
                wid,
                use_stochastic_intervention=self.use_stochastic_intervention,
                position=self.position,
                assistant_suffix=self.assistant_suffix,
                from_chat_template=self.from_chat_template,
                intervention_token_id=self.intervention_token_id,
                content_span=self.content_span,
            )
            for wid in eval_ids
        ]
        embed_arr = None
        embed_mean = None
        embed_min = None
        embed_frac_at_tau = None
        embed_sim_error = None
        try:
            embed_arr = embedding_sim_per_text(
                eval_targets, generated_list, task=self.task
            )
            embed_mean = float(np.mean(embed_arr))
            embed_min = float(np.min(embed_arr))
            embed_frac_at_tau = embed_sim_frac_at_or_above(
                embed_arr, self.embed_sim_tau
            )
        except Exception as e:
            embed_arr = None
            embed_sim_error = e

        validity = None
        if self.log_validity:
            from boreft.chem import validity_rate

            validity = validity_rate(generated_list)

        tfs_arr = None
        tfs_mean = None
        tfs_frac = None
        if self.log_tfs:
            from boreft.chem import tanimoto_sim_per_text

            arr = tanimoto_sim_per_text(eval_targets, generated_list)
            if arr.size:
                tfs_arr = arr
                tfs_mean = float(np.mean(arr))
                tfs_frac = embed_sim_frac_at_or_above(arr, DEFAULT_TFS_TAU)

        rdkit_sim_arr = None
        rdkit_sim_mean = None
        rdkit_sim_frac = None
        if self.log_rdkit_sim:
            from boreft.chem import rdkit_sim_per_text

            arr = rdkit_sim_per_text(
                eval_targets,
                generated_list,
                map_path=self.rdkit_map_path,
            )
            if arr.size:
                rdkit_sim_arr = arr
                rdkit_sim_mean = float(np.mean(arr))
                rdkit_sim_frac = embed_sim_frac_at_or_above(
                    arr, DEFAULT_RDKIT_SIM_TAU
                )

        edit_dist_mean = None
        if self.log_edit_dist:
            from boreft.chem import smiles_edit_dist_per_text

            arr = smiles_edit_dist_per_text(eval_targets, generated_list)
            if arr.size:
                edit_dist_mean = float(np.mean(arr))

        # Every available metric is reported; only the selected one drives the
        # early-stop and threshold-checkpoint decisions below.
        sel_arr = {
            "embed_sim": embed_arr,
            "tfs": tfs_arr,
            "rdkit_sim": rdkit_sim_arr,
        }[self.selection_metric]
        mean_sim = float(np.mean(sel_arr)) if sel_arr is not None else None
        min_sim = float(np.min(sel_arr)) if sel_arr is not None else None

        if sel_arr is not None:
            rows = sorted(
                zip(eval_targets, generated_list, sel_arr),
                key=lambda row: row[2],
                reverse=True,
            )
            self._print_eval_table(
                [(target, gen) for target, gen, _ in rows],
                sims=[float(sim) for _, _, sim in rows],
            )
        else:
            self._print_eval_table(list(zip(eval_targets, generated_list)))

        frac_at_min = None
        if sel_arr is not None and self.stop_threshold_min is not None:
            frac_at_min = embed_sim_frac_at_or_above(
                sel_arr, self.stop_threshold_min
            )

        line_parts = []
        if sel_arr is not None:
            sel_frac_at_tau = embed_sim_frac_at_or_above(sel_arr, self.selection_tau)
            line_parts.append(
                f"{self.selection_metric}  mean={mean_sim:.4f}  min={min_sim:.4f}"
                f"  frac>={self.selection_tau:g}={sel_frac_at_tau:.4f}"
            )
            if frac_at_min is not None and (
                abs(float(self.stop_threshold_min) - self.selection_tau) > 1e-12
            ):
                line_parts.append(
                    f"frac>={self.stop_threshold_min:g}={frac_at_min:.4f}"
                )
        for name, metric_mean, tau, frac in (
            ("embed_sim", embed_mean, self.embed_sim_tau, embed_frac_at_tau),
            ("tfs", tfs_mean, DEFAULT_TFS_TAU, tfs_frac),
            ("rdkit_sim", rdkit_sim_mean, DEFAULT_RDKIT_SIM_TAU, rdkit_sim_frac),
        ):
            if name == self.selection_metric or metric_mean is None:
                continue
            line_parts.append(
                f"{name}={metric_mean:.4f}  {name}>={tau:g}={frac:.4f}"
            )
        if edit_dist_mean is not None:
            line_parts.append(f"edit_dist={edit_dist_mean:.2f}")
        if validity is not None:
            line_parts.append(f"validity={validity:.4f}")
        if line_parts:
            print("\n  " + "  ".join(line_parts))

        if wandb.run is not None:
            log_payload = {}
            if embed_mean is not None:
                log_payload["eval/embed_sim"] = embed_mean
                log_payload["eval/embed_sim_min"] = embed_min
                log_payload["eval/embed_sim_gte_tau"] = embed_frac_at_tau
                log_payload["eval/embed_sim_tau"] = self.embed_sim_tau
            if tfs_mean is not None:
                log_payload["eval/tfs"] = tfs_mean
                log_payload["eval/tfs_gte_tau"] = tfs_frac
                log_payload["eval/tfs_tau"] = DEFAULT_TFS_TAU
            if rdkit_sim_mean is not None:
                log_payload["eval/rdkit_sim"] = rdkit_sim_mean
                log_payload["eval/rdkit_sim_gte_tau"] = rdkit_sim_frac
                log_payload["eval/rdkit_sim_tau"] = DEFAULT_RDKIT_SIM_TAU
            if edit_dist_mean is not None:
                log_payload["eval/edit_dist"] = edit_dist_mean
            if validity is not None:
                log_payload["eval/validity"] = validity
            if sel_arr is not None:
                log_payload["eval/selection_sim"] = mean_sim
                log_payload["eval/selection_sim_min"] = min_sim
                if frac_at_min is not None:
                    log_payload["eval/selection_gte_stop_min"] = frac_at_min
                    if self.selection_metric == "embed_sim":
                        log_payload["eval/embed_sim_gte_stop_min"] = frac_at_min
            if log_payload:
                wandb.log(log_payload)

        if embed_sim_error is not None:
            print(f"  (embed_sim unavailable: {embed_sim_error})")
        if sel_arr is None:
            if self.selection_metric != "embed_sim":
                print(
                    f"  ({self.selection_metric} unavailable: "
                    f"no scorable eval targets)"
                )
            return

        if self._checkpoint_thresholds_sorted and self.output_dir is not None:
            for t in embed_sim_checkpoint_thresholds_to_save(
                mean_sim, self._checkpoint_thresholds_sorted, self._embed_sim_checkpoint_done
            ):
                self._save_embed_sim_checkpoint(t, mean_sim, state.global_step)
                self._embed_sim_checkpoint_done.add(t)

        if self._checkpoint_min_thresholds_sorted and self.output_dir is not None:
            for t in embed_sim_checkpoint_thresholds_to_save(
                min_sim,
                self._checkpoint_min_thresholds_sorted,
                self._embed_sim_min_checkpoint_done,
            ):
                self._save_embed_sim_min_checkpoint(
                    t, min_sim, mean_sim, state.global_step
                )
                self._embed_sim_min_checkpoint_done.add(t)

        self._maybe_save_best_checkpoint(
            mean_sim,
            global_step=int(state.global_step),
            epoch=float(state.epoch) if state.epoch is not None else None,
        )

        if embed_sim_stop_triggered(
            mean_sim,
            min_sim,
            stop_threshold=self.stop_threshold,
            stop_threshold_min=self.stop_threshold_min,
            stop_threshold_frac=self.stop_threshold_frac,
            sims=sel_arr,
        ):
            parts = []
            if mean_sim >= 1.0 - 1e-12:
                parts.append(f"mean {mean_sim:.4f} >= 1")
            if self.stop_threshold is not None:
                parts.append(f"mean {mean_sim:.4f} >= {self.stop_threshold:.4f}")
            if self.stop_threshold_min is not None:
                frac = (
                    frac_at_min
                    if frac_at_min is not None
                    else embed_sim_frac_at_or_above(
                        sel_arr, self.stop_threshold_min
                    )
                )
                if self.stop_threshold_frac >= 1.0 - 1e-12:
                    parts.append(
                        f"min {min_sim:.4f} >= {self.stop_threshold_min:.4f}"
                    )
                else:
                    parts.append(
                        f"frac>={self.stop_threshold_min:g} "
                        f"{frac:.4f} >= {self.stop_threshold_frac:.4f}"
                    )
            # Deduplicate when perfect sim also matched an explicit threshold.
            unique_parts = list(dict.fromkeys(parts))
            print(
                f"  [early stop] {self.selection_metric} criteria met "
                f"({' · '.join(unique_parts)}). Stopping."
            )
            control.should_training_stop = True
            return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps is None:
            return
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        return self._run_periodic_eval(
            state, control, label=f"step {state.global_step}"
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_every_epochs is None:
            return
        epoch = int(state.epoch)
        if epoch == 0 or epoch % self.eval_every_epochs != 0:
            return
        return self._run_periodic_eval(state, control, label=f"epoch {epoch}")
