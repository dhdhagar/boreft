"""SDPO test-time training (TTT).

SDPO reference: Hübotter et al., "Reinforcement Learning via Self-Distillation"
https://arxiv.org/abs/2601.20802

Search proposals use only ``--task-description`` (no OPRO in-context history).
Each scored batch — warm starts first, then every later ``observe`` — becomes
teacher feedback in OPRO's ``solution:`` / ``score:`` layout. ``last_k_incontext``
(default 0) optionally keeps *k* unique observations (current batch and
earlier) in that dump; ``last_k_strategy`` is ``recent`` (newest unique) or
``best`` (highest-scoring unique). 0 leaves the unique latest batch only. Each ``update_steps``
iteration draws a fresh on-policy student batch and distills it with reverse
KL into a task-local LoRA adapter, so the next proposal can stay task-only.

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline sdpo_ttt \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/sdpo-ttt-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from boreft.chem import maybe_repair_invalid_smiles, wrap_mist_smiles_tags, wrap_smiles_tags
from boreft.data_utils import (
    IGNORE_INDEX,
    decode_generated_text,
    system_prompt_from_cfg,
    tokenize_model_text,
)
from boreft.interactive import build_checkpoint_prompt, maybe_append_decode_smiles_open_tag
from boreft.pyreft.losses import sdpo_distillation_loss
from boreft.search import normalize_search_text

from .base import (
    BaselineObservation,
    Candidate,
    GenerationOptions,
    SearchContext,
    solution_key,
)
from .opro import _build_prompt

TeacherPolicy = Literal["current", "ema"]
SDPODivergence = Literal["forward_kl", "reverse_kl", "js"]
LastKStrategy = Literal["recent", "best"]
_BLANK_SAMPLE_RETRIES = 3
_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "query_key_value",
    "dense",
    "wc1",
    "wc2",
)


class LoRAInitConfig(Protocol):
    """Shared LoRA knobs used by SDPO-TTT and MiGrATe."""

    lora_rank: int
    learning_rate: float

    @property
    def resolved_lora_alpha(self) -> float: ...


class StudentRuntime(Protocol):
    """Generate from the student and distill a scored feedback batch into it."""

    def generate(
        self, prompt: str, count: int, options: GenerationOptions
    ) -> Sequence[str]: ...

    def distill(
        self,
        student_prompt: str,
        teacher_prompt: str,
        options: GenerationOptions,
    ) -> float | None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: Mapping) -> None: ...


@dataclass(frozen=True)
class SDPOTTTConfig:
    teacher_policy: TeacherPolicy = "ema"
    distillation_alpha: float = 1.0
    learning_rate: float = 1e-6
    update_steps: int = 1
    lora_rank: int = 16
    lora_alpha: float = 0.0
    n_onpolicy: int = 16
    # 0.99 decay ≡ SDPO TTT teacher_update_rate=0.01.
    ema_decay: float = 0.99
    divergence: SDPODivergence = "reverse_kl"
    temperature: float = 1.0
    distillation_topk: int = 20
    distillation_add_tail: bool = True
    last_k_incontext: int = 0
    last_k_strategy: LastKStrategy = "recent"

    def __post_init__(self) -> None:
        if self.teacher_policy not in ("current", "ema"):
            raise ValueError(
                f"unknown SDPO-TTT teacher policy: {self.teacher_policy!r}"
            )
        if self.divergence not in ("forward_kl", "reverse_kl", "js"):
            raise ValueError(f"unknown SDPO divergence: {self.divergence!r}")
        if self.distillation_alpha < 0:
            raise ValueError("distillation_alpha must be nonnegative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.update_steps < 1:
            raise ValueError("update_steps must be positive")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha < 0:
            raise ValueError("lora_alpha must be nonnegative")
        if self.n_onpolicy < 1:
            raise ValueError("n_onpolicy must be positive")
        if not 0 < self.ema_decay <= 1:
            raise ValueError("ema_decay must be in (0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.distillation_topk < 0:
            raise ValueError("distillation_topk must be nonnegative")
        if self.last_k_incontext < 0:
            raise ValueError("last_k_incontext must be nonnegative")
        if self.last_k_strategy not in ("recent", "best"):
            raise ValueError(
                f"unknown last-k strategy: {self.last_k_strategy!r}"
            )

    @property
    def resolved_lora_alpha(self) -> float:
        return float(self.lora_rank if self.lora_alpha == 0 else self.lora_alpha)


def _merge_observations(
    *groups: Sequence[BaselineObservation],
) -> list[BaselineObservation]:
    by_index: dict[int, BaselineObservation] = {}
    for group in groups:
        for item in group:
            by_index[item.index] = item
    return [by_index[index] for index in sorted(by_index)]


def _unique_recent_observations(
    items: Sequence[BaselineObservation],
    *,
    limit: int | None = None,
) -> list[BaselineObservation]:
    """Keep the most recent unique solutions, walking newest-first.

    Identity is :func:`solution_key`. ``limit`` caps how many unique items to
    keep (``None`` uniques the whole sequence). The result is still sorted
    low-to-high score for the teacher dump.
    """
    selected: list[BaselineObservation] = []
    seen: set[str] = set()
    for item in reversed(items):
        key = solution_key(item.solution)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    selected.sort(key=lambda item: (item.score, item.index))
    return selected


def _unique_best_observations(
    items: Sequence[BaselineObservation],
    *,
    limit: int | None = None,
) -> list[BaselineObservation]:
    """Keep the highest-scoring unique solutions, then the top ``limit``.

    Ties break toward the later ``index``. Identity is :func:`solution_key`.
    The result is sorted low-to-high score for the teacher dump.
    """
    by_key: dict[str, BaselineObservation] = {}
    for item in items:
        key = solution_key(item.solution)
        if not key:
            continue
        previous = by_key.get(key)
        if previous is None or (item.score, item.index) > (
            previous.score,
            previous.index,
        ):
            by_key[key] = item
    selected = sorted(
        by_key.values(), key=lambda item: (item.score, item.index), reverse=True
    )
    if limit is not None:
        selected = selected[:limit]
    selected.sort(key=lambda item: (item.score, item.index))
    return selected


def select_teacher_observations(
    batch: Sequence[BaselineObservation],
    *,
    history: Sequence[BaselineObservation] = (),
    last_k_incontext: int = 0,
    last_k_strategy: LastKStrategy = "recent",
) -> list[BaselineObservation]:
    """Latest batch, or *k* unique scored items when ``last_k_incontext`` > 0."""
    if not batch:
        raise ValueError("teacher prompt requires a nonempty observation batch")
    if last_k_incontext < 0:
        raise ValueError("last_k_incontext must be nonnegative")
    if last_k_strategy not in ("recent", "best"):
        raise ValueError(f"unknown last-k strategy: {last_k_strategy!r}")
    unique = (
        _unique_best_observations
        if last_k_strategy == "best"
        else _unique_recent_observations
    )
    if last_k_incontext == 0:
        return unique(batch)
    return unique(_merge_observations(history, batch), limit=last_k_incontext)


def build_teacher_prompt(
    task_description: str,
    batch: Sequence[BaselineObservation],
    *,
    history: Sequence[BaselineObservation] = (),
    last_k_incontext: int = 0,
    last_k_strategy: LastKStrategy = "recent",
    task: str | None = None,
) -> str:
    """OPRO-style scored pairs for teacher feedback.

    ``last_k_incontext=0`` dumps the unique latest batch. Otherwise *k* unique
    observations from ``history`` union ``batch`` are used: newest-first
    (``recent``) or highest-scoring (``best``).
    """
    selected = select_teacher_observations(
        batch,
        history=history,
        last_k_incontext=last_k_incontext,
        last_k_strategy=last_k_strategy,
    )
    if last_k_incontext == 0:
        heading = (
            "Feedback from the latest evaluated batch, "
            "ordered from lowest score to highest:"
        )
    elif last_k_strategy == "best":
        heading = (
            f"Feedback from the best {len(selected)} observation(s), "
            "ordered from lowest score to highest:"
        )
    else:
        heading = (
            f"Feedback from the last {len(selected)} observation(s), "
            "ordered from lowest score to highest:"
        )
    return _build_prompt(
        task_description, selected, history_heading=heading, task=task
    )


def _normalize_generated_text(text: str, task: str | None) -> str:
    if task == "molopt":
        smiles, _repaired = maybe_repair_invalid_smiles(text)
        return smiles
    if task != "semantle":
        return str(text)
    return "\n".join(
        normalize_search_text(line, task=task)
        for line in str(text).splitlines()
        if str(line).strip()
    )


class LoRALinear(nn.Module):
    """Frozen base linear plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, **factory)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, **factory)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("ema_A", self.lora_A.detach().clone())
        self.register_buffer("ema_B", self.lora_B.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scale

    def as_identity(self) -> None:
        """Zero the residual so forward matches the frozen base linear."""
        with torch.no_grad():
            self.lora_A.zero_()
            self.lora_B.zero_()
            self.ema_A.zero_()
            self.ema_B.zero_()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.ema_A.copy_(self.lora_A.detach())
        self.ema_B.copy_(self.lora_B.detach())

    def update_ema(self, decay: float) -> None:
        self.ema_A.mul_(decay).add_(self.lora_A.detach(), alpha=1.0 - decay)
        self.ema_B.mul_(decay).add_(self.lora_B.detach(), alpha=1.0 - decay)

    def apply_ema_weights(self) -> None:
        self.lora_A.data.copy_(self.ema_A)
        self.lora_B.data.copy_(self.ema_B)

    def restore_weights(self, adapter_a: torch.Tensor, adapter_b: torch.Tensor) -> None:
        self.lora_A.data.copy_(adapter_a)
        self.lora_B.data.copy_(adapter_b)


def _module_device(module: nn.Module) -> torch.device:
    device = getattr(module, "device", None)
    if isinstance(device, torch.device):
        return device
    return next(module.parameters()).device


def _iter_lora(module: nn.Module) -> list[LoRALinear]:
    return [child for child in module.modules() if isinstance(child, LoRALinear)]


def clear_checkpoint_lora(checkpoint: object | None) -> None:
    """Zero leftover task-local LoRA so a later seed's warm starts see the base model.

    Adapters stay attached on the shared checkpoint backbone. Zeroing A/B/EMA
    makes the residual a no-op without drawing kaiming noise that would shift
    warm-start sampling. The next seed still reinitializes (or reloads) adapter
    weights before search.
    """
    reft = getattr(checkpoint, "reft_model", None)
    seen: set[int] = set()
    for root in (reft, getattr(reft, "model", None)):
        if not isinstance(root, nn.Module) or id(root) in seen:
            continue
        seen.add(id(root))
        for adapter in _iter_lora(root):
            adapter.as_identity()


def _inject_lora(model: nn.Module, rank: int, alpha: float) -> list[LoRALinear]:
    existing = _iter_lora(model)
    if existing:
        if any(adapter.rank != rank for adapter in existing):
            raise ValueError(
                "task-local LoRA found an existing adapter with a different rank; "
                "refusing to reuse it"
            )
        return existing
    injected: list[LoRALinear] = []
    for name, child in list(model.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        stem = name.rsplit(".", 1)[-1]
        if stem not in _LORA_TARGETS:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)
        wrapped = LoRALinear(child, rank, alpha)
        setattr(parent, attr, wrapped)
        injected.append(wrapped)
    if not injected:
        raise ValueError(
            "could not attach LoRA: no matching linear modules "
            f"(looked for {', '.join(_LORA_TARGETS)})"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for adapter in injected:
        adapter.lora_A.requires_grad = True
        adapter.lora_B.requires_grad = True
    return injected


def _jsonable_tensor(value: torch.Tensor) -> list:
    return value.detach().cpu().float().tolist()


def _tensor_from_json(value: object, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device=like.device).to(
        dtype=like.dtype
    )


class _ContextRuntime:
    """Fake-backend path: sample via ``context.generate`` and skip distillation."""

    def __init__(self, context: SearchContext) -> None:
        self.context = context

    def generate(
        self, prompt: str, count: int, options: GenerationOptions
    ) -> list[str]:
        return [
            str(text).strip()
            for text in self.context.generate(prompt, count, options)
            if str(text).strip()
        ]

    def distill(
        self,
        student_prompt: str,
        teacher_prompt: str,
        options: GenerationOptions,
    ) -> float | None:
        del student_prompt, teacher_prompt, options
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: Mapping) -> None:
        del state


class LoRAStudentRuntime:
    """Task-local LoRA on the checkpoint backbone, used for sample and distill.

    Adapters are injected in place on ``checkpoint.reft_model.model``. A later
    seed clears those modules before warm starts, then reuses them and resets
    (or reloads) their weights.
    """

    def __init__(self, context: SearchContext, config: LoRAInitConfig) -> None:
        checkpoint = context.checkpoint
        if checkpoint is None or getattr(checkpoint, "reft_model", None) is None:
            raise ValueError("task-local LoRA requires a loaded checkpoint")
        model_name = context.model_name
        if not model_name:
            raise ValueError("task-local LoRA requires context.model_name")
        self.context = context
        self.config = config
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.model = checkpoint.reft_model.model
        self.tokenizer = checkpoint.tokenizer
        self.adapters = _inject_lora(
            self.model, config.lora_rank, config.resolved_lora_alpha
        )
        self.device = _module_device(self.model)
        self.optimizer = torch.optim.AdamW(
            [adapter.lora_A for adapter in self.adapters]
            + [adapter.lora_B for adapter in self.adapters],
            lr=config.learning_rate,
        )

    def generate(
        self, prompt: str, count: int, options: GenerationOptions
    ) -> list[str]:
        prompt = self._candidate_prompt(prompt)
        texts = []
        for _ in range(count):
            text, _ids = self._sample_one(prompt, options)
            if text:
                texts.append(text)
        return texts

    def _sample_rollouts(
        self, student_prompt: str, options: GenerationOptions
    ) -> list[list[int]]:
        student_prompt = self._candidate_prompt(student_prompt)
        rollouts: list[list[int]] = []
        for _ in range(_BLANK_SAMPLE_RETRIES):
            rollouts = []
            for _ in range(self.config.n_onpolicy):
                _text, token_ids = self._sample_one(student_prompt, options)
                if token_ids:
                    rollouts.append(token_ids)
            if rollouts:
                return rollouts
        raise ValueError(
            "sdpo_ttt received only blank on-policy samples from the student"
        )

    def distill(
        self,
        student_prompt: str,
        teacher_prompt: str,
        options: GenerationOptions,
    ) -> float | None:
        student_prefix, from_chat = self._render(self._candidate_prompt(student_prompt))
        teacher_prefix, _ = self._render(self._candidate_prompt(teacher_prompt))
        student_ids = self._encode(student_prefix, from_chat)
        teacher_ids = self._encode(teacher_prefix, from_chat)
        last_loss = None
        was_training = self.model.training
        # Eval keeps backbone dropout off so teacher and student logits match.
        # LoRA still gets gradients via ``enable_grad`` in ``_labeled_logits``.
        self.model.eval()
        try:
            for _ in range(self.config.update_steps):
                rollouts = self._sample_rollouts(student_prompt, options)
                teacher_logits = self._labeled_logits(
                    teacher_ids,
                    rollouts,
                    use_ema=self.config.teacher_policy == "ema",
                    requires_grad=False,
                )
                student_logits = self._labeled_logits(
                    student_ids, rollouts, use_ema=False, requires_grad=True
                )
                if student_logits.shape != teacher_logits.shape:
                    raise ValueError(
                        "sdpo_ttt student/teacher logit shapes do not align: "
                        f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
                    )
                if student_logits.numel() == 0:
                    raise ValueError("sdpo_ttt distillation produced no labeled tokens")
                loss = self.config.distillation_alpha * sdpo_distillation_loss(
                    student_logits,
                    teacher_logits,
                    temperature=self.config.temperature,
                    divergence=self.config.divergence,
                    topk=self.config.distillation_topk or None,
                    add_tail=self.config.distillation_add_tail,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.config.teacher_policy == "ema":
                    for adapter in self.adapters:
                        adapter.update_ema(self.config.ema_decay)
                last_loss = float(loss.detach())
        finally:
            self.model.train(was_training)
        return last_loss

    def state_dict(self) -> dict:
        adapters = []
        for adapter in self.adapters:
            adapters.append(
                {
                    "lora_A": _jsonable_tensor(adapter.lora_A),
                    "lora_B": _jsonable_tensor(adapter.lora_B),
                    "ema_A": _jsonable_tensor(adapter.ema_A),
                    "ema_B": _jsonable_tensor(adapter.ema_B),
                }
            )
        return {"adapters": adapters}

    def load_state_dict(self, state: Mapping) -> None:
        raw = state.get("adapters")
        if not raw:
            for adapter in self.adapters:
                adapter.reset_parameters()
            return
        if not isinstance(raw, list) or len(raw) != len(self.adapters):
            raise ValueError("LoRA adapter state does not match the attached modules")
        for adapter, payload in zip(self.adapters, raw):
            if not isinstance(payload, dict) or "lora_A" not in payload or "lora_B" not in payload:
                raise ValueError("LoRA adapter state is missing weights")
            adapter.lora_A.data.copy_(
                _tensor_from_json(payload["lora_A"], adapter.lora_A)
            )
            adapter.lora_B.data.copy_(
                _tensor_from_json(payload["lora_B"], adapter.lora_B)
            )
            adapter.ema_A.copy_(
                _tensor_from_json(
                    payload.get("ema_A", payload["lora_A"]), adapter.ema_A
                )
            )
            adapter.ema_B.copy_(
                _tensor_from_json(
                    payload.get("ema_B", payload["lora_B"]), adapter.ema_B
                )
            )

    def reset_parameters(self) -> None:
        for adapter in self.adapters:
            adapter.reset_parameters()

    def _candidate_prompt(self, prompt: str) -> str:
        """Keep MiST ``[START_SMILES]`` as the last completion token."""
        return maybe_append_decode_smiles_open_tag(self.checkpoint, prompt)

    def _completion_text(self, solution: str) -> str:
        """Tokenize LoRA completions in the same tagged space as training gold."""
        cfg = getattr(self.checkpoint, "saved_cfg", None) or {}
        if bool(cfg.get("mist_smiles_tags")):
            return wrap_mist_smiles_tags(
                solution,
                include_open=bool(cfg.get("use_chat_template")),
            )
        if bool(cfg.get("smiles_tags")):
            return wrap_smiles_tags(solution)
        return str(solution)

    def _render(self, user_text: str) -> tuple[str, bool]:
        rendered, from_chat, _ = build_checkpoint_prompt(
            self.checkpoint,
            user_text=user_text,
            history=[],
            accumulate_history=False,
            use_checkpoint_prompt=False,
            generation_mode="base",
            model_name=self.model_name,
            system_prompt=system_prompt_from_cfg(self.checkpoint.saved_cfg),
        )
        return rendered, from_chat

    def _encode(self, text: str, from_chat: bool) -> list[int]:
        encoded = tokenize_model_text(
            self.tokenizer, text, from_chat_template=from_chat, return_tensors="pt"
        )
        return encoded["input_ids"][0].tolist()

    def _sample_one(
        self,
        prompt: str,
        options: GenerationOptions,
    ) -> tuple[str, list[int]]:
        rendered, from_chat = self._render(prompt)
        encoded = tokenize_model_text(
            self.tokenizer,
            rendered,
            from_chat_template=from_chat,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attn_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]
        gen_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if options.temperature > 0:
            gen_kwargs["temperature"] = max(options.temperature, 1e-8)
            gen_kwargs["top_p"] = options.top_p
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
        finally:
            self.model.train(was_training)
        continuation = _trim_generated(
            output_ids[0, prompt_len:].tolist(),
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        text = decode_generated_text(
            self.tokenizer,
            output_ids,
            prompt_len,
            assistant_suffix=self.checkpoint.assistant_suffix,
        )
        text = _normalize_generated_text(text, self.context.task)
        return text, continuation

    def _labeled_logits(
        self,
        prefix_ids: Sequence[int],
        rollouts: Sequence[Sequence[int]],
        *,
        use_ema: bool,
        requires_grad: bool,
    ) -> torch.Tensor:
        saved = None
        if use_ema:
            saved = [
                (adapter.lora_A.data.clone(), adapter.lora_B.data.clone())
                for adapter in self.adapters
            ]
            for adapter in self.adapters:
                adapter.apply_ema_weights()
        try:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            rows = [list(prefix_ids) + list(y) for y in rollouts]
            labels = [[IGNORE_INDEX] * len(prefix_ids) + list(y) for y in rollouts]
            max_len = max(len(row) for row in rows)
            input_ids = torch.full(
                (len(rows), max_len), pad_id, dtype=torch.long, device=self.device
            )
            label_ids = torch.full(
                (len(rows), max_len),
                IGNORE_INDEX,
                dtype=torch.long,
                device=self.device,
            )
            attn = torch.zeros(
                (len(rows), max_len), dtype=torch.long, device=self.device
            )
            for index, (row, label) in enumerate(zip(rows, labels)):
                input_ids[index, : len(row)] = torch.tensor(
                    row, dtype=torch.long, device=self.device
                )
                label_ids[index, : len(label)] = torch.tensor(
                    label, dtype=torch.long, device=self.device
                )
                attn[index, : len(row)] = 1
            ctx = torch.enable_grad() if requires_grad else torch.no_grad()
            with ctx:
                outputs = self.model(input_ids=input_ids, attention_mask=attn)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                logits = outputs[0]
            gathered = []
            for row in range(label_ids.shape[0]):
                idx = (label_ids[row] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                idx = idx[idx > 0]
                if idx.numel() == 0:
                    continue
                gathered.append(logits[row, idx - 1, :])
            if not gathered:
                return logits.new_zeros((0, logits.shape[-1]))
            return torch.cat(gathered, dim=0)
        finally:
            if saved is not None:
                for adapter, (adapter_a, adapter_b) in zip(self.adapters, saved):
                    adapter.restore_weights(adapter_a, adapter_b)


def _trim_generated(
    token_ids: Sequence[int], eos_id: int | None, pad_id: int | None
) -> list[int]:
    out: list[int] = []
    for token in token_ids:
        token = int(token)
        if pad_id is not None and token == pad_id and token != eos_id:
            break
        out.append(token)
        if eos_id is not None and token == eos_id:
            break
    return out


class SDPOTTTBaseline:
    name = "sdpo_ttt"

    def __init__(
        self,
        config: SDPOTTTConfig | None = None,
        *,
        runtime: StudentRuntime | None = None,
    ) -> None:
        self.config = config or SDPOTTTConfig()
        self._injected_runtime = runtime
        self._runtime: StudentRuntime | None = runtime
        self._consumed_through = -1
        self._history: list[BaselineObservation] = []
        self._pending_state: dict | None = None
        self._context: SearchContext | None = None

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        if count < 1:
            raise ValueError("candidate count must be positive")
        self._bind(context)
        assert self._runtime is not None
        self._history = list(history)
        pending = [item for item in history if item.index > self._consumed_through]
        if pending:
            self._distill_batch(context, pending)
        prompt = context.task_description.strip()
        sampled: list[str] = []
        for _ in range(_BLANK_SAMPLE_RETRIES):
            sampled = [
                item
                for item in self._runtime.generate(
                    prompt, count, context.generation_options
                )
                if item.strip()
            ]
            if sampled:
                break
        if not sampled:
            raise ValueError(
                f"{self.name} received only blank samples from the student"
            )
        seen_history = {solution_key(item.solution) for item in history}
        seen_batch: set[str] = set()
        candidates: list[Candidate] = []
        for solution in sampled[:count]:
            key = solution_key(solution)
            candidates.append(
                Candidate(
                    solution=solution,
                    metadata={
                        "prompt": prompt,
                        "is_repeat_proposal": key in seen_history or key in seen_batch,
                    },
                )
            )
            seen_batch.add(key)
        return candidates

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        if not observations:
            return
        if self._context is None or self._runtime is None:
            raise RuntimeError("sdpo_ttt.observe() requires a prior propose() call")
        self._history = _merge_observations(self._history, observations)
        self._distill_batch(self._context, observations)

    def state_dict(self) -> dict:
        payload = {"consumed_through": self._consumed_through}
        if self._runtime is not None:
            payload.update(self._runtime.state_dict())
        elif self._pending_state:
            payload.update(
                {
                    key: value
                    for key, value in self._pending_state.items()
                    if key != "consumed_through"
                }
            )
        return payload

    def load_state_dict(self, state: Mapping) -> None:
        if not state:
            return
        self._consumed_through = int(state.get("consumed_through", -1))
        self._pending_state = dict(state)
        if self._runtime is not None:
            self._runtime.load_state_dict(state)
            self._pending_state = None

    def _bind(self, context: SearchContext) -> None:
        self._context = context
        if self._injected_runtime is not None:
            self._runtime = self._injected_runtime
        elif getattr(context.checkpoint, "reft_model", None) is not None:
            if not isinstance(self._runtime, LoRAStudentRuntime):
                self._runtime = LoRAStudentRuntime(context, self.config)
                if self._pending_state is None:
                    self._runtime.reset_parameters()
        else:
            self._runtime = _ContextRuntime(context)
        if self._pending_state is not None:
            self._runtime.load_state_dict(self._pending_state)
            self._pending_state = None

    def _distill_batch(
        self, context: SearchContext, batch: Sequence[BaselineObservation]
    ) -> None:
        assert self._runtime is not None
        teacher_prompt = build_teacher_prompt(
            context.task_description,
            batch,
            history=self._history,
            last_k_incontext=self.config.last_k_incontext,
            last_k_strategy=self.config.last_k_strategy,
            task=context.task,
        )
        student_prompt = context.task_description.strip()
        loss = self._runtime.distill(
            student_prompt, teacher_prompt, context.generation_options
        )
        self._consumed_through = max(
            self._consumed_through, max(item.index for item in batch)
        )
        if loss is not None:
            print(
                f"[sdpo_ttt] distilled {len(batch)} observation(s)  loss={loss:.4f}",
                flush=True,
            )
