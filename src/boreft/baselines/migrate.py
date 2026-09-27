"""MiGrATe: Mixed-policy GRPO for Adaptation at Test-Time.

Reference: Phan et al., "MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time"
https://arxiv.org/abs/2508.08641

Each step builds a GRPO group from on-policy samples (task prompt only),
top-k greedy history, and neighborhood samples around those greedy solutions.
Only the new on-policy and neighborhood completions are scored; greedy members
are reused. The mixed group is then used to update a task-local LoRA adapter
with GRPO (no KL, Dr.GRPO mean-only advantages, DAPO clip; sequence-mean of
token-means, matching the original trainer).

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline migrate \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/migrate-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from .base import (
    BaselineObservation,
    Candidate,
    GenerationOptions,
    SearchContext,
    solution_key,
)
from .llm_sample import _one_line, build_completion_fewshot
from .sdpo_ttt import LoRAStudentRuntime, _normalize_generated_text

_BLANK_SAMPLE_RETRIES = 3


class PolicyRuntime(Protocol):
    """Sample from the current policy and apply a mixed-group GRPO update."""

    def generate(
        self, prompt: str, count: int, options: GenerationOptions
    ) -> Sequence[str]: ...

    def update_grpo(
        self, task_prompt: str, group: Sequence[tuple[str, float]]
    ) -> float | None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: Mapping) -> None: ...


@dataclass(frozen=True)
class MiGrATeConfig:
    on_policy_count: int = 2
    greedy_count: int = 1
    neighborhood_count: int = 2
    greedy_topk: int = 3
    learning_rate: float = 1e-5
    update_steps: int = 1
    lora_rank: int = 16
    lora_alpha: float = 0.0
    clip_epsilon: float = 0.2
    clip_epsilon_high: float = 0.28

    def __post_init__(self) -> None:
        if self.on_policy_count < 0:
            raise ValueError("on_policy_count must be nonnegative")
        if self.greedy_count < 0:
            raise ValueError("greedy_count must be nonnegative")
        if self.neighborhood_count < 0:
            raise ValueError("neighborhood_count must be nonnegative")
        if self.on_policy_count + self.neighborhood_count < 1:
            raise ValueError("on_policy_count + neighborhood_count must be positive")
        if self.greedy_topk < 1:
            raise ValueError("greedy_topk must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.update_steps < 1:
            raise ValueError("update_steps must be positive")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha < 0:
            raise ValueError("lora_alpha must be nonnegative")
        if self.clip_epsilon < 0:
            raise ValueError("clip_epsilon must be nonnegative")
        if self.clip_epsilon_high < 0:
            raise ValueError("clip_epsilon_high must be nonnegative")

    @property
    def new_sample_count(self) -> int:
        return self.on_policy_count + self.neighborhood_count

    @property
    def group_size(self) -> int:
        return self.new_sample_count + self.greedy_count

    @property
    def resolved_lora_alpha(self) -> float:
        return float(self.lora_rank if self.lora_alpha == 0 else self.lora_alpha)


def select_greedy(
    history: Sequence[BaselineObservation],
    count: int,
    topk: int,
    rng: random.Random,
) -> list[BaselineObservation]:
    """Uniform sample of ``count`` items from the unique top-``topk`` of ``history``."""
    if count < 1 or not history:
        return []
    unique: dict[str, BaselineObservation] = {}
    for item in history:
        key = solution_key(item.solution)
        previous = unique.get(key)
        if previous is None or (item.score, item.index) > (
            previous.score,
            previous.index,
        ):
            unique[key] = item
    pool = sorted(
        unique.values(), key=lambda item: (item.score, item.index), reverse=True
    )[:topk]
    if count >= len(pool):
        return list(pool)
    return rng.sample(pool, count)


def build_neighborhood_prompt(
    task_description: str,
    greedy: Sequence[BaselineObservation],
    *,
    task: str | None = None,
) -> str:
    """Task prompt plus greedy solutions, asking for a related variant."""
    if task == "molopt":
        return build_completion_fewshot(
            task_description,
            [(item.solution, None) for item in greedy],
        )
    if not greedy:
        return task_description.strip()
    lines = [task_description.strip(), "", "High-scoring solutions:"]
    lines.extend(f"solution: {_one_line(item.solution)}" for item in greedy)
    lines.append("")
    lines.append(
        "Generate a new solution related to those above. Reply with only the solution."
    )
    return "\n".join(lines)


def _grpo_loss(
    new_logps: Sequence[torch.Tensor],
    old_logps: Sequence[torch.Tensor],
    advantages: Sequence[float],
    scores: Sequence[float],
    eps_low: float,
    eps_high: float,
) -> torch.Tensor:
    """Original GRPO: mean of per-sequence token-means (TRL ``loss_type="grpo"``).

    Sequences with reward ``0`` are masked out of the token loss, matching
    ``TTT_GRPOTrainer`` (they still affect the group-mean advantage).
    """
    seq_losses: list[torch.Tensor] = []
    for logp_new, logp_old, advantage, score in zip(
        new_logps, old_logps, advantages, scores
    ):
        if logp_new.numel() == 0 or score == 0:
            seq_losses.append(logp_new.new_zeros(()))
            continue
        ratio = torch.exp((logp_new - logp_old.detach()).clamp(-20.0, 20.0))
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantage
        seq_losses.append((-torch.minimum(unclipped, clipped)).mean())
    return torch.stack(seq_losses).mean()


class _ContextRuntime:
    """Fake-backend path: sample via ``context.generate`` and skip GRPO."""

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

    def update_grpo(
        self, task_prompt: str, group: Sequence[tuple[str, float]]
    ) -> float | None:
        del task_prompt, group
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: Mapping) -> None:
        del state


class LoRAPolicyRuntime(LoRAStudentRuntime):
    """Task-local LoRA used for sampling and mixed-group GRPO."""

    config: MiGrATeConfig

    def __init__(self, context: SearchContext, config: MiGrATeConfig) -> None:
        super().__init__(context, config)
        self.config = config

    def update_grpo(
        self, task_prompt: str, group: Sequence[tuple[str, float]]
    ) -> float | None:
        completions: list[list[int]] = []
        scores: list[float] = []
        for solution, score in group:
            token_ids = self._encode(
                self._completion_text(solution), from_chat=True
            )
            if token_ids:
                completions.append(token_ids)
                scores.append(float(score))
        if len(completions) < 2:
            return None
        mean_score = sum(scores) / len(scores)
        advantages = [score - mean_score for score in scores]
        if all(abs(value) < 1e-12 for value in advantages):
            return None
        prefix, from_chat = self._render(self._candidate_prompt(task_prompt))
        prefix_ids = self._encode(prefix, from_chat)
        if not prefix_ids:
            raise ValueError("migrate GRPO requires a nonempty task prompt")
        last_loss = None
        was_training = self.model.training
        self.model.eval()
        try:
            old_logps = self._token_logprobs(
                prefix_ids, completions, requires_grad=False
            )
            for _ in range(self.config.update_steps):
                new_logps = self._token_logprobs(
                    prefix_ids, completions, requires_grad=True
                )
                loss = _grpo_loss(
                    new_logps,
                    old_logps,
                    advantages,
                    scores,
                    self.config.clip_epsilon,
                    self.config.clip_epsilon_high,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.detach())
        finally:
            self.model.train(was_training)
        return last_loss

    def _token_logprobs(
        self,
        prefix_ids: Sequence[int],
        completions: Sequence[Sequence[int]],
        *,
        requires_grad: bool,
    ) -> list[torch.Tensor]:
        logits = self._labeled_logits(
            prefix_ids, completions, use_ema=False, requires_grad=requires_grad
        )
        logp = F.log_softmax(logits.float(), dim=-1)
        offset = 0
        out: list[torch.Tensor] = []
        for tokens in completions:
            n_tokens = len(tokens)
            token_ids = torch.tensor(
                list(tokens), dtype=torch.long, device=logp.device
            )
            gathered = logp[offset : offset + n_tokens].gather(
                -1, token_ids.unsqueeze(-1)
            ).squeeze(-1)
            out.append(gathered)
            offset += n_tokens
        if offset != logp.shape[0]:
            raise ValueError("migrate GRPO token alignment mismatch")
        return out


class MiGrATeBaseline:
    name = "migrate"

    def __init__(
        self,
        config: MiGrATeConfig | None = None,
        *,
        runtime: PolicyRuntime | None = None,
    ) -> None:
        self.config = config or MiGrATeConfig()
        self._injected_runtime = runtime
        self._runtime: PolicyRuntime | None = runtime
        self._pending_state: dict | None = None
        self._pending_greedy: list[tuple[str, float]] = []
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
        rng = random.Random(context.seed + 1_000_003 * len(history))
        greedy = select_greedy(
            history, self.config.greedy_count, self.config.greedy_topk, rng
        )
        # NS still conditions on a top-k reference when β=0 (original random_topk).
        ns_refs = greedy or (
            select_greedy(history, 1, self.config.greedy_topk, rng)
            if self.config.neighborhood_count > 0
            else []
        )
        n_online = min(self.config.on_policy_count, count)
        n_ns = min(self.config.neighborhood_count, count - n_online)
        task_prompt = context.task_description.strip()
        ns_prompt = build_neighborhood_prompt(
            task_prompt, ns_refs, task=context.task
        )
        sampled = [
            (text, task_prompt, "online")
            for text in self._sample(task_prompt, n_online, context)
        ] + [
            (text, ns_prompt, "neighborhood")
            for text in self._sample(ns_prompt, n_ns, context)
        ]
        if not sampled:
            raise ValueError(f"{self.name} received only blank samples from the policy")
        seen_history = {solution_key(item.solution) for item in history}
        seen_batch: set[str] = set()
        candidates: list[Candidate] = []
        for solution, prompt, provenance in sampled:
            key = solution_key(solution)
            candidates.append(
                Candidate(
                    solution=solution,
                    metadata={
                        "prompt": prompt,
                        "provenance": provenance,
                        "is_repeat_proposal": key in seen_history or key in seen_batch,
                    },
                )
            )
            seen_batch.add(key)
        self._pending_greedy = [(item.solution, item.score) for item in greedy]
        return candidates

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        if not observations:
            self._pending_greedy = []
            return
        if self._context is None or self._runtime is None:
            raise RuntimeError("migrate.observe() requires a prior propose() call")
        group = list(self._pending_greedy)
        group.extend((item.solution, item.score) for item in observations)
        self._pending_greedy = []
        loss = self._runtime.update_grpo(
            self._context.task_description.strip(), group
        )
        if loss is not None:
            print(
                f"[migrate] GRPO group={len(group)}  loss={loss:.4f}",
                flush=True,
            )

    def state_dict(self) -> dict:
        if self._runtime is not None:
            return self._runtime.state_dict()
        return dict(self._pending_state or {})

    def load_state_dict(self, state: Mapping) -> None:
        if not state:
            return
        self._pending_state = dict(state)
        if self._runtime is not None:
            self._runtime.load_state_dict(state)
            self._pending_state = None

    def _bind(self, context: SearchContext) -> None:
        self._context = context
        if self._injected_runtime is not None:
            self._runtime = self._injected_runtime
        elif getattr(context.checkpoint, "reft_model", None) is not None:
            if not isinstance(self._runtime, LoRAPolicyRuntime):
                self._runtime = LoRAPolicyRuntime(context, self.config)
                if self._pending_state is None:
                    self._runtime.reset_parameters()
        else:
            self._runtime = _ContextRuntime(context)
        if self._pending_state is not None:
            self._runtime.load_state_dict(self._pending_state)
            self._pending_state = None

    def _sample(
        self, prompt: str, count: int, context: SearchContext
    ) -> list[str]:
        if count < 1:
            return []
        assert self._runtime is not None
        sampled: list[str] = []
        for _ in range(_BLANK_SAMPLE_RETRIES):
            sampled = []
            for raw in self._runtime.generate(
                prompt, count, context.generation_options
            ):
                text = _normalize_generated_text(str(raw), context.task).strip()
                if text:
                    sampled.append(text)
            if sampled:
                break
        return sampled[:count]
