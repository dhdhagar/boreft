"""Optimization by PROmpting (OPRO).

Reference: Yang et al., "Large Language Models as Optimizers"
https://arxiv.org/abs/2309.03409

Each step serializes scored history into a maximize-score meta-prompt, samples
the base model, and takes one solution per completion. Repeats against the full
history are flagged rather than resampled away, matching the shared loop.

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline opro \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/opro-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .base import (
    BaselineObservation,
    Candidate,
    SearchContext,
    StatelessBaseline,
    solution_key,
)
from .llm_sample import build_completion_fewshot

HistoryStrategy = Literal["all", "top_k", "recent"]


@dataclass(frozen=True)
class OPROConfig:
    history_strategy: HistoryStrategy = "top_k"
    history_count: int = 20

    def __post_init__(self) -> None:
        if self.history_strategy not in ("all", "top_k", "recent"):
            raise ValueError(
                f"unknown OPRO history strategy: {self.history_strategy!r}"
            )
        if self.history_count < 1:
            raise ValueError("history_count must be positive")


def _select_history(
    history: Sequence[BaselineObservation],
    config: OPROConfig,
) -> list[BaselineObservation]:
    """Subset for the prompt, then sort low-to-high score for recency bias."""
    limit = config.history_count
    if config.history_strategy == "all":
        selected = list(history)
    elif config.history_strategy == "top_k":
        selected = sorted(
            history, key=lambda item: (item.score, item.index), reverse=True
        )[:limit]
    elif config.history_strategy == "recent":
        selected = list(history[-limit:])
    else:
        raise ValueError(
            f"unknown OPRO history strategy: {config.history_strategy!r}"
        )
    selected.sort(key=lambda item: (item.score, item.index))
    return selected


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _build_prompt(
    task_description: str,
    history: Sequence[BaselineObservation],
    *,
    history_heading: str = (
        "Previous solutions and scores, ordered from lowest score to highest:"
    ),
    task: str | None = None,
) -> str:
    if task == "molopt":
        return build_completion_fewshot(
            task_description,
            [(item.solution, float(item.score)) for item in history],
        )
    lines = [task_description.strip(), ""]
    if history:
        lines.append(history_heading)
        lines.append("")
        for item in history:
            lines.append(f"solution: {_one_line(item.solution)}")
            lines.append(f"score: {item.score:.6g}")
            lines.append("")
        lines.append("Propose a new solution that scores higher than those above.")
    else:
        lines.append("No previous solutions are available yet.")
        lines.append("Propose a solution for the task.")
    lines.append("Reply with only the solution.")
    return "\n".join(lines)


def _parse_solution(text: str) -> str | None:
    """Use the first non-empty line, stripping an optional ``solution:`` label."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if first.casefold().startswith("solution:"):
        first = first.split(":", 1)[1].strip() or (
            lines[1] if len(lines) > 1 else ""
        )
    return first or None


class OPROBaseline(StatelessBaseline):
    name = "opro"

    def __init__(self, config: OPROConfig | None = None) -> None:
        self.config = config or OPROConfig()

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        selected = _select_history(history, self.config)
        prompt = _build_prompt(context.task_description, selected, task=context.task)
        history_indices = [item.index for item in selected]
        seen_history = {solution_key(item.solution) for item in history}
        seen_batch: set[str] = set()
        candidates: list[Candidate] = []
        for text in context.generate(prompt, count, context.generation_options):
            solution = _parse_solution(text)
            if solution is None:
                continue
            key = solution_key(solution)
            if key in seen_batch:
                continue
            seen_batch.add(key)
            candidates.append(
                Candidate(
                    solution=solution,
                    metadata={
                        "prompt": prompt,
                        "history_indices": history_indices,
                        "is_repeat_proposal": key in seen_history,
                    },
                )
            )
        if not candidates:
            raise ValueError(
                f"{self.name} received only blank samples from the base model"
            )
        return candidates
