"""Independent sampling from the base model distribution.

The prompt is the task description plus the last 20 unique previous candidates
(history plus earlier samples in the same step) so the model can avoid repeats.
``--random-sampling.last-k-incontext 0`` restores a task-only prompt with no
history. ``--random-sampling.candidates-per-call`` asks for that many numbered
candidates in each completion (``1. ...`` per line; default 1) and scores all of
them in that search step, even when ``--batch-size`` is 1. Set ``--batch-size``
larger than ``candidates-per-call`` to score more than one packed completion per
step. Repeated solutions are accepted and flagged rather than resampled away.

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline random_sampling \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/random-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite

``--lora-sft-adapter`` loads a trained proposal LoRA onto the frozen LM after
the per-seed LoRA clear, so Random samples from the SFT word distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import (
    BaselineObservation,
    Candidate,
    SearchContext,
    StatelessBaseline,
    solution_key,
)
from .llm_sample import sample_llm_candidates, validate_llm_sample_knobs

_BLANK_SAMPLE_RETRIES = 3


@dataclass(frozen=True)
class RandomSamplingConfig:
    last_k_incontext: int = 20
    candidates_per_call: int = 1

    def __post_init__(self) -> None:
        validate_llm_sample_knobs(
            last_k_incontext=self.last_k_incontext,
            candidates_per_call=self.candidates_per_call,
        )


class RandomSamplingBaseline(StatelessBaseline):
    name = "random_sampling"

    def __init__(self, config: RandomSamplingConfig | None = None) -> None:
        self.config = config or RandomSamplingConfig()

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        seen = {solution_key(item.solution) for item in history}
        previous = [item.solution for item in history]
        sampled = []
        for _ in range(_BLANK_SAMPLE_RETRIES):
            sampled = sample_llm_candidates(
                context.generate,
                context.generation_options,
                context.task_description,
                count=count,
                last_k_incontext=self.config.last_k_incontext,
                candidates_per_call=self.config.candidates_per_call,
                previous=previous,
                task=context.task,
            )
            if sampled:
                break
        if not sampled:
            raise ValueError(
                f"{self.name} received only blank samples from the base model"
            )
        candidates = []
        for item in sampled:
            key = solution_key(item.solution)
            candidates.append(
                Candidate(
                    solution=item.solution,
                    metadata={
                        "prompt": item.prompt,
                        "is_repeat_proposal": key in seen,
                    },
                )
            )
            seen.add(key)
        return candidates
