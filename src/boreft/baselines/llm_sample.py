"""Shared LLM sampling: last-k in-context conditioning and multi-candidate completions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from boreft.chem import wrap_mist_smiles_tags

from .base import Generate, GenerationOptions, solution_key

_GENERATION_SLOT_MULTIPLIER = 8
_NUMBERED_PREFIX = re.compile(r"^\s*\d+[\.\)\:]\s*")


def validate_llm_sample_knobs(
    *, last_k_incontext: int, candidates_per_call: int
) -> None:
    if last_k_incontext < 0:
        raise ValueError("last_k_incontext must be nonnegative")
    if candidates_per_call < 1:
        raise ValueError("candidates_per_call must be positive")


def parse_generated_candidates(text: str, limit: int) -> list[str]:
    """Parse one completion into candidates.

    A single-candidate completion is the full stripped text, including internal
    newlines (hypotheses). Multi-candidate completions are split on lines, and
    a leading ``1.`` / ``1)`` / ``1:`` index is stripped.
    """
    if limit < 1:
        raise ValueError("candidates_per_call must be positive")
    if limit == 1:
        solution = str(text).strip()
        return [solution] if solution else []
    solutions: list[str] = []
    for line in str(text).splitlines():
        solution = _NUMBERED_PREFIX.sub("", line.strip(), count=1).strip()
        if not solution:
            continue
        solutions.append(solution)
        if len(solutions) >= limit:
            break
    return solutions


def _one_line(text: str) -> str:
    return " ".join(text.split())


def split_completion_header(task_description: str) -> tuple[str | None, str]:
    """Split an optional one-line header from the repeating completion prefix."""
    text = task_description.strip()
    if "\n" not in text:
        return None, text
    header, prefix = text.split("\n", 1)
    header, prefix = header.strip(), prefix.strip()
    if not prefix:
        return None, header
    return (header or None), prefix


def build_completion_fewshot(
    prefix: str,
    examples: Sequence[tuple[str, float | None]] = (),
    *,
    wrap_history: bool = True,
) -> str:
    """Repeat a pretrained completion prefix, then leave it open for decode.

    A leading header line (before the first newline in ``prefix``) is written
    once. History molecules are wrapped in MiST ``[START_SMILES]`` /
    ``[END_SMILES]``. The prompt always ends with the completion prefix; MiST
    decode then appends ``[START_SMILES]`` last.
    """
    header, prefix = split_completion_header(prefix)
    if not prefix:
        raise ValueError("completion prefix must not be blank")
    blocks: list[str] = []
    for text, score in examples:
        solution = _one_line(text)
        if not solution:
            continue
        if wrap_history:
            solution = wrap_mist_smiles_tags(solution, include_open=True)
        block = f"{prefix} {solution}"
        if score is not None:
            block = f"{block}\nscore: {float(score):.6g}"
        blocks.append(block)
    blocks.append(prefix)
    body = "\n\n".join(blocks)
    if header:
        return f"{header}\n{body}"
    return body


def recent_incontext(previous: Sequence[str], last_k_incontext: int) -> list[str]:
    """Most recent ``last_k_incontext`` unique previous candidates, oldest first."""
    if last_k_incontext <= 0:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    for item in reversed(previous):
        key = solution_key(item)
        flattened = _one_line(item)
        if not key or not flattened or key in seen:
            continue
        seen.add(key)
        selected.append(flattened)
        if len(selected) >= last_k_incontext:
            break
    selected.reverse()
    return selected


def build_candidate_prompt(
    task_description: str,
    *,
    previous: Sequence[str],
    last_k_incontext: int,
    n_candidates: int,
    task: str | None = None,
) -> str:
    """Task prompt, optionally with last-k in-context samples and a multi-candidate request."""
    validate_llm_sample_knobs(
        last_k_incontext=last_k_incontext, candidates_per_call=n_candidates
    )
    shown = recent_incontext(previous, last_k_incontext)
    if task == "molopt":
        return build_completion_fewshot(
            task_description,
            [(item, None) for item in shown],
        )
    lines = [task_description.strip()]
    if not shown and n_candidates == 1:
        return lines[0]
    lines.append("")
    if shown:
        lines.append("Previous candidates:")
        lines.extend(shown)
        lines.append("")
        if n_candidates == 1:
            lines.append(
                "Generate a new candidate different from those above. "
                "Reply with only the candidate."
            )
        else:
            lines.append(
                f"Generate {n_candidates} new candidates, one numbered new line each "
                f"(e.g., 1. word1\n2. word2\n...), different from those above. "
                "Reply with only the numbered candidates."
            )
    else:
        lines.append(
            f"Generate {n_candidates} candidates, one numbered new line each "
            f"(e.g., 1. word1\n2. word2\n...). "
            "Reply with only the numbered candidates."
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class SampledCandidate:
    solution: str
    prompt: str


def sample_llm_candidates(
    generate: Generate,
    options: GenerationOptions,
    task_description: str,
    *,
    count: int,
    last_k_incontext: int = 0,
    candidates_per_call: int = 1,
    previous: Sequence[str] = (),
    unique: bool = False,
    task: str | None = None,
) -> list[SampledCandidate]:
    """Sample ``count`` candidates, optionally unique, from sequential or batched calls."""
    validate_llm_sample_knobs(
        last_k_incontext=last_k_incontext, candidates_per_call=candidates_per_call
    )
    if count < 1:
        raise ValueError("candidate count must be positive")
    packed = 1 if task == "molopt" else candidates_per_call
    collected: list[SampledCandidate] = []
    seen = {solution_key(item) for item in previous if str(item).strip()}
    rolling = [item for item in previous if str(item).strip()]
    slots_used = 0
    if unique or last_k_incontext > 0:
        slot_budget = count * _GENERATION_SLOT_MULTIPLIER
    else:
        slot_budget = math.ceil(count / packed) * packed

    while len(collected) < count and slots_used < slot_budget:
        remaining = count - len(collected)
        if last_k_incontext > 0:
            n_calls = 1
        else:
            n_calls = min(
                math.ceil(remaining / packed),
                max(1, (slot_budget - slots_used) // packed),
            )
        prompt = build_candidate_prompt(
            task_description,
            previous=rolling,
            last_k_incontext=last_k_incontext,
            n_candidates=packed,
            task=task,
        )
        texts = list(generate(prompt, n_calls, options))
        slots_used += max(n_calls, 1) * packed
        for text in texts:
            for solution in parse_generated_candidates(text, packed):
                key = solution_key(solution)
                if unique and key in seen:
                    continue
                seen.add(key)
                rolling.append(solution)
                collected.append(SampledCandidate(solution=solution, prompt=prompt))
                if len(collected) >= count:
                    break
            if len(collected) >= count:
                break
    return collected
