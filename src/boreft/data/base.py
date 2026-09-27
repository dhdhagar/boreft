"""
Base data types for REFT training.

ReftItem — parent dataclass for all task-specific items.
parse_positions, get_intervention_locations — shared utilities for intervention placement.
"""

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple


@dataclass
class ReftItem:
    """
    Parent dataclass for one REFT training example.
    Child classes: SemantleItem, MolOptItem, HypoGenItem, ArcItem.
    """

    id: int  # integer index → word_mu / word_logvar row (DistributionalWordIntervention)
    prompt: str  # text fed to the LLM (no target suffix)
    target: str  # text the model should generate
    weight: float = 1.0  # loss weight; < 1.0 for neighborhood-augmented examples


def item_target(item: ReftItem) -> str:
    """Undecorated target text for an item.

    ``target`` carries chat/EOS decoration once the item has been formatted for
    training, so ``_raw_word`` (stashed by ``data_utils``) is preferred when set.
    Anything comparing or embedding a target wants this, not ``item.target``.
    """
    return str(getattr(item, "_raw_word", None) or item.target)


def sample_eval_subset(items: List, n: Optional[int], seed: int) -> List:
    """Return a fixed eval subset of ``n`` items (same draw for a given seed)."""
    items = list(items)
    if n is None or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def parse_positions(positions: str) -> Tuple[int, int]:
    """Parse position string (e.g. 'l1', 'f1', 'f1+l1') into (first_n, last_n)."""
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    elif "f" in positions:
        first_n = int(positions.strip("f"))
    elif "l" in positions:
        last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(
    *,
    last_position: int,
    first_n: int,
    last_n: int,
    num_interventions: int,
    share_weights: bool = False,
    pad_mode: str = "first",
) -> List[List[int]]:
    """Compute intervention locations within the prompt."""
    first_n = min(last_position // 2, first_n)
    last_n = min(last_position // 2, last_n)

    if share_weights or (first_n == 0 or last_n == 0):
        locs = [i for i in range(first_n)] + [
            i for i in range(last_position - last_n, last_position)
        ]
        return [locs] * num_interventions

    left_locs = [i for i in range(first_n)]
    right_locs = [i for i in range(last_position - last_n, last_position)]
    max_len = max(len(left_locs), len(right_locs))
    pad_pos = -1 if pad_mode == "first" else last_position
    left_locs += [pad_pos] * (max_len - len(left_locs))
    right_locs += [pad_pos] * (max_len - len(right_locs))
    return [left_locs] * (num_interventions // 2) + [right_locs] * (
        num_interventions // 2
    )
