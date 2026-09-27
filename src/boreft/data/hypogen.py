"""
Hypothesis-generation data: child dataclass and loader.

Targets are natural-language hypotheses. Semantic similarity uses the task's
embedding model in :mod:`boreft.text_similarity`; there is no structural
fingerprint family.
"""

import csv
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from boreft.task_config import task_instruction

from .base import ReftItem

HYPOTHESIS_COLUMN = "hypothesis"


def _iter_rows(
    csv_path: str, top_k: Optional[int] = None
) -> Iterator[Tuple[str, Dict[str, str]]]:
    """``(hypothesis, row)`` per usable row, in file order, skipping blanks.

    ``top_k`` counts kept rows, so a blank row never consumes a vocabulary slot.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or HYPOTHESIS_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: hypogen CSV needs a {HYPOTHESIS_COLUMN!r} column "
                f"(found {reader.fieldnames})"
            )
        kept = 0
        for row in reader:
            if top_k is not None and kept >= top_k:
                return
            hypothesis = (row.get(HYPOTHESIS_COLUMN) or "").strip()
            if not hypothesis:
                continue
            kept += 1
            yield hypothesis, row


def read_csv_hypotheses(csv_path: str, top_k: Optional[int] = None) -> List[str]:
    """Hypothesis strings from a hypogen CSV, in file order, blank rows skipped.

    ``top_k`` caps how many rows are read (matching the training vocabulary head);
    ``None`` reads the whole file, which is what the held-out pool needs.
    """
    return [hypothesis for hypothesis, _ in _iter_rows(csv_path, top_k)]


@dataclass
class HypoGenItem(ReftItem):
    """One hypothesis; target is the hypothesis string."""

    @classmethod
    def load_csv(cls, csv_path: str, top_k: Optional[int] = None) -> "List[HypoGenItem]":
        """Load one HypoGenItem per row, in file order.

        ``top_k`` caps the vocabulary size, leaving later rows for the held-out
        eval pool.
        """
        prompt = task_instruction("hypogen", use_chat_template=False)
        return [
            cls(id=index, prompt=prompt, target=hypothesis)
            for index, (hypothesis, _row) in enumerate(_iter_rows(csv_path, top_k))
        ]
