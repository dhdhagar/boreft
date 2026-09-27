"""
Molecular optimization data: child dataclass and loader.

The during-training eval callback is task-neutral (see
:mod:`boreft.data.eval_callback`); molecule similarity comes from the task's
embedding model in :mod:`boreft.text_similarity` (semantic) and Morgan
fingerprints in :mod:`boreft.chem` (structural).
"""

import csv
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from boreft.chem import generation_smiles, unwrap_smiles_tags
from boreft.task_config import task_instruction

from .base import ReftItem, item_target

SMILES_COLUMN = "smiles"
INCHIKEY_COLUMN = "inchikey"


def _iter_rows(
    csv_path: str, top_k: Optional[int] = None
) -> Iterator[Tuple[str, Dict[str, str]]]:
    """``(smiles, row)`` per usable row, in file order, skipping blank SMILES.

    ``top_k`` counts kept rows, so a blank row never consumes a vocabulary slot.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or SMILES_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: molopt CSV needs a {SMILES_COLUMN!r} column "
                f"(found {reader.fieldnames})"
            )
        kept = 0
        for row in reader:
            if top_k is not None and kept >= top_k:
                return
            smiles = (row.get(SMILES_COLUMN) or "").strip()
            if not smiles:
                continue
            kept += 1
            yield smiles, row


def read_csv_smiles(csv_path: str, top_k: Optional[int] = None) -> List[str]:
    """SMILES strings from a molopt CSV, in file order, blank rows skipped.

    ``top_k`` caps how many rows are read (matching the training vocabulary head);
    ``None`` reads the whole file, which is what the held-out pool needs.
    """
    return [smiles for smiles, _ in _iter_rows(csv_path, top_k)]


@dataclass
class MolOptItem(ReftItem):
    """One molecule; target is its SMILES string."""

    inchikey: str = ""

    @classmethod
    def load_csv(cls, csv_path: str, top_k: Optional[int] = None) -> "List[MolOptItem]":
        """Load one MolOptItem per row, in file order.

        ``smiles`` is required and ``inchikey`` read when present; objective columns
        (docking scores, QED, …) are carried in the CSV but unused here. ``top_k``
        caps the vocabulary size, leaving later rows for the held-out eval pool.
        """
        prompt = task_instruction("molopt", use_chat_template=False)
        return [
            cls(
                id=index,
                prompt=prompt,
                target=smiles,
                inchikey=(row.get(INCHIKEY_COLUMN) or "").strip(),
            )
            for index, (smiles, row) in enumerate(_iter_rows(csv_path, top_k))
        ]


def apply_smiles_tags(items: List[MolOptItem]) -> None:
    """Rewrite generation targets as ``<SMILES>...</SMILES>``; keep identity bare.

    Sets ``_raw_word`` to the untagged SMILES so embeddings, definitions, and
    exact-match keys stay on the molecule, while ``target`` (the supervised
    suffix) includes the tags.
    """
    for item in items:
        bare = unwrap_smiles_tags(item_target(item).strip())
        item._raw_word = bare
        item.target = generation_smiles(bare, smiles_tags=True)


def apply_mist_smiles_tags(
    items: List[MolOptItem], *, include_open: bool = False
) -> None:
    """Rewrite generation targets with MiST SMILES tags; keep identity bare.

    Completion (``include_open=False``): gold is ``SMILES [END_SMILES]``; the
    prompt (not the gold) carries ``[START_SMILES]``. Chat gold includes the
    open tag. Sets ``_raw_word`` so embeddings, definitions, and exact-match
    keys stay on the molecule.
    """
    for item in items:
        bare = unwrap_smiles_tags(item_target(item).strip())
        item._raw_word = bare
        item.target = generation_smiles(
            bare, mist_smiles_tags=True, mist_open_in_target=include_open
        )
