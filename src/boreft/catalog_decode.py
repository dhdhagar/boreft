"""Compare catalog-max train molecules to decodes at their learned means.

Property search can retrieve a train-set ceiling (discrete BO / SFT) or fail
to, depending on whether decoding at ``μ_i`` recovers that molecule. This
module finds the per-oracle training maximum and scores greedy / T=1 decodes
against it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from boreft.chem import (
    canonical_target_key,
    is_valid_smiles,
    maybe_repair_invalid_smiles,
    tanimoto_similarity,
    unwrap_smiles_tags,
)
from boreft.oracles import TDC_ORACLE_NAMES, normalize_tdc_oracle_name


@dataclass(frozen=True)
class CatalogMax:
    oracle: str
    index: int
    smiles: str
    score: float


@dataclass(frozen=True)
class DecodeMatch:
    decoded: str
    exact: bool
    tanimoto: float | None
    valid: bool
    gold_score: float
    decoded_score: float | None
    temperature: float
    sample: int


def train_smiles_from_words(words: Sequence[str]) -> list[str]:
    """Unwrap checkpoint ``words`` / items.json targets to bare SMILES."""
    out: list[str] = []
    for raw in words:
        text = unwrap_smiles_tags(str(raw).strip())
        if not text:
            raise ValueError("checkpoint train word is blank after unwrap")
        out.append(text)
    return out


def catalog_maxima(
    smiles: Sequence[str],
    scores_by_oracle: Mapping[str, Sequence[float]],
    *,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> list[CatalogMax]:
    """Highest-scoring train row per oracle (first index on ties)."""
    n = len(smiles)
    if n < 1:
        raise ValueError("catalog maxima need at least one train SMILES")
    scores = {
        normalize_tdc_oracle_name(raw_name): values
        for raw_name, values in scores_by_oracle.items()
    }
    found: list[CatalogMax] = []
    for raw_name in oracles:
        name = normalize_tdc_oracle_name(raw_name)
        values = scores.get(name)
        if values is None:
            raise ValueError(f"missing scores for oracle {name}")
        if len(values) != n:
            raise ValueError(
                f"{name} has {len(values)} scores for {n} train SMILES"
            )
        best_index = max(range(n), key=lambda i: (float(values[i]), -i))
        found.append(
            CatalogMax(
                oracle=name,
                index=int(best_index),
                smiles=str(smiles[best_index]),
                score=float(values[best_index]),
            )
        )
    return found


def compare_decode(
    gold: str,
    decoded: str,
    *,
    gold_score: float,
    decoded_score: float | None = None,
    temperature: float = 0.0,
    sample: int = 0,
) -> DecodeMatch:
    """Canonical exact-match, Tanimoto, and optional oracle score of a decode."""
    prepared, _repaired = maybe_repair_invalid_smiles(decoded)
    gold_key = canonical_target_key(gold)
    dec_key = canonical_target_key(prepared)
    valid = is_valid_smiles(prepared)
    tani = tanimoto_similarity(gold, prepared) if valid else None
    return DecodeMatch(
        decoded=prepared,
        exact=bool(gold_key) and gold_key == dec_key,
        tanimoto=None if tani is None else float(tani),
        valid=valid,
        gold_score=float(gold_score),
        decoded_score=None if decoded_score is None else float(decoded_score),
        temperature=float(temperature),
        sample=int(sample),
    )


def match_payload(row: DecodeMatch) -> dict:
    return asdict(row)
