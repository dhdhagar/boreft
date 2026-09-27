"""
Semantle word-search data: child dataclass and loaders.

The during-training eval callback and its embed_sim early-stop helpers are
task-neutral and live in :mod:`boreft.data.eval_callback`.
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from boreft.task_config import task_instruction

from .base import ReftItem


@dataclass
class SemantleItem(ReftItem):
    """Semantle item: word as target, with semantic similarity to the target word."""

    similarity: float = 0.0  # cosine similarity to the puzzle's target word

    @staticmethod
    def load_csv(
        csv_path: str, top_k: int = 100
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """Load Semantle CSV (Word, Similarity). Returns (target_word, words, sim_map)."""
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["Word"].strip(), float(row["Similarity"])))
        rows.sort(key=lambda x: -x[1])
        target_word = rows[0][0]
        top_rows = rows[:top_k]
        words = [w for w, _ in top_rows]
        sim_map = {w: s for w, s in top_rows}
        return target_word, words, sim_map

    @staticmethod
    def load_csvs(
        csv_paths: List[str], top_k: int = 100
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """Load several Semantle CSVs and merge vocabularies for one shared intervention.

        For each file, takes the top_k rows (same as load_csv). Union of words across files;
        if a word appears in multiple files, keeps the maximum similarity score. Words are
        ordered by descending merged similarity, then alphabetically for ties.
        """
        if len(csv_paths) == 1:
            return SemantleItem.load_csv(csv_paths[0], top_k=top_k)
        merged: Dict[str, float] = {}
        targets: List[str] = []
        for path in csv_paths:
            target_word, _words, sim_map = SemantleItem.load_csv(path, top_k=top_k)
            targets.append(target_word)
            for w, s in sim_map.items():
                merged[w] = max(merged.get(w, float("-inf")), s)
        words_sorted = sorted(merged.keys(), key=lambda w: (-merged[w], w))
        sim_map_out = {w: merged[w] for w in words_sorted}
        label = f"merged-{len(csv_paths)}files-{len(words_sorted)}words"
        return label, words_sorted, sim_map_out

    @classmethod
    def load(
        cls, words: List[str], sim_map: Optional[Dict[str, float]] = None
    ) -> "List[SemantleItem]":
        """Convert a list of words to SemantleItems using the Semantle plain prompt."""
        prompt = task_instruction("semantle", use_chat_template=False)
        sim_map = sim_map or {}
        return [
            cls(id=i, prompt=prompt, target=w, similarity=sim_map.get(w, 0.0))
            for i, w in enumerate(words)
        ]

    @staticmethod
    def build_augmented_items(
        items: "List[SemantleItem]",
        word_sim_matrix: torch.Tensor,
        nbr_top_k: int,
        nbr_lambda: float = 1.0,
    ) -> "Tuple[List[SemantleItem], int]":
        """Augment items with neighborhood-weighted CE targets.

        ``word_sim_matrix`` must be precomputed pairwise similarity S_ij (same ordering
        as ``items``). Use :func:`text_similarity.pairwise_embedding_similarity_matrix`.

        For each anchor i, takes the top-k other words j by S_ij, and adds training rows
        that use bias b_i with target word j. CE weight is ``nbr_lambda * (S_ij + 1) / 2``
        (clamped nonnegative; can be 0.0 for a slot if you need fixed block sizes).

        Returns ``(flat_list, block_size)`` where:
        - Each **block** is ``[anchor row, NBR_1, ..., NBR_k]`` with ``k = min(nbr_top_k, n-1)``,
          so ``block_size = 1 + k`` always (fixed size for batched sampling).
        - Neighbor slots always appear; **zero weight** if the similarity term would
          have been skipped before (no row dropped).

        Training uses ``data_utils.NeighborhoodBlockBatchSampler`` so each batch can be
        ``c * block_size`` indices = ``c`` shuffled anchor blocks (see ``train.py``).
        """
        n = len(items)
        if n <= 1:
            return list(items), max(1, len(items))

        if word_sim_matrix.shape != (n, n):
            raise ValueError(
                f"word_sim_matrix must be [{n}, {n}], got {tuple(word_sim_matrix.shape)}"
            )

        sim = word_sim_matrix
        if sim.device.type != "cpu":
            sim = sim.detach().cpu()

        out: List[SemantleItem] = []
        k_eff = min(nbr_top_k, n - 1)

        for anchor in items:
            i = anchor.id
            if i < 0 or i >= n:
                raise ValueError(f"SemantleItem id {i} out of range for n={n}")
            out.append(anchor)  # original row (weight=1.0), same object as in ``items``
            row = sim[i].clone()
            row[i] = float("-inf")
            vals, indices = torch.topk(row, k_eff)

            for t in range(k_eff):
                j = int(indices[t].item())
                s_ij = float(vals[t].item())
                w_ij = nbr_lambda * max(0.0, (s_ij + 1.0) / 2.0)
                neighbor = items[j]
                out.append(
                    SemantleItem(
                        id=anchor.id,
                        prompt=anchor.prompt,
                        target=neighbor.target,
                        similarity=s_ij,
                        weight=w_ij,
                    )
                )

        block_size = 1 + k_eff
        return out, block_size
