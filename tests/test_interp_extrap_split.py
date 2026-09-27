"""Exploratory test: GENZ interp/extrap split vs. --eval-bbox-pca-var.

Builds a realistic 512-word train set and a 512-word (default ``test_n_samples``)
non-train test set from the repo's Semantle CSVs, encodes both with the eval
embedding model, and prints how the PCA bounding-box splits the test set into
interp vs. extrap for a range of ``pca_var`` values.

This is a diagnostic aid rather than a strict unit test — run it with output
visible to inspect the split:

    pytest -s tests/test_interp_extrap_split.py

It is skipped automatically when the embedding model cannot be loaded (e.g. no
network / model cache in CI).
"""

from __future__ import annotations

import glob
import os
import unittest

from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_TEST_N_SAMPLES,
    build_non_train_pool,
    normalize_text,
    split_interp_extrap,
)

N_TRAIN = 512
PCA_VARS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
SEED = 42
N_SAMPLE_WORDS = 15  # words to preview per group (train / interp / extrap)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_DIR = os.path.join(_REPO_ROOT, "data", "semantle", "train")


def _merged_train_words(csv_paths: list[str], n_train: int) -> list[str]:
    """First ``n_train`` distinct words by descending per-CSV similarity (merged)."""
    from boreft.eval.eval_suite import _read_csv_words_sorted

    seen: dict[str, None] = {}
    # Round-robin across CSVs so the train set spans all puzzles.
    per_csv = [_read_csv_words_sorted(p) for p in csv_paths]
    idx = 0
    while len(seen) < n_train and any(idx < len(w) for w in per_csv):
        for words in per_csv:
            if idx < len(words):
                seen.setdefault(normalize_text(words[idx]), None)
                if len(seen) >= n_train:
                    break
        idx += 1
    return list(seen.keys())


class InterpExtrapSplitTests(unittest.TestCase):
    def test_split_across_pca_var(self) -> None:
        csv_paths = sorted(glob.glob(os.path.join(_CSV_DIR, "*.csv")))
        if not csv_paths:
            self.skipTest(f"no semantle CSVs under {_CSV_DIR}")

        train_words = _merged_train_words(csv_paths, N_TRAIN)
        if len(train_words) < N_TRAIN:
            self.skipTest(
                f"only {len(train_words)} distinct train words available (< {N_TRAIN})"
            )

        pool = build_non_train_pool(csv_paths, train_targets=train_words)
        if len(pool) < DEFAULT_TEST_N_SAMPLES:
            self.skipTest(
                f"non-train pool has {len(pool)} words (< {DEFAULT_TEST_N_SAMPLES})"
            )

        import random

        test_words = random.Random(SEED).sample(sorted(pool), DEFAULT_TEST_N_SAMPLES)

        try:
            from sklearn.decomposition import PCA

            from boreft.text_similarity import encode_texts_normalized

            train_emb = encode_texts_normalized(train_words)
            test_emb = encode_texts_normalized(test_words)
        except Exception as e:  # missing model / network / deps
            self.skipTest(f"embedding model unavailable: {e}")

        print(
            f"\n[interp/extrap] train={len(train_words)}  test={len(test_words)}  "
            f"embed_dim={train_emb.shape[1]}"
        )
        print(f"{'pca_var':>8} {'n_comp':>7} {'interp':>7} {'extrap':>7} {'interp%':>8}")
        splits: dict[float, tuple[list[str], list[str]]] = {}
        for pca_var in PCA_VARS:
            n_comp = (
                PCA(n_components=pca_var, svd_solver="full")
                .fit(train_emb)
                .n_components_
            )
            interp, extrap = split_interp_extrap(
                train_emb, test_emb, test_words, pca_var=pca_var
            )
            splits[pca_var] = (interp, extrap)
            self.assertEqual(len(interp) + len(extrap), len(test_words))
            frac = len(interp) / len(test_words)
            print(
                f"{pca_var:>8.2f} {n_comp:>7d} {len(interp):>7d} "
                f"{len(extrap):>7d} {frac:>7.1%}"
            )

        # Preview sample words per group at the default pca_var.
        default_var = DEFAULT_BBOX_PCA_VAR
        interp, extrap = splits.get(
            default_var,
            split_interp_extrap(train_emb, test_emb, test_words, pca_var=default_var),
        )
        rng = random.Random(SEED)

        def _preview(words: list[str]) -> str:
            n = min(N_SAMPLE_WORDS, len(words))
            return ", ".join(rng.sample(words, n)) if words else "(none)"

        print(f"\n[sample words] (interp/extrap at pca_var={default_var})")
        print(f"  train  ({len(train_words)}): {_preview(train_words)}")
        print(f"  interp ({len(interp)}): {_preview(interp)}")
        print(f"  extrap ({len(extrap)}): {_preview(extrap)}")


if __name__ == "__main__":
    unittest.main()
