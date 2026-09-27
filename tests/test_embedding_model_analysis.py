"""Tests for ``scripts/analyze_embedding_models.py``.

The experiment's conclusions are only worth as much as its scoring, so the parts
that turn embeddings into numbers are pinned here against inputs whose answers
are known by construction: a perfectly aligned space must retrieve at 1.0, an
independent one must retrieve at chance, and the leakage control must actually
strip the SMILES out of the definition text.

Encoders are stubbed. Loading five pretrained models is the experiment's job, not
the test suite's.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    """Import the script by path; it lives in scripts/, not the package."""
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_embedding_models.py")
    spec = importlib.util.spec_from_file_location("analyze_embedding_models", path)
    module = importlib.util.module_from_spec(spec)
    # dataclass() resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aem = _load_script()

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


class _StubEncoder(aem.Encoder):
    """Returns fixed rows and records which strings each tower was asked for."""

    def __init__(self, rows_by_text=None, dim=8, dual_tower=False):
        self.key = "stub"
        self.label = "stub"
        self.dual_tower = dual_tower
        self.rows_by_text = rows_by_text or {}
        self.dim = dim
        self.text_calls: list[list[str]] = []
        self.molecule_calls: list[tuple[list[str], list[str]]] = []
        self.closed = False

    def _encode_text(self, texts, batch_size):
        self.text_calls.append(list(texts))
        return np.asarray(
            [
                self.rows_by_text.get(t, np.arange(1, self.dim + 1, dtype=np.float64))
                for t in texts
            ],
            dtype=np.float64,
        )

    def encode_molecules(self, decorated, raw_smiles, batch_size):
        self.molecule_calls.append((list(decorated), list(raw_smiles)))
        return super().encode_molecules(decorated, raw_smiles, batch_size)

    def close(self):
        self.closed = True


def _orthonormal(n: int, seed: int = 0) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


class TextTreatmentTests(unittest.TestCase):
    def test_bare_molecule_is_the_smiles_itself(self):
        self.assertEqual(aem.molecule_text(f"  {ASPIRIN} ", aem.MOL_BARE), ASPIRIN)

    def test_prompted_molecule_wraps_the_smiles(self):
        out = aem.molecule_text(ASPIRIN, aem.MOL_PROMPTED)
        self.assertIn(ASPIRIN, out)
        self.assertNotEqual(out, ASPIRIN)

    def test_bare_definition_carries_no_template(self):
        self.assertEqual(aem.definition_text(ASPIRIN, "  a drug ", aem.DEFN_BARE), "a drug")

    def test_prompted_definition_leaks_the_smiles(self):
        """Documents the confound the no-smiles variant exists to remove."""
        out = aem.definition_text(ASPIRIN, "a drug", aem.DEFN_PROMPTED)
        self.assertIn(ASPIRIN, out)
        self.assertIn("a drug", out)

    def test_no_smiles_definition_strips_the_molecule_but_keeps_the_template(self):
        prompted = aem.definition_text(ASPIRIN, "a drug", aem.DEFN_PROMPTED)
        stripped = aem.definition_text(ASPIRIN, "a drug", aem.DEFN_NO_SMILES)
        self.assertNotIn(ASPIRIN, stripped)
        self.assertIn("a drug", stripped)
        self.assertNotEqual(stripped, prompted)

    def test_no_smiles_definition_is_constant_across_molecules(self):
        """A row-invariant placeholder cannot carry retrieval signal."""
        self.assertEqual(
            aem.definition_text(ASPIRIN, "a drug", aem.DEFN_NO_SMILES),
            aem.definition_text(ETHANOL, "a drug", aem.DEFN_NO_SMILES),
        )

    def test_unknown_treatments_raise(self):
        with self.assertRaises(ValueError):
            aem.molecule_text(ASPIRIN, "nope")
        with self.assertRaises(ValueError):
            aem.definition_text(ASPIRIN, "a drug", "nope")

    def test_variant_names_describe_both_sides(self):
        for variant in aem.VARIANTS:
            self.assertEqual(variant.name, f"{variant.mol}__{variant.defn}")

    def test_variant_names_are_unique(self):
        names = [v.name for v in aem.VARIANTS]
        self.assertEqual(len(names), len(set(names)))

    def test_the_undecorated_definition_variant_exists(self):
        """The honest cross-modal row: prompted molecule, no definition template."""
        self.assertIn(
            aem.variant_name(aem.MOL_PROMPTED, aem.DEFN_BARE),
            [v.name for v in aem.VARIANTS],
        )


class RetrievalMetricTests(unittest.TestCase):
    def test_perfect_alignment_retrieves_everything(self):
        emb = _orthonormal(12)
        res = aem.alignment_metrics(emb, emb)
        self.assertAlmostEqual(res["retrieval"]["recall_at_1"], 1.0)
        self.assertAlmostEqual(res["retrieval"]["mrr"], 1.0)
        self.assertAlmostEqual(res["retrieval"]["median_rank"], 1.0)

    def test_independent_spaces_retrieve_at_chance(self):
        rng = np.random.default_rng(0)
        n = 200
        mol = rng.normal(size=(n, 32))
        txt = rng.normal(size=(n, 32))
        mol /= np.linalg.norm(mol, axis=1, keepdims=True)
        txt /= np.linalg.norm(txt, axis=1, keepdims=True)
        res = aem.alignment_metrics(mol, txt)
        chance = res["retrieval"]["chance_recall_at_1"]
        self.assertAlmostEqual(chance, 1.0 / n)
        self.assertLess(res["retrieval"]["recall_at_1"], 20 * chance)
        self.assertLess(abs(res["gap_matched_minus_random"]), 0.05)

    def test_chance_is_one_over_n(self):
        emb = _orthonormal(7)
        res = aem.alignment_metrics(emb, emb)
        self.assertAlmostEqual(res["retrieval"]["chance_recall_at_1"], 1.0 / 7)

    def test_recall_is_monotone_in_k(self):
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(50, 16))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        r = aem.alignment_metrics(emb, emb + 0.5 * rng.normal(size=emb.shape))[
            "retrieval"
        ]
        self.assertLessEqual(r["recall_at_1"], r["recall_at_5"])
        self.assertLessEqual(r["recall_at_5"], r["recall_at_10"])

    def test_both_retrieval_directions_are_reported(self):
        emb = _orthonormal(9)
        res = aem.alignment_metrics(emb, emb)
        self.assertIn("retrieval", res)
        self.assertIn("retrieval_molecule_to_text", res)

    def test_matched_and_random_are_separated_correctly(self):
        """Matched values come off the diagonal; the control is everything else."""
        emb = _orthonormal(10)
        res = aem.alignment_metrics(emb, emb)
        np.testing.assert_allclose(res["_matched_values"], np.ones(10), atol=1e-9)
        self.assertEqual(res["_random_values"].size, 10 * 10 - 10)
        np.testing.assert_allclose(res["_random_values"], 0.0, atol=1e-9)

    def test_a_collapsed_space_reports_zero_effect_size(self):
        """Every pair identical: no gap, and d must not divide by zero."""
        emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (6, 1))
        res = aem.alignment_metrics(emb, emb)
        self.assertAlmostEqual(res["gap_matched_minus_random"], 0.0)
        self.assertEqual(res["cohens_d"], 0.0)


class StructureAgreementTests(unittest.TestCase):
    def test_identical_molecules_have_tanimoto_one(self):
        smiles = [ASPIRIN, ASPIRIN]
        emb = _orthonormal(2)
        out = aem.structure_agreement(emb, smiles, [(0, 1)])
        self.assertAlmostEqual(out["tanimoto"]["mean"], 1.0)

    def test_correlation_is_omitted_when_a_metric_is_constant(self):
        """Pearson/Spearman are undefined here, and printing them would mislead."""
        smiles = [ASPIRIN, ASPIRIN, ASPIRIN]
        emb = np.tile(np.array([[1.0, 0.0]]), (3, 1))
        out = aem.structure_agreement(emb, smiles, [(0, 1), (1, 2)])
        self.assertNotIn("pearson_r", out)

    def test_pair_count_is_reported(self):
        smiles = [ASPIRIN, ETHANOL, BENZENE]
        emb = _orthonormal(3)
        out = aem.structure_agreement(emb, smiles, [(0, 1), (1, 2), (0, 2)])
        self.assertEqual(out["n_pairs"], 3)


class SamplingTests(unittest.TestCase):
    def test_structure_pairs_are_distinct_indices(self):
        pairs = aem.sample_structure_pairs(10, 50, seed=0)
        self.assertEqual(len(pairs), 50)
        self.assertTrue(all(i != j for i, j in pairs))

    def test_structure_pairs_are_seeded(self):
        """Every model must be compared on the same pairs."""
        self.assertEqual(
            aem.sample_structure_pairs(20, 30, seed=3),
            aem.sample_structure_pairs(20, 30, seed=3),
        )
        self.assertNotEqual(
            aem.sample_structure_pairs(20, 30, seed=3),
            aem.sample_structure_pairs(20, 30, seed=4),
        )


class RunEncoderTests(unittest.TestCase):
    PAIRS = [
        (ASPIRIN, "a salicylate ester"),
        (ETHANOL, "a primary alcohol"),
        (BENZENE, "an aromatic hydrocarbon"),
    ]
    STRUCTURE_PAIRS = [(0, 1), (1, 2)]

    def _run(self, encoder):
        return aem.run_encoder(encoder, self.PAIRS, self.STRUCTURE_PAIRS, batch_size=2)

    def test_single_tower_runs_every_variant(self):
        out = self._run(_StubEncoder())
        self.assertEqual(list(out["variants"]), [v.name for v in aem.VARIANTS])
        self.assertEqual(out["skipped_variants"], {})

    def test_dual_tower_folds_molecule_decoration_onto_the_bare_smiles(self):
        out = self._run(_StubEncoder(dual_tower=True))
        for name in out["variants"]:
            self.assertTrue(name.startswith(f"{aem.MOL_BARE}__"), name)

    def test_dual_tower_keeps_every_definition_side_treatment(self):
        """The text tower reads text, so these comparisons stay meaningful."""
        out = self._run(_StubEncoder(dual_tower=True))
        self.assertEqual(
            sorted(name.split("__", 1)[1] for name in out["variants"]),
            sorted({v.defn for v in aem.VARIANTS}),
        )

    def test_dual_tower_skips_only_the_duplicated_row(self):
        out = self._run(_StubEncoder(dual_tower=True))
        self.assertEqual(
            list(out["skipped_variants"]),
            [aem.variant_name(aem.MOL_PROMPTED, aem.DEFN_BARE)],
        )
        self.assertIn("no-op", next(iter(out["skipped_variants"].values())))

    def test_molecule_side_receives_raw_smiles_alongside_decorated_text(self):
        encoder = _StubEncoder()
        self._run(encoder)
        for decorated, raw in encoder.molecule_calls:
            self.assertEqual(raw, [s for s, _ in self.PAIRS])
            self.assertEqual(len(decorated), len(raw))

    def test_each_distinct_string_set_is_encoded_once(self):
        """Four variants share two molecule sets and three definition sets."""
        encoder = _StubEncoder()
        self._run(encoder)
        self.assertEqual(len(encoder.molecule_calls), 2)
        self.assertEqual(len(encoder.text_calls), 2 + 3)

    def test_structure_agreement_covers_both_scored_spaces(self):
        out = self._run(_StubEncoder())
        self.assertEqual(
            sorted(out["structure"]), sorted([aem.MOL_PROMPTED, aem.DEFN_PROMPTED])
        )


class ReportingTests(unittest.TestCase):
    def test_raw_arrays_are_stripped_before_serialization(self):
        emb = _orthonormal(4)
        res = {"variants": {"v": aem.alignment_metrics(emb, emb)}}
        stripped = aem.strip_arrays(res)
        self.assertNotIn("_matched_values", stripped["variants"]["v"])
        self.assertIn("matched", stripped["variants"]["v"])

    def test_stripped_report_is_json_serializable(self):
        emb = _orthonormal(4)
        json_ready = aem.strip_arrays({"v": aem.alignment_metrics(emb, emb)})
        import json

        json.dumps(json_ready)

    def test_variant_order_is_stable_across_models(self):
        results = {
            "a": {"variants": {"x": {}, "y": {}}},
            "b": {"variants": {"y": {}, "z": {}}},
        }
        self.assertEqual(aem.ordered_variant_names(results), ["x", "y", "z"])

    def test_chance_is_none_when_no_model_produced_a_variant(self):
        self.assertIsNone(aem.chance_recall_at_1({"a": {"variants": {}}}))


class EncoderRegistryTests(unittest.TestCase):
    def test_every_requested_model_is_registered(self):
        for key in ("chemate", "molt5", "moleculestm", "qwen3"):
            self.assertIn(key, aem.ENCODER_KEYS)

    def test_all_four_compared_models_run_by_default(self):
        self.assertEqual(
            sorted(aem.DEFAULT_ENCODER_KEYS),
            sorted(["chemate", "molt5", "moleculestm", "qwen3"]),
        )

    def test_the_sentence_transformers_qwen3_crosscheck_is_opt_in(self):
        self.assertIn("qwen3-st", aem.ENCODER_KEYS)
        self.assertNotIn("qwen3-st", aem.DEFAULT_ENCODER_KEYS)

    def test_unavailable_models_are_reported_with_an_install_hint(self):
        args = aem.parse_args(["--models", "moleculestm"])
        encoders, unavailable = aem.build_encoders(args)
        if not encoders:
            self.assertIn("moleculestm", unavailable)
            self.assertIn("install", unavailable["moleculestm"])

    def test_molt5_default_is_not_finetuned_on_chebi20(self):
        """A *-smiles2caption checkpoint would be scored on its own training split."""
        self.assertNotIn("smiles2caption", aem.parse_args([]).molt5_model)

    def test_rows_are_l2_normalized(self):
        rows = aem._l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1.0, 1.0])

    def test_normalizing_a_zero_row_does_not_divide_by_zero(self):
        rows = aem._l2_normalize(np.array([[0.0, 0.0]]))
        self.assertTrue(np.all(np.isfinite(rows)))


if __name__ == "__main__":
    unittest.main()
