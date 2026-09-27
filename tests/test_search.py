from __future__ import annotations

from dataclasses import asdict
import json
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from boreft.bo.acquisition import latent_ellipsoid
from boreft.bo.state import Observation, RunState
from boreft.interactive import maybe_append_decode_smiles_open_tag
from boreft.search_expand import SearchExpandConfig
from boreft.search import (
    BOConfig,
    MoleculeVerifier,
    PropertyOracleVerifier,
    SearchConfig,
    TextSimilarityVerifier,
    Verification,
    _load_file_warmstarts,
    _resolved_seeds,
    _safe_search_root,
    _validate_resume_config,
    checkpoint_decode,
    checkpoint_indices_for_labels,
    checkpoint_is_molopt_p90_1024,
    checkpoint_reconstruct_decode,
    molopt_p90_catalog_size,
    PINNED_MOLOPT_P90_1024,
    PINNED_MOLOPT_P90_1024_RELATIVE,
    PINNED_MOLOPT_P90_SIZES,
    pinned_molopt_p90_relative,
    checkpoint_warmstart_indices,
    found_target_stats,
    load_word_warmstarts,
    load_pinned_checkpoint_labels,
    memoize_point_decode,
    normalize_search_text,
    pinned_checkpoint_labels_file,
    resolve_file_warmstart_path,
    resolve_search_decode_prompt,
    resolve_search_instruction,
    run_bo,
    select_warmstarts,
    warmstart_result_fields,
    warmstart_verification_cost,
)


class _NumericVerifier:
    maximum = 1.0

    def __call__(self, decoded: str) -> Verification:
        score = float(decoded)
        return Verification(score=score, components={"numeric": score})


class _FakeSurrogate:
    def metadata(self):
        return {"kind": "static", "noise": 0.01}

    def checkpoint(self):
        return {"kind": "static"}


class VerifierTest(unittest.TestCase):
    @patch("boreft.search.embedding_sim_per_text")
    def test_text_similarity_and_exact_match(self, similarity):
        similarity.return_value = np.array([0.75])
        result = TextSimilarityVerifier("Apple")(" apple ")
        self.assertEqual(result.score, 0.75)
        self.assertTrue(result.components["exact_match"])
        similarity.assert_called_once_with(["apple"], ["apple"], task="semantle")

    @patch("boreft.search.embedding_sim_per_text")
    def test_text_similarity_preserves_hypogen_case(self, similarity):
        similarity.return_value = np.array([0.5])
        verifier = TextSimilarityVerifier("  The Cat  ", task="hypogen")
        result = verifier("The CAT")
        self.assertFalse(result.components["exact_match"])
        similarity.assert_called_once_with(
            ["The Cat"], ["The CAT"], task="hypogen"
        )

    @patch("boreft.search.rdkit_similarity", return_value=0.8)
    @patch("boreft.search.embedding_sim_per_text")
    def test_molecule_logs_all_components(self, similarity, rdkit):
        similarity.return_value = np.array([0.6])
        result = MoleculeVerifier(
            "CCO", objective="tfs", rdkit_map_path="custom-map.json"
        )("OCC")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.components["valid"])
        self.assertTrue(result.components["exact_match"])
        self.assertIn("rdkit_sim", result.components)
        self.assertFalse(result.components["repaired"])
        self.assertEqual(result.components["decoded"], "OCC")
        self.assertEqual(result.components["raw_decoded"], "OCC")
        rdkit.assert_called_once_with(
            "CCO", "OCC", map_path="custom-map.json"
        )

    def test_molecule_rejects_invalid_target(self):
        with self.assertRaisesRegex(ValueError, "valid SMILES"):
            MoleculeVerifier("not-smiles")

    def test_property_oracle_scores_and_skips_found_target(self):
        class FakeOracle:
            def __call__(self, smiles):
                return [1.2 if text == "CCO" else 0.4 for text in smiles]

        verifier = PropertyOracleVerifier("gsk3beta", score_fn=FakeOracle())
        self.assertIsNone(verifier.maximum)
        self.assertEqual(verifier.oracle_name, "GSK3B")
        hit = verifier("CCO [END_SMILES]")
        self.assertEqual(hit.score, 1.2)
        self.assertTrue(hit.components["valid"])
        self.assertEqual(hit.components["canonical"], "CCO")
        self.assertFalse(hit.components["exact_match"])
        self.assertIn("qed", hit.components)
        with patch("boreft.chem.repair_smiles", return_value=None):
            miss = verifier("not a molecule")
        self.assertEqual(miss.score, 0.0)
        self.assertFalse(miss.components["valid"])
        self.assertFalse(miss.components["repaired"])
        self.assertEqual(miss.components["decoded"], "not a molecule")
        self.assertEqual(miss.components["raw_decoded"], "not a molecule")
        found, index, used = found_target_stats(
            [
                Observation(
                    index=0,
                    point=[0.0],
                    decoded="CCO",
                    score=1.2,
                    sample_count=1,
                    sample_scores=[1.2],
                    decoded_samples=["CCO"],
                    source="acquisition",
                )
            ],
            "GSK3B",
            "molopt",
            property_search=True,
        )
        self.assertFalse(found)
        self.assertIsNone(index)
        self.assertIsNone(used)

    def test_property_oracle_scores_repaired_invalid_decode(self):
        class FakeOracle:
            def __call__(self, smiles):
                return [1.2 if text == "CCO" else 0.4 for text in smiles]

        verifier = PropertyOracleVerifier("DRD2", score_fn=FakeOracle())
        with patch("boreft.chem.repair_smiles", return_value="CCO"):
            result = verifier("not a molecule [END_SMILES]")
        self.assertEqual(result.score, 1.2)
        self.assertTrue(result.components["valid"])
        self.assertTrue(result.components["repaired"])
        self.assertEqual(result.components["decoded"], "CCO")
        self.assertEqual(result.components["raw_decoded"], "not a molecule")

    def test_dual_kinase_scores_product_and_records_factors(self):
        class Gsk:
            def __call__(self, smiles):
                return [0.5 if text == "CCO" else 0.25 for text in smiles]

        class Jnk:
            def __call__(self, smiles):
                return [0.4 if text == "CCO" else 0.5 for text in smiles]

        verifier = PropertyOracleVerifier(
            "jnk3*gsk3b",
            factor_oracles={"GSK3B": Gsk(), "JNK3": Jnk()},
        )
        self.assertEqual(verifier.oracle_name, "GSK3B_JNK3")
        hit = verifier("CCO [END_SMILES]")
        self.assertAlmostEqual(hit.score, 0.2)
        self.assertEqual(hit.components["oracle"], "GSK3B_JNK3")
        self.assertEqual(hit.components["oracle_GSK3B"], 0.5)
        self.assertEqual(hit.components["oracle_JNK3"], 0.4)
        with patch("boreft.chem.repair_smiles", return_value=None):
            miss = verifier("not a molecule")
        self.assertEqual(miss.score, 0.0)
        self.assertFalse(miss.components["valid"])
        self.assertEqual(miss.components["oracle_GSK3B"], 0.0)
        self.assertEqual(miss.components["oracle_JNK3"], 0.0)

    @patch("boreft.search.rdkit_similarity", return_value=0.0)
    @patch("boreft.search.embedding_sim_per_text")
    def test_molecule_verifier_scores_repaired_invalid_decode(self, similarity, rdkit):
        similarity.return_value = np.array([0.6])
        with patch("boreft.chem.repair_smiles", return_value="OCC"):
            result = MoleculeVerifier("CCO", objective="tfs")("not a molecule")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.components["valid"])
        self.assertTrue(result.components["exact_match"])
        self.assertTrue(result.components["repaired"])
        self.assertEqual(result.components["decoded"], "OCC")
        similarity.assert_called_once_with(["CCO"], ["OCC"], task="molopt")
        rdkit.assert_called_once_with("CCO", "OCC", map_path=None)

    def test_search_config_normalizes_oracle_alias(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="unused",
            oracle="GSK3β",
            warmstart_source="sobol",
            warmstart_count=2,
            budget=4,
        )
        config.validate()
        self.assertEqual(config.oracle, "GSK3B")

    def test_search_config_normalizes_dual_kinase_oracle(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="",
            oracle="jnk3_gsk3b",
            warmstart_source="sobol",
            warmstart_count=2,
            budget=4,
        )
        config.validate()
        self.assertEqual(config.oracle, "GSK3B_JNK3")
        self.assertEqual(config.target, "GSK3B_JNK3")

    def test_file_warmstart_directory_is_seed_keyed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seed_2.jsonl").write_text(
                '{"point":[1.0,2.0],"decoded":"CCO"}\n'
                '{"point":[3.0,4.0],"decoded":"c1ccccc1"}\n',
                encoding="utf-8",
            )
            resolved = resolve_file_warmstart_path(str(root), 2)
            self.assertEqual(Path(resolved).name, "seed_2.jsonl")
            points, records = _load_file_warmstarts(resolved, 2, 2)
            np.testing.assert_array_equal(points, [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(records[0]["decoded"], "CCO")
            with self.assertRaisesRegex(ValueError, "missing seed_1"):
                resolve_file_warmstart_path(str(root), 1)

    def test_file_warmstarts_unwrap_mist_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm.jsonl"
            path.write_text(
                '{"point":[1,2],"decoded":"CCO [END_SMILES]"}\n'
                '{"point":[3,4],"decoded":"[START_SMILES] c1ccccc1 [END_SMILES]"}\n',
                encoding="utf-8",
            )
            _points, records = _load_file_warmstarts(str(path), 2, 2)
            self.assertEqual(records[0]["decoded"], "CCO")
            self.assertEqual(records[1]["decoded"], "c1ccccc1")

    def test_normalize_search_text_unwraps_molopt_tags(self):
        self.assertEqual(
            normalize_search_text("CCO [END_SMILES]", task="molopt"),
            "CCO",
        )
        self.assertEqual(normalize_search_text("  The Cat  ", task="hypogen"), "The Cat")

    def test_decode_prompt_appends_mist_open_tag_last(self):
        mist = SimpleNamespace(
            saved_cfg={"mist_smiles_tags": True, "use_chat_template": False}
        )
        prompt = maybe_append_decode_smiles_open_tag(mist, "Generate a SMILES.")
        self.assertTrue(prompt.endswith("[START_SMILES]"))
        self.assertEqual(
            maybe_append_decode_smiles_open_tag(mist, prompt),
            prompt,
        )
        chat = SimpleNamespace(
            saved_cfg={"mist_smiles_tags": True, "use_chat_template": True}
        )
        self.assertEqual(
            maybe_append_decode_smiles_open_tag(chat, "Generate a SMILES."),
            "Generate a SMILES.",
        )
        off = SimpleNamespace(
            saved_cfg={"mist_smiles_tags": False, "use_chat_template": False}
        )
        self.assertEqual(
            maybe_append_decode_smiles_open_tag(off, "Generate a SMILES."),
            "Generate a SMILES.",
        )

    def test_found_target_stats_uses_sample_offset(self):
        observations = [
            Observation(
                index=0,
                point=[0.0],
                decoded="near",
                score=0.7,
                sample_count=2,
                sample_scores=[0.6, 0.8],
                decoded_samples=["berry", "near"],
                source="warmstart",
            ),
            Observation(
                index=1,
                point=[1.0],
                decoded="strawberry",
                score=0.78,
                sample_count=3,
                sample_scores=[0.7, 1.0, 0.65],
                decoded_samples=["raspberry", "strawberry", "parsley"],
                source="acquisition",
            ),
        ]
        found, index, used = found_target_stats(
            observations, "Strawberry", "semantle"
        )
        self.assertTrue(found)
        self.assertEqual(index, 1)
        self.assertEqual(used, 4)

    def test_found_target_stats_uses_stop_tolerance(self):
        observations = [
            Observation(
                index=0,
                point=[0.0],
                decoded="near",
                score=1.0 - 5e-7,
                sample_count=1,
                sample_scores=[1.0 - 5e-7],
                decoded_samples=["near"],
                source="acquisition",
            )
        ]
        found, index, used = found_target_stats(
            observations, "gold", "semantle"
        )
        self.assertTrue(found)
        self.assertEqual(index, 0)
        self.assertEqual(used, 1)


class SearchLoopTest(unittest.TestCase):
    def test_bbox_evaluation_time_is_recorded(self):
        class SlowVerifier:
            maximum = None

            def __call__(self, decoded):
                time.sleep(0.002)
                return Verification(score=float(decoded), components={})

        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [0.2]],
                decode=lambda point: str(float(point[0])),
                verifier=SlowVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            summary = state.summary()
            self.assertGreaterEqual(summary["bbox_eval_seconds"], 0.004)
            self.assertGreaterEqual(
                summary["elapsed_seconds"], summary["bbox_eval_seconds"]
            )
            loaded = RunState.load(Path(tmp) / "observations.jsonl")
            self.assertGreater(loaded.observations[0].bbox_eval_seconds, 0)

    def test_repeated_observations_store_mean_and_uncertainty(self):
        class UnboundedNumericVerifier:
            maximum = None

            def __call__(self, decoded):
                return Verification(score=float(decoded), components={})

        decoded = iter(["0.1", "0.3", "0.5", "0.2", "0.2", "0.2"])
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [0.2]],
                decode=lambda _point: next(decoded),
                verifier=UnboundedNumericVerifier(),
                state=state,
                config=BOConfig(budget=6, observation_samples=3),
            )
            first = state.observations[0]
            self.assertAlmostEqual(first.score, 0.3)
            self.assertAlmostEqual(first.score_std, 0.2)
            self.assertAlmostEqual(first.score_sem, 0.2 / np.sqrt(3))
            self.assertEqual(first.sample_count, 3)
            self.assertEqual(first.sample_scores, [0.1, 0.3, 0.5])
            self.assertAlmostEqual(first.peak_score(), 0.5)
            self.assertAlmostEqual(first.best_so_far, 0.5)
            self.assertAlmostEqual(state.best_score, 0.5)
            self.assertEqual(state.summary()["n_verifications"], 6)
            loaded = RunState.load(Path(tmp) / "observations.jsonl")
            self.assertAlmostEqual(loaded.observations[0].score_sem, first.score_sem)

    def test_repeated_observation_variance_is_passed_to_gp(self):
        calls = 0

        def decode(point):
            nonlocal calls
            offset = -0.1 if calls % 2 == 0 else 0.1
            calls += 1
            return str(float(point[0] + offset))

        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=decode,
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=6, observation_samples=2),
                )
            np.testing.assert_allclose(
                fit.call_args.kwargs["observation_variances"],
                [0.01, 0.01],
                atol=1e-8,
            )

    def test_sample_peak_updates_best_and_stops_search(self):
        decoded = iter(
            ["0.1", "0.2", "0.3", "0.2", "0.2", "0.2", "0.4", "1.0", "0.9"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ) as propose,
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda _point: next(decoded),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=9, observation_samples=3),
                )
            self.assertEqual(len(state.observations), 3)
            last = state.observations[-1]
            self.assertEqual(last.sample_count, 2)
            self.assertAlmostEqual(last.score, 0.7)
            self.assertEqual(last.sample_scores, [0.4, 1.0])
            self.assertAlmostEqual(last.peak_score(), 1.0)
            self.assertAlmostEqual(state.best_score, 1.0)
            self.assertEqual(state.summary()["n_verifications"], 8)
            self.assertEqual(state.summary()["best_decoded"], "1.0")
            np.testing.assert_allclose(fit.call_args.args[1], [0.2, 0.2])
            propose.assert_called_once()

    def test_exact_match_component_stops_remaining_samples(self):
        class ExactMatchVerifier:
            maximum = None

            def __call__(self, decoded):
                hit = decoded == "gold"
                return Verification(
                    score=0.4 if not hit else 0.9,
                    components={"exact_match": hit},
                )

        decoded = iter(["miss", "gold", "later"])
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [0.2]],
                decode=lambda _point: next(decoded),
                verifier=ExactMatchVerifier(),
                state=state,
                config=BOConfig(budget=4, observation_samples=2),
            )
            first = state.observations[0]
            self.assertEqual(first.decoded_samples, ["miss", "gold"])
            self.assertEqual(first.sample_count, 2)
            self.assertAlmostEqual(first.score, 0.65)
            self.assertAlmostEqual(state.best_score, 0.9)
            self.assertEqual(len(state.observations), 1)

    def test_repeat_sample_sees_non_representative_decodes(self):
        decoded = iter(["0.1", "0.3", "0.2", "0.25", "0.1", "0.05"])
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ),
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda _point: next(decoded),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=6, observation_samples=2),
                )
            last = state.observations[-1]
            self.assertEqual(last.source, "acquisition")
            self.assertEqual(last.decoded, "0.1")
            self.assertTrue(last.is_repeat_sample)

    def test_use_ard_is_passed_to_surrogate_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=3, use_ard=True),
                )
            self.assertTrue(fit.call_args.args[3].use_ard)

    def test_kernel_is_passed_to_surrogate_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=3, kernel="rbf"),
                )
            self.assertEqual(fit.call_args.args[3].kernel, "rbf")

    def test_projection_layers_are_passed_to_surrogate_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(
                        budget=3, surrogate="projected", projection_layers=2
                    ),
                )
            self.assertEqual(fit.call_args.args[3].projection_layers, 2)

    def test_rejects_nonfinite_verifier_score(self):
        class NonfiniteVerifier:
            maximum = None

            def __call__(self, _decoded):
                return Verification(score=float("nan"), components={})

        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=NonfiniteVerifier(),
                    state=state,
                    config=BOConfig(budget=2),
                )

    def test_rejects_precomputed_score_above_known_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with self.assertRaisesRegex(ValueError, "exceeds maximum"):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    warmstart_records=[{"score": 1.1}, {"score": 0.2}],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=2),
                )

    def test_budget_and_best_so_far(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch("boreft.bo.runner.fit_surrogate", return_value=_FakeSurrogate()),
                patch(
                    "boreft.bo.runner.propose_candidates",
                    side_effect=[
                        np.array([[0.6, 0.0]]),
                        np.array([[0.8, 0.0]]),
                    ],
                ),
            ):
                run_bo(
                    seed=3,
                    bounds=np.array([[0.0, 0.0], [1.0, 1.0]]),
                    warmstart_points=[[0.1, 0.0], [0.2, 0.0]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=4),
                )
            self.assertEqual(len(state.observations), 4)
            self.assertAlmostEqual(state.best_score, 0.8)
            self.assertEqual(state.summary()["n_acquired"], 2)

    def test_budget_counts_verifications_not_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ) as propose,
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=9, observation_samples=3),
                )
            fit.assert_called_once()
            self.assertEqual(propose.call_args.kwargs["batch_size"], 1)
            self.assertEqual(len(state.observations), 3)
            self.assertEqual(state.summary()["n_acquired"], 1)
            self.assertEqual(state.summary()["n_verifications"], 9)

    def test_warmstart_does_not_exceed_verification_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(
                Observation(
                    index=0,
                    point=[0.1],
                    decoded="0.1",
                    score=0.1,
                    sample_count=4,
                    source="warmstart",
                    seed=1,
                )
            )
            with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=6, observation_samples=3),
                )
            self.assertEqual(len(state.observations), 1)
            self.assertEqual(state.summary()["n_verifications"], 4)

    def test_precomputed_warmstarts_charge_recorded_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ),
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ) as propose,
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    warmstart_records=[
                        {"score": 0.2, "decoded": "a", "sample_count": 1},
                        {"score": 0.3, "decoded": "b", "sample_count": 1},
                    ],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=6, observation_samples=3),
                )
            self.assertEqual(propose.call_args.kwargs["batch_size"], 1)
            self.assertEqual(state.summary()["n_warmstart"], 2)
            self.assertEqual(state.summary()["n_acquired"], 1)
            self.assertEqual(state.summary()["n_verifications"], 5)

    def test_leftover_verifications_do_not_start_a_partial_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with patch("boreft.bo.runner.fit_surrogate") as fit:
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=7, observation_samples=3),
                )
            fit.assert_not_called()
            self.assertEqual(len(state.observations), 2)
            self.assertEqual(state.summary()["n_verifications"], 6)

    def test_unaffordable_pending_batch_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            for index, value in enumerate((0.1, 0.2)):
                state.append(
                    Observation(
                        index=index,
                        point=[value],
                        decoded=str(value),
                        score=value,
                        sample_count=3,
                        source="warmstart",
                        seed=1,
                    )
                )
            (Path(tmp) / "pending_batch.json").write_text(
                json.dumps(
                    {
                        "start_index": 2,
                        "batch_index": 0,
                        "acquisition": "log_ei",
                        "points": [[0.7]],
                        "surrogate_metadata": {"kind": "static"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("boreft.bo.runner.fit_surrogate") as fit:
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=7, observation_samples=3),
                )
            fit.assert_not_called()
            self.assertEqual(len(state.observations), 2)
            self.assertEqual(state.summary()["n_verifications"], 6)
            self.assertFalse((Path(tmp) / "pending_batch.json").exists())

    def test_pending_batch_observes_only_affordable_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            for index, value in enumerate((0.1, 0.2)):
                state.append(
                    Observation(
                        index=index,
                        point=[value],
                        decoded=str(value),
                        score=value,
                        sample_count=3,
                        source="warmstart",
                        seed=1,
                    )
                )
            (Path(tmp) / "pending_batch.json").write_text(
                json.dumps(
                    {
                        "start_index": 2,
                        "batch_index": 0,
                        "acquisition": "log_ei",
                        "points": [[0.7], [0.8]],
                        "surrogate_metadata": {"kind": "static"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("boreft.bo.runner.fit_surrogate") as fit:
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=9, observation_samples=3),
                )
            fit.assert_not_called()
            self.assertEqual(len(state.observations), 3)
            self.assertAlmostEqual(state.observations[-1].score, 0.7)
            self.assertEqual(state.summary()["n_verifications"], 9)
            self.assertFalse((Path(tmp) / "pending_batch.json").exists())

    def test_joint_batch_is_observed_without_refitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6], [0.8]]),
                ) as propose,
            ):
                run_bo(
                    seed=3,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=4, batch_size=2, acquisition="ucb"),
                )
            fit.assert_called_once()
            self.assertEqual(propose.call_args.kwargs["batch_size"], 2)
            self.assertEqual(propose.call_args.kwargs["acquisition"], "ucb")
            self.assertEqual(
                [o.components["bo_batch"] for o in state.observations[2:]],
                [0, 0],
            )
            self.assertFalse((Path(tmp) / "pending_batch.json").exists())

    def test_gp_surrogate_checkpoints_are_opt_in(self):
        def run(tmp: str, **kwargs):
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ),
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=3, **kwargs),
                )
            return list(Path(tmp).glob("surrogate_*.pt"))

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(tmp), [])
        with tempfile.TemporaryDirectory() as tmp:
            paths = run(tmp, log_gp_surrogate=True)
            self.assertEqual([path.name for path in paths], ["surrogate_0002.pt"])

    def test_resume_finishes_persisted_pending_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = RunState(run_dir / "observations.jsonl")
            run_bo(
                seed=3,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [0.2]],
                decode=lambda point: str(float(point[0])),
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            (run_dir / "pending_batch.json").write_text(
                json.dumps(
                    {
                        "start_index": 2,
                        "batch_index": 0,
                        "acquisition": "log_ei",
                        "points": [[0.7]],
                        "surrogate_metadata": {"kind": "static"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("boreft.bo.runner.fit_surrogate") as fit:
                run_bo(
                    seed=3,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=3),
                )
            fit.assert_not_called()
            self.assertAlmostEqual(state.best_score, 0.7)
            self.assertFalse((run_dir / "pending_batch.json").exists())

    def test_known_maximum_stops_during_warmstart(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [1.0]],
                decode=lambda point: str(float(point[0])),
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=5),
            )
            self.assertEqual(len(state.observations), 2)
            self.assertEqual(state.best_score, 1.0)

    def test_seed_summary_counts_repeat_samples_and_proposals(self):
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch(
                    "boreft.bo.runner._log",
                    side_effect=lambda message="": logs.append(message),
                ),
                patch(
                    "boreft.bo.runner.fit_surrogate",
                    return_value=_FakeSurrogate(),
                ),
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.1], [0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=_NumericVerifier(),
                    state=state,
                    config=BOConfig(budget=4, batch_size=2),
                )
        joined = "\n".join(logs)
        self.assertIn("[repeat sample] [repeat proposal]", joined)
        self.assertIn("repeat samples=1, repeat proposals=1", joined)
        self.assertEqual(state.summary()["n_repeat_samples"], 1)
        self.assertEqual(state.summary()["n_repeat_proposals"], 1)

    def test_warmstart_seed_text_does_not_replace_point_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [1.0]],
                warmstart_records=[
                    {"components": {"llm_seed_text": "base-a"}},
                    {"components": {"llm_seed_text": "base-b"}},
                ],
                decode=lambda point: str(float(point[0])),
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            self.assertAlmostEqual(float(state.observations[0].decoded), 0.1)
            self.assertEqual(
                state.observations[0].components["llm_seed_text"], "base-a"
            )

    def test_labeled_warmstart_scores_known_text_without_decode(self):
        def boom(_point):
            raise AssertionError("labeled warm starts must not decode the vector")

        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [0.2]],
                warmstart_records=[
                    {"decoded": "0.7", "sample_count": 1},
                    {"decoded": "0.4", "sample_count": 1},
                ],
                decode=boom,
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            self.assertEqual(state.observations[0].decoded, "0.7")
            self.assertAlmostEqual(state.observations[0].score, 0.7)
            self.assertEqual(state.observations[0].sample_count, 1)
            self.assertEqual(state.summary()["n_verifications"], 2)
            self.assertEqual(state.summary()["n_warmstart"], 2)

    def test_blank_decoded_warmstart_falls_back_to_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.25], [0.5]],
                warmstart_records=[{"decoded": "  "}, {"decoded": ""}],
                decode=lambda point: str(float(point[0])),
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            self.assertEqual(len(state.observations), 2)
            self.assertAlmostEqual(state.observations[0].score, 0.25, places=5)
            self.assertAlmostEqual(state.observations[1].score, 0.5, places=5)

    def test_resume_does_not_reverify_completed_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            calls = 0

            def decode(point):
                nonlocal calls
                calls += 1
                return str(float(point[0]))

            run_bo(
                seed=2,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [1.0]],
                decode=decode,
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=2),
            )
            self.assertEqual(calls, 2)
            loaded = RunState.load(path)
            run_bo(
                seed=2,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[0.1], [1.0]],
                decode=decode,
                verifier=_NumericVerifier(),
                state=loaded,
                config=BOConfig(budget=2),
            )
            self.assertEqual(calls, 2)

    def test_resume_at_known_maximum_does_not_add_warmstart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            state.append(
                Observation(
                    index=0,
                    point=[1.0],
                    decoded="1.0",
                    score=1.0,
                    source="warmstart",
                    seed=1,
                )
            )
            calls = 0

            def decode(_point):
                nonlocal calls
                calls += 1
                return "0.2"

            run_bo(
                seed=1,
                bounds=np.array([[0.0], [1.0]]),
                warmstart_points=[[1.0], [0.2]],
                decode=decode,
                verifier=_NumericVerifier(),
                state=state,
                config=BOConfig(budget=3),
            )
            self.assertEqual(calls, 0)
            self.assertEqual(len(state.observations), 1)


class WarmstartAndSeedTest(unittest.TestCase):
    def test_overwrite_wins_over_resume(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            resume=True,
            overwrite=True,
        )
        config.validate()
        self.assertTrue(config.overwrite)
        self.assertFalse(config.resume)

    def test_budget_must_cover_warmstart_verifications(self):
        self.assertEqual(warmstart_verification_cost("checkpoint", 10, 5), 10)
        self.assertEqual(warmstart_verification_cost("words", 10, 5), 10)
        self.assertEqual(warmstart_verification_cost("sobol", 10, 5), 50)
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            budget=2,
            warmstart_count=2,
            observation_samples=5,
            warmstart_source="checkpoint",
        ).validate()
        with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                budget=1,
                warmstart_count=2,
                warmstart_source="checkpoint",
            ).validate()
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            budget=6,
            warmstart_count=2,
            observation_samples=3,
            warmstart_source="sobol",
        ).validate()
        with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                budget=5,
                warmstart_count=2,
                observation_samples=3,
                warmstart_source="sobol",
            ).validate()

    def test_projection_layers_must_be_positive(self):
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            projection_layers=1,
        ).validate()
        with self.assertRaisesRegex(ValueError, "projection_layers"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                projection_layers=0,
            ).validate()

    def test_aabb_std_k_must_be_nonnegative(self):
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            aabb_std_k=0.0,
        ).validate()
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            aabb_std_k=2.0,
        ).validate()
        with self.assertRaisesRegex(ValueError, "aabb_std_k"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                aabb_std_k=-0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "aabb_std_k"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                aabb_std_k=float("nan"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "aabb_std_k"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                aabb_std_k=float("inf"),
            ).validate()

    def test_search_domain_must_be_aabb_or_ellipsoid(self):
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            search_domain="aabb",
        ).validate()
        SearchConfig(
            output_dir="checkpoint",
            target="target",
            search_domain="ellipsoid",
        ).validate()
        with self.assertRaisesRegex(ValueError, "search_domain"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                search_domain="sphere",
            ).validate()

    def test_sampling_temperature_zero_is_greedy(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            sampling_temperature=0.0,
        )
        config.validate()
        with self.assertRaisesRegex(ValueError, "sampling_temperature"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                sampling_temperature=-0.1,
            ).validate()

    @patch("boreft.search.generate_text", return_value="Decoded")
    def test_checkpoint_decode_uses_temperature_without_nucleus(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        decode = checkpoint_decode(ckpt, 16, sampling_temperature=1.5)
        self.assertEqual(decode(np.array([0.1])), "decoded")
        self.assertTrue(generate.call_args.kwargs["use_sample"])
        self.assertEqual(generate.call_args.kwargs["temperature"], 1.5)
        self.assertEqual(generate.call_args.kwargs["top_p"], 1.0)

        greedy = checkpoint_decode(ckpt, 16, sampling_temperature=0.0)
        greedy(np.array([0.2]))
        self.assertFalse(generate.call_args.kwargs["use_sample"])

    @patch("boreft.search.generate_text", return_value="CCO")
    def test_checkpoint_decode_uses_override_prompt(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="checkpoint prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        decode = checkpoint_decode(
            ckpt,
            16,
            task="molopt",
            prompt="task prompt",
            from_chat_template=False,
            content_span=None,
        )
        with patch("boreft.chem.repair_smiles", return_value=None):
            self.assertEqual(decode(np.array([0.1])), "CCO")
        self.assertEqual(generate.call_args.args[2], "task prompt")

    @patch("boreft.search.generate_text", return_value="  Barn  ")
    def test_checkpoint_decode_lowercases_semantle_only(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        semantle = checkpoint_decode(ckpt, 16, task="semantle")
        hypogen = checkpoint_decode(ckpt, 16, task="hypogen")
        molopt = checkpoint_decode(ckpt, 16, task="molopt")
        self.assertEqual(semantle(np.array([0.1])), "barn")
        self.assertEqual(hypogen(np.array([0.1])), "  Barn  ")
        with patch("boreft.chem.repair_smiles", return_value=None):
            self.assertEqual(molopt(np.array([0.1])), "Barn")

    @patch("boreft.search.generate_text", return_value="CCO [END_SMILES]")
    def test_checkpoint_decode_unwraps_molopt_tags(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        molopt = checkpoint_decode(ckpt, 16, task="molopt")
        self.assertEqual(molopt(np.array([0.1])), "CCO")

    @patch("boreft.search.generate_text", return_value="not a molecule [END_SMILES]")
    def test_checkpoint_decode_repairs_invalid_molopt(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        molopt = checkpoint_decode(ckpt, 16, task="molopt")
        with patch("boreft.chem.repair_smiles", return_value="CCO"):
            self.assertEqual(molopt(np.array([0.1])), "CCO")

    @patch("boreft.search.generate_text", return_value="not a molecule [END_SMILES]")
    def test_checkpoint_reconstruct_decode_does_not_repair(self, generate):
        ckpt = SimpleNamespace(
            reft_model=object(),
            tokenizer=object(),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
            saved_cfg={"position": "l1"},
        )
        decode = checkpoint_reconstruct_decode(ckpt, 16, task="molopt")
        with patch("boreft.chem.repair_smiles", return_value="CCO") as repair:
            self.assertEqual(decode(np.array([0.1])), "not a molecule")
        repair.assert_not_called()
        self.assertFalse(generate.call_args.kwargs["use_sample"])

    def test_memoize_point_decode_reuses_identical_mu(self):
        calls = []

        def decode(point):
            calls.append(np.asarray(point).copy())
            return "hit"

        cached = memoize_point_decode(decode)
        self.assertEqual(cached(np.array([0.1, 0.2], dtype=np.float32)), "hit")
        self.assertEqual(cached(np.array([0.1, 0.2], dtype=np.float32)), "hit")
        self.assertEqual(cached(np.array([0.3], dtype=np.float32)), "hit")
        self.assertEqual(len(calls), 2)

    def test_checkpoint_warmstarts_exclude_target_and_use_labels(self):
        words = ["Alpha", "target", "beta", "gamma", "alpha", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "semantle"})
        config = SearchConfig(
            output_dir="checkpoint",
            target="Target",
            warmstart_source="checkpoint",
            warmstart_count=3,
        )
        points, records = select_warmstarts(
            config, ckpt, train_mu, np.array([[0.0], [5.0]]), seed=7
        )
        again, again_records = select_warmstarts(
            config, ckpt, train_mu, np.array([[0.0], [5.0]]), seed=7
        )
        labels = [record["decoded"] for record in records]
        self.assertEqual(labels, [record["decoded"] for record in again_records])
        np.testing.assert_array_equal(points, again)
        self.assertNotIn("target", labels)
        self.assertEqual(len(set(labels)), 3)
        for record, point in zip(records, points):
            self.assertTrue(record["components"]["warmstart_labeled"])
            self.assertEqual(record["sample_count"], 1)
            index = int(round(float(point[0])))
            self.assertEqual(record["decoded"], words[index].lower())
        eligible = checkpoint_warmstart_indices(
            words, target="Target", count=3, seed=7, task="semantle"
        )
        self.assertNotIn(1, eligible)
        self.assertNotIn(4, eligible)

    def test_sobol_warmstarts_sample_inside_ellipsoid(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        ell = latent_ellipsoid(mu)
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            warmstart_source="sobol",
            warmstart_count=8,
            budget=8,
            search_domain="ellipsoid",
        )
        points, records = select_warmstarts(
            config,
            None,
            mu,
            ell.aabb(),
            seed=3,
            ellipsoid=ell,
        )
        self.assertIsNone(records)
        self.assertEqual(points.shape, (8, 2))
        self.assertTrue(np.all(ell.contains(points, atol=1e-5)))

    def test_checkpoint_warmstarts_reject_short_labeled_pool(self):
        ckpt = SimpleNamespace(
            words=["only", "target"],
            saved_cfg={"task": "semantle"},
        )
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            warmstart_source="checkpoint",
            warmstart_count=2,
        )
        with self.assertRaisesRegex(ValueError, "excluding the target"):
            select_warmstarts(
                config,
                ckpt,
                np.array([[0.0], [1.0]]),
                np.array([[0.0], [1.0]]),
                seed=1,
            )

    def test_checkpoint_warmstarts_exclude_molopt_canonical_duplicates(self):
        words = ["CCO", "CCC", "C(O)C", "CC", "CCCC"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})
        config = SearchConfig(
            output_dir="checkpoint",
            target="OCC",
            warmstart_source="checkpoint",
            warmstart_count=2,
        )
        _, records = select_warmstarts(
            config, ckpt, train_mu, np.array([[0.0], [5.0]]), seed=1
        )
        labels = {record["decoded"] for record in records}
        self.assertNotIn("CCO", labels)
        self.assertNotIn("C(O)C", labels)
        eligible = checkpoint_warmstart_indices(
            words, target="OCC", count=2, seed=1, task="molopt"
        )
        self.assertNotIn(0, eligible)
        self.assertNotIn(2, eligible)

    def test_property_search_checkpoint_warmstarts_match_across_oracles(self):
        words = ["CCO", "CCC", "CCCC", "CC", "C"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})
        labels = []
        for target in ("DRD2", "GSK3B", "JNK3"):
            config = SearchConfig(
                output_dir="checkpoint",
                target=target,
                oracle=target,
                warmstart_source="checkpoint",
                warmstart_count=3,
            )
            points, records = select_warmstarts(
                config, ckpt, train_mu, np.array([[0.0], [5.0]]), seed=4
            )
            labels.append([record["decoded"] for record in records])
            self.assertEqual(len(points), 3)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(len(set(labels[0])), 3)

    def test_checkpoint_warmstart_indices_replace_rejected_rows(self):
        words = ["alpha", "target", "beta", "gamma", "delta", "epsilon"]
        rejected = {2, 3}

        def accept(index: int) -> bool:
            return index not in rejected

        indices = checkpoint_warmstart_indices(
            words,
            target="Target",
            count=3,
            seed=11,
            task="semantle",
            accept=accept,
        )
        again = checkpoint_warmstart_indices(
            words,
            target="Target",
            count=3,
            seed=11,
            task="semantle",
            accept=accept,
        )
        self.assertEqual(indices, again)
        self.assertEqual(len(indices), 3)
        self.assertTrue(all(index not in rejected for index in indices))
        self.assertNotIn(1, indices)

    def test_checkpoint_warmstart_indices_return_short_list_when_pool_exhausted(self):
        words = ["alpha", "beta", "gamma", "delta"]
        indices = checkpoint_warmstart_indices(
            words,
            target="omega",
            count=3,
            seed=0,
            task="semantle",
            accept=lambda index: index == 0,
        )
        self.assertEqual(indices, [0])

    def test_checkpoint_warmstarts_fill_noisy_when_reconstructing_pool_runs_out(self):
        words = ["alpha", "beta", "gamma", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1) * 10
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "semantle"})
        config = SearchConfig(
            output_dir="checkpoint",
            target="omega",
            warmstart_source="checkpoint",
            warmstart_count=3,
        )

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0]) / 10.0))
            if 0 <= index < len(words) and abs(float(point[0]) - 10.0 * index) < 1e-5:
                return words[index] if index == 0 else "mismatch"
            return "noisy-decode"

        points, records = select_warmstarts(
            config,
            ckpt,
            train_mu,
            np.array([[-50.0], [50.0]]),
            seed=0,
            decode=decode,
        )
        self.assertEqual(len(points), 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["decoded"], "alpha")
        self.assertTrue(records[0]["components"]["warmstart_reconstructed"])
        self.assertTrue(records[1]["components"]["warmstart_noisy"])
        self.assertTrue(records[2]["components"]["warmstart_noisy"])
        self.assertEqual(records[1]["decoded"], "noisy-decode")
        self.assertEqual(records[2]["decoded"], "noisy-decode")
        np.testing.assert_allclose(points[0], [0.0])
        self.assertFalse(np.allclose(points[1], points[0]))
        again, again_records = select_warmstarts(
            config,
            ckpt,
            train_mu,
            np.array([[-50.0], [50.0]]),
            seed=0,
            decode=decode,
        )
        np.testing.assert_allclose(again, points)
        self.assertEqual(
            [record["decoded"] for record in again_records],
            [record["decoded"] for record in records],
        )
        fields = warmstart_result_fields(records)
        self.assertEqual(fields["n_warmstart_noisy"], 2)
        self.assertEqual(fields["n_warmstart_reconstruct_hits"], 1)

    def test_checkpoint_warmstarts_skip_mu_that_do_not_reconstruct(self):
        words = ["alpha", "target", "beta", "gamma", "delta", "epsilon"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "semantle"})
        config = SearchConfig(
            output_dir="checkpoint",
            target="Target",
            warmstart_source="checkpoint",
            warmstart_count=3,
        )

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0])))
            if words[index] in {"beta", "gamma"}:
                return "mismatch"
            return words[index]

        points, records = select_warmstarts(
            config,
            ckpt,
            train_mu,
            np.array([[0.0], [6.0]]),
            seed=11,
            decode=decode,
        )
        labels = [record["decoded"] for record in records]
        self.assertEqual(len(labels), 3)
        self.assertNotIn("beta", labels)
        self.assertNotIn("gamma", labels)
        self.assertNotIn("target", labels)
        for record, point in zip(records, points):
            self.assertTrue(record["components"]["warmstart_reconstructed"])
            index = int(round(float(point[0])))
            self.assertEqual(record["decoded"], words[index])

    def test_checkpoint_warmstarts_accept_canonical_molopt_greedy_match(self):
        words = ["CCO", "CCC", "CCCC", "CC", "C"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})
        config = SearchConfig(
            output_dir="checkpoint",
            target="DRD2",
            oracle="DRD2",
            warmstart_source="checkpoint",
            warmstart_count=4,
        )

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0])))
            gold = words[index]
            if gold == "CCO":
                return "OCC"
            if gold == "CCC":
                return "not a molecule"
            return gold

        _, records = select_warmstarts(
            config,
            ckpt,
            train_mu,
            np.array([[0.0], [5.0]]),
            seed=4,
            decode=decode,
        )
        labels = {record["decoded"] for record in records}
        self.assertEqual(labels, {"CCO", "CCCC", "CC", "C"})

    def test_pinned_checkpoint_labels_are_looked_up_not_resampled(self):
        words = ["CCO", "CCC", "CCCC", "CC", "C"]
        train_mu = np.array(
            [[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=np.float32
        )
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p90_1024.json"
            path.write_text(
                json.dumps({"catalog": "molopt_p90_1024", "seeds": {"1": ["OCC", "CC"]}}),
                encoding="utf-8",
            )
            labels = load_pinned_checkpoint_labels(
                str(path), seed=1, count=2, target="DRD2", task="molopt"
            )
            self.assertEqual(labels, ["OCC", "CC"])
            self.assertEqual(
                checkpoint_indices_for_labels(
                    words, labels, task="molopt", target="DRD2"
                ),
                [0, 3],
            )
            config = SearchConfig(
                output_dir="checkpoint",
                target="DRD2",
                oracle="DRD2",
                warmstart_source="checkpoint",
                warmstart_file=str(path),
                warmstart_count=2,
            )
            points, records = select_warmstarts(
                config, ckpt, train_mu, np.array([[0.0], [50.0]]), seed=1
            )
            np.testing.assert_allclose(points[:, 0], [10.0, 40.0])
            self.assertEqual(
                [record["decoded"] for record in records], ["CCO", "CC"]
            )
            self.assertTrue(records[0]["components"]["warmstart_pinned"])
            other = SearchConfig(
                output_dir="checkpoint",
                target="GSK3B",
                oracle="GSK3B",
                warmstart_source="checkpoint",
                warmstart_file=str(path),
                warmstart_count=2,
            )
            _, gsk = select_warmstarts(
                other, ckpt, train_mu, np.array([[0.0], [50.0]]), seed=1
            )
            self.assertEqual(
                [record["decoded"] for record in gsk],
                [record["decoded"] for record in records],
            )

    def test_pinned_checkpoint_warmstarts_keep_means_on_reconstruct_miss(self):
        words = ["alpha", "beta", "gamma", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "semantle"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.json"
            path.write_text(
                json.dumps({"seeds": {"3": ["beta", "gamma"]}}),
                encoding="utf-8",
            )
            config = SearchConfig(
                output_dir="checkpoint",
                target="omega",
                warmstart_source="checkpoint",
                warmstart_file=str(path),
                warmstart_count=2,
            )

            def decode(point):
                index = int(round(float(np.asarray(point).reshape(-1)[0])))
                if words[index] == "beta":
                    return "mismatch"
                return words[index]

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                points, records = select_warmstarts(
                    config,
                    ckpt,
                    train_mu,
                    np.array([[0.0], [4.0]]),
                    seed=3,
                    decode=decode,
                )
            self.assertTrue(any("did not greedily reconstruct" in str(item.message) for item in caught))
            np.testing.assert_allclose(points[:, 0], [1.0, 2.0])
            self.assertEqual(
                [record["decoded"] for record in records], ["beta", "gamma"]
            )
            self.assertFalse(records[0]["components"]["warmstart_reconstructed"])
            self.assertEqual(records[0]["components"]["warmstart_greedy"], "mismatch")
            self.assertTrue(records[1]["components"]["warmstart_reconstructed"])
            fields = warmstart_result_fields(records)
            self.assertEqual(fields["n_warmstart_reconstruct_misses"], 1)
            self.assertEqual(fields["warmstart_reconstruct_misses"][0]["label"], "beta")
            missing = Path(tmp) / "missing.json"
            missing.write_text(
                json.dumps({"seeds": {"3": ["beta", "absent"]}}),
                encoding="utf-8",
            )
            config.warmstart_file = str(missing)
            with self.assertRaisesRegex(ValueError, "not in this checkpoint"):
                checkpoint_indices_for_labels(
                    words,
                    load_pinned_checkpoint_labels(
                        str(missing), seed=3, count=2, target="omega", task="semantle"
                    ),
                    task="semantle",
                    target="omega",
                )
            ckpt.reft_model = object()
            with patch(
                "boreft.search.predict_bias_vectors_for_words",
                return_value=[np.array([99.0], dtype=np.float32)],
            ) as predict:
                points, records = select_warmstarts(
                    config, ckpt, train_mu, np.array([[0.0], [4.0]]), seed=3
                )
            predict.assert_called_once()
            np.testing.assert_allclose(points[:, 0], [1.0, 4.0])
            self.assertEqual(
                [record["decoded"] for record in records], ["beta", "absent"]
            )
            self.assertTrue(records[1]["components"]["warmstart_predicted"])
            self.assertNotIn("warmstart_train_index", records[1]["components"])
            self.assertEqual(warmstart_result_fields(records)["n_warmstart_predicted"], 1)

    def test_molopt_p90_catalogs_use_repo_pin_files(self):
        self.assertTrue(PINNED_MOLOPT_P90_1024.is_file(), PINNED_MOLOPT_P90_1024)
        self.assertEqual(PINNED_MOLOPT_P90_SIZES, (1024, 2048, 3072))
        for n in PINNED_MOLOPT_P90_SIZES:
            pin = Path(pinned_molopt_p90_relative(n))
            self.assertTrue(pin.is_file(), pin)
            for seed in (1, 2, 3, 4, 5):
                labels = load_pinned_checkpoint_labels(
                    str(pin), seed=seed, count=10, target="DRD2", task="molopt"
                )
                self.assertEqual(len(labels), 10)
                self.assertEqual(len(set(labels)), 10)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "task": "molopt",
                "molopt_oracle_cap_percentile": 90,
                "num_training_examples": 1024,
            }
            Path(tmp, "training_config.json").write_text(json.dumps(cfg), encoding="utf-8")
            self.assertTrue(checkpoint_is_molopt_p90_1024(tmp))
            self.assertEqual(molopt_p90_catalog_size(tmp), 1024)
            self.assertEqual(
                pinned_checkpoint_labels_file(tmp), PINNED_MOLOPT_P90_1024_RELATIVE
            )
            config = SearchConfig(
                output_dir=tmp,
                target="DRD2",
                oracle="DRD2",
                warmstart_source="checkpoint",
                warmstart_count=10,
            )
            config.validate()
            self.assertEqual(config.warmstart_file, PINNED_MOLOPT_P90_1024_RELATIVE)
            Path(tmp, "training_config.json").write_text(
                json.dumps({**cfg, "num_training_examples": 2048}),
                encoding="utf-8",
            )
            self.assertFalse(checkpoint_is_molopt_p90_1024(tmp))
            self.assertEqual(molopt_p90_catalog_size(tmp), 2048)
            self.assertEqual(
                pinned_checkpoint_labels_file(tmp),
                pinned_molopt_p90_relative(2048),
            )
            Path(tmp, "training_config.json").write_text(
                json.dumps({**cfg, "num_training_examples": 3072}),
                encoding="utf-8",
            )
            self.assertEqual(molopt_p90_catalog_size(tmp), 3072)
            self.assertEqual(
                pinned_checkpoint_labels_file(tmp),
                pinned_molopt_p90_relative(3072),
            )
            Path(tmp, "training_config.json").write_text(
                json.dumps(
                    {
                        **cfg,
                        "train_n_samples": 1024,
                        "num_training_examples": 512,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(checkpoint_is_molopt_p90_1024(tmp))
            self.assertIsNone(molopt_p90_catalog_size(tmp))
            self.assertIsNone(pinned_checkpoint_labels_file(tmp))
            missing = SearchConfig(
                output_dir="checkpoint",
                target="DRD2",
                oracle="DRD2",
                warmstart_source="checkpoint",
                warmstart_file=str(Path(tmp) / "absent.json"),
                warmstart_count=2,
            )
            ckpt = SimpleNamespace(words=["CCO", "CC"], saved_cfg={"task": "molopt"})
            with self.assertRaisesRegex(ValueError, "does not exist"):
                select_warmstarts(
                    missing,
                    ckpt,
                    np.array([[0.0], [1.0]]),
                    np.array([[0.0], [1.0]]),
                    seed=1,
                )

    def test_search_prompt_task_uses_opro_header(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="DRD2",
            oracle="DRD2",
            search_prompt="task",
        )
        config.validate()
        text = resolve_search_instruction(config, task="molopt")
        self.assertIn("DRD2", text or "")
        self.assertIn("valid SMILES", text or "")
        self.assertIsNone(
            resolve_search_instruction(
                SearchConfig(output_dir="checkpoint", target="DRD2"),
                task="molopt",
            )
        )

    def test_search_prompt_task_requires_oracle_or_description(self):
        with self.assertRaisesRegex(ValueError, "search_prompt=task"):
            SearchConfig(
                output_dir="checkpoint",
                target="word",
                search_prompt="task",
            ).validate()
        config = SearchConfig(
            output_dir="checkpoint",
            target="word",
            search_prompt="task",
            task_description="Generate a molecule.",
        )
        config.validate()
        self.assertEqual(
            resolve_search_instruction(config, task="molopt"),
            "Generate a molecule.",
        )
        with self.assertRaisesRegex(ValueError, "only valid for molopt"):
            resolve_search_instruction(
                SearchConfig(
                    output_dir="checkpoint",
                    target="DRD2",
                    oracle="DRD2",
                    search_prompt="task",
                ),
                task="semantle",
            )

    def test_resolve_search_decode_prompt_rebuilds_opro_header(self):
        ckpt = SimpleNamespace(
            prompt="checkpoint prompt",
            from_chat_template=False,
            content_span=(0, 1),
            tokenizer=None,
            intervention_token_id=None,
            saved_cfg={
                "task": "molopt",
                "position": "l1",
                "mist_smiles_tags": True,
                "use_chat_template": False,
                "intervention_inject": "none",
            },
        )
        kept = resolve_search_decode_prompt(
            ckpt,
            SearchConfig(output_dir="checkpoint", target="DRD2"),
            task="molopt",
            model_name="dummy",
        )
        self.assertEqual(kept, ("checkpoint prompt", False, (0, 1)))

        prompt, from_chat, span = resolve_search_decode_prompt(
            ckpt,
            SearchConfig(
                output_dir="checkpoint",
                target="DRD2",
                oracle="DRD2",
                search_prompt="task",
            ),
            task="molopt",
            model_name="dummy",
        )
        self.assertNotEqual(prompt, ckpt.prompt)
        self.assertIn("DRD2", prompt)
        self.assertIn("[START_SMILES]", prompt)
        self.assertFalse(from_chat)
        self.assertEqual(span, (0, 1))

    def test_word_warmstarts_encode_listed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warmstarts.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "Target": {"7": ["Alpha", "beta", "gamma"]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            words = load_word_warmstarts(
                str(path), target="target", seed=7, count=2, task="semantle"
            )
            self.assertEqual(words, ["Alpha", "beta"])
            ckpt = SimpleNamespace(
                reft_model=object(),
                tokenizer=object(),
                saved_cfg={"task": "semantle"},
            )
            config = SearchConfig(
                output_dir="checkpoint",
                target="Target",
                warmstart_source="words",
                warmstart_file=str(path),
                warmstart_count=2,
            )
            with patch(
                "boreft.search.predict_bias_vectors_for_words",
                return_value=[np.array([0.1]), np.array([0.2])],
            ) as predict, patch(
                "boreft.search.bias_predict_kwargs",
                return_value={"embed_model": "custom/encoder"},
            ):
                points, records = select_warmstarts(
                    config,
                    ckpt,
                    np.array([[0.0], [1.0]]),
                    np.array([[0.0], [1.0]]),
                    seed=7,
                )
            np.testing.assert_allclose(points[:, 0], [0.1, 0.2])
            self.assertEqual(
                [record["decoded"] for record in records], ["alpha", "beta"]
            )
            self.assertTrue(records[0]["components"]["warmstart_labeled"])
            self.assertEqual(predict.call_args.args[1], ["Alpha", "beta"])
            self.assertEqual(predict.call_args.kwargs["embed_model"], "custom/encoder")

    def test_word_warmstarts_require_a_file(self):
        with self.assertRaisesRegex(ValueError, "warmstart-file"):
            SearchConfig(
                output_dir="checkpoint",
                target="target",
                warmstart_source="words",
            ).validate()

    def test_file_warmstarts_accept_bias_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm.jsonl"
            path.write_text(
                '{"bias":[1,2],"decoded":"a","score":0.2}\n'
                '{"point":[3,4],"decoded":"b"}\n',
                encoding="utf-8",
            )
            points, records = _load_file_warmstarts(str(path), 2, 2)
            np.testing.assert_array_equal(points, [[1, 2], [3, 4]])
            self.assertEqual(records[0]["score"], 0.2)

    def test_repeat_seed_expansion(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            seeds=(5,),
            repeats=3,
        )
        self.assertEqual(_resolved_seeds(config), (5, 6, 7))

    def test_rejects_explicit_seeds_plus_repeats(self):
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            seeds=(1, 2),
            repeats=2,
        )
        with self.assertRaisesRegex(ValueError, "either"):
            _resolved_seeds(config)

    def test_rejects_empty_and_duplicate_seeds(self):
        for seeds in ((), (1, 1)):
            config = SearchConfig(
                output_dir="checkpoint",
                target="target",
                seeds=seeds,
            )
            with self.assertRaisesRegex(ValueError, "seed"):
                _resolved_seeds(config)

    def test_search_root_cannot_delete_checkpoint_or_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "outputs" / "run"
            checkpoint.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "contain"):
                _safe_search_root(str(checkpoint), str(checkpoint))
            with self.assertRaisesRegex(ValueError, "contain"):
                _safe_search_root(str(Path(tmp) / "outputs"), str(checkpoint))
            nested = _safe_search_root(
                str(checkpoint / "search" / "target"), str(checkpoint)
            )
            self.assertEqual(nested, (checkpoint / "search" / "target").resolve())

    @patch("boreft.search.predict_bias_vectors_for_words")
    @patch("boreft.search.generate_base", side_effect=["seed-a", "seed-b"])
    def test_llm_warmstart_preserves_custom_embed_model(self, _generate, predict):
        predict.return_value = [np.array([0.1]), np.array([0.2])]
        intervention = SimpleNamespace(
            bias_network=object(),
            bias_input_source="embed_cache",
        )
        ckpt = SimpleNamespace(
            reft_model=SimpleNamespace(
                interventions={"x": intervention},
                model=object(),
            ),
            tokenizer=object(),
            prompt="prompt",
            from_chat_template=False,
            assistant_suffix=None,
            saved_cfg={
                "task": "semantle",
                "bias_network_embed_model": "custom/encoder",
            },
        )
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            warmstart_source="llm",
            warmstart_count=2,
        )
        points, _ = select_warmstarts(
            config,
            ckpt,
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [1.0]]),
            seed=1,
        )
        np.testing.assert_allclose(points[:, 0], [0.1, 0.2])
        self.assertEqual(predict.call_args.kwargs["embed_model"], "custom/encoder")

    @patch("boreft.search.predict_bias_vectors_from_raw_texts")
    @patch("boreft.search.generate_base", side_effect=["seed-a", "seed-b"])
    def test_llm_warmstart_preserves_encoder_settings(self, _generate, predict):
        predict.return_value = [np.array([0.1]), np.array([0.2])]
        intervention = SimpleNamespace(
            bias_network=object(),
            bias_input_source="llm_encoder",
        )
        ckpt = SimpleNamespace(
            reft_model=SimpleNamespace(
                interventions={"x": intervention},
                model=object(),
            ),
            tokenizer=object(),
            prompt="prompt",
            from_chat_template=False,
            assistant_suffix=None,
            saved_cfg={
                "task": "semantle",
                "bias_input_source": "llm_encoder",
                "bias_encoder_max_length": 123,
                "bias_encoder_layer_index": 7,
            },
        )
        config = SearchConfig(
            output_dir="checkpoint",
            target="target",
            warmstart_source="llm",
            warmstart_count=2,
        )
        select_warmstarts(
            config,
            ckpt,
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [1.0]]),
            seed=1,
        )
        self.assertEqual(predict.call_args.kwargs["encoder_max_length"], 123)
        self.assertEqual(predict.call_args.kwargs["encoder_layer_index"], 7)

    def test_resume_rejects_changed_objective_but_allows_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=10,
                resume=False,
            )
            path.write_text(json.dumps(asdict(original)), encoding="utf-8")
            extended = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                resume=True,
            )
            _validate_resume_config(path, extended)
            longer = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                max_new_tokens=256,
                resume=True,
            )
            _validate_resume_config(path, longer)
            changed = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="rdkit_sim",
                budget=20,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "objective"):
                _validate_resume_config(path, changed)
            changed_target = SearchConfig(
                output_dir="checkpoint",
                target="CCN",
                objective="tfs",
                budget=20,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "target"):
                _validate_resume_config(path, changed_target)
            changed_ard = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                use_ard=True,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "use_ard"):
                _validate_resume_config(path, changed_ard)
            changed_layers = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                projection_layers=2,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "projection_layers"):
                _validate_resume_config(path, changed_layers)
            changed_kernel = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                kernel="rbf",
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "kernel"):
                _validate_resume_config(path, changed_kernel)
            changed_aabb = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                aabb_std_k=1.0,
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "aabb_std_k"):
                _validate_resume_config(path, changed_aabb)
            changed_domain = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                search_domain="ellipsoid",
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "search_domain"):
                _validate_resume_config(path, changed_domain)
            wandb_on = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                resume=True,
                wandb_project="boreft",
                wandb_entity="example",
            )
            _validate_resume_config(path, wandb_on)
            changed_expand = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                objective="tfs",
                budget=20,
                resume=True,
                expand=SearchExpandConfig(every=2),
            )
            with self.assertRaisesRegex(ValueError, "expand.every"):
                _validate_resume_config(path, changed_expand)

    def test_resume_allows_expand_recipe_eval_and_stop_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                budget=10,
                resume=False,
                expand=SearchExpandConfig(every=50, epochs=5),
            )
            path.write_text(json.dumps(asdict(original)), encoding="utf-8")
            resumed = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                budget=10,
                resume=True,
                expand=SearchExpandConfig(
                    every=50,
                    new_only_epochs=4,
                    epochs=1,
                    stop_threshold_min=0.8,
                    eval_epochs=1,
                ),
            )
            _validate_resume_config(path, resumed)
            changed_every = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                budget=10,
                resume=True,
                expand=SearchExpandConfig(every=10, epochs=1, eval_epochs=1),
            )
            with self.assertRaisesRegex(ValueError, "expand.every"):
                _validate_resume_config(path, changed_every)
            changed_learn = SearchConfig(
                output_dir="checkpoint",
                target="CCO",
                budget=10,
                resume=True,
                expand=SearchExpandConfig(every=50, learn_w=False),
            )
            with self.assertRaisesRegex(ValueError, "expand.learn_w"):
                _validate_resume_config(path, changed_learn)

    def test_resume_accepts_older_config_missing_new_surrogate_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = asdict(
                SearchConfig(
                    output_dir="checkpoint",
                    target="CCO",
                    objective="tfs",
                    budget=10,
                )
            )
            del original["use_ard"]
            del original["projection_layers"]
            del original["kernel"]
            del original["expand"]
            path.write_text(json.dumps(original), encoding="utf-8")
            _validate_resume_config(
                path,
                SearchConfig(
                    output_dir="checkpoint",
                    target="CCO",
                    objective="tfs",
                    budget=20,
                    resume=True,
                ),
            )
            with self.assertRaisesRegex(ValueError, "use_ard"):
                _validate_resume_config(
                    path,
                    SearchConfig(
                        output_dir="checkpoint",
                        target="CCO",
                        objective="tfs",
                        budget=20,
                        use_ard=True,
                        resume=True,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "kernel"):
                _validate_resume_config(
                    path,
                    SearchConfig(
                        output_dir="checkpoint",
                        target="CCO",
                        objective="tfs",
                        budget=20,
                        kernel="rbf",
                        resume=True,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
