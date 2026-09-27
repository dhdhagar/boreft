"""TDC oracle scoring and the three-way molopt screen (no live TDC download)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from boreft.chem import qed_value
from boreft.oracle_screen import (
    format_screen_table,
    load_library_splits,
    overlay_generation_prompt_cfg,
    plot_oracle_histograms,
    repair_invalid_smiles,
    run_three_way_screen,
    score_payload,
    score_split,
    screen_metrics_for_wandb,
    smiles_from_items_json,
    smiles_from_sobol_results,
    sobol_decode_tag,
    sobol_generation_kwargs,
    summarize_split,
    write_decoded_smiles,
    write_screen_report,
)
from boreft.oracles import (
    DUAL_KINASE_ORACLE,
    INVALID_SCORE,
    MOLOPT_SEARCH_ORACLE_NAMES,
    TDC_ORACLE_NAMES,
    MoleculeScore,
    coerce_oracle_scores,
    load_property_oracle,
    load_tdc_oracle,
    normalize_property_oracle_name,
    normalize_tdc_oracle_name,
    prepare_tdc_oracle_dir,
    property_oracle_factors,
    property_search_task_description,
    score_molecules,
    summarize_scores,
    _upgrade_sklearn_tree_nodes,
)

ETHANOL = "CCO"
ETHANOL_PERM = "OCC"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class FakeOracle:
    """Deterministic stand-in: ethanol 0.8, benzene 0.2, else 0.1."""

    def __call__(self, smiles):
        out = []
        for text in smiles:
            if text in {ETHANOL, ETHANOL_PERM}:
                out.append(0.8)
            elif text == BENZENE:
                out.append(0.2)
            else:
                out.append(0.1)
        return out


class QedTests(unittest.TestCase):
    def test_ethanol_is_in_unit_interval(self):
        value = qed_value(ETHANOL)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_invalid_is_none(self):
        self.assertIsNone(qed_value("not smiles"))


class OracleNameTests(unittest.TestCase):
    def test_aliases_normalize_to_tdc_names(self):
        self.assertEqual(normalize_tdc_oracle_name("drd2"), "DRD2")
        self.assertEqual(normalize_tdc_oracle_name("GSK3β"), "GSK3B")
        self.assertEqual(normalize_tdc_oracle_name("gsk3beta"), "GSK3B")
        self.assertEqual(normalize_tdc_oracle_name("jnk3"), "JNK3")

    def test_unknown_oracle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown TDC oracle"):
            normalize_tdc_oracle_name("QED")

    def test_dual_kinase_is_a_search_oracle_not_a_tdc_name(self):
        self.assertEqual(normalize_property_oracle_name("jnk3*gsk3b"), DUAL_KINASE_ORACLE)
        self.assertEqual(normalize_property_oracle_name("dual_kinase"), DUAL_KINASE_ORACLE)
        self.assertEqual(normalize_property_oracle_name("GSK3β_JNK3"), DUAL_KINASE_ORACLE)
        self.assertEqual(property_oracle_factors("JNK3_GSK3B"), ("GSK3B", "JNK3"))
        with self.assertRaisesRegex(ValueError, "unknown TDC oracle"):
            normalize_tdc_oracle_name(DUAL_KINASE_ORACLE)
        with self.assertRaisesRegex(ValueError, "unknown property oracle"):
            normalize_property_oracle_name("QED")

    def test_property_search_prompt_matches_mist_completion_prefix(self):
        from boreft.task_config import task_instruction

        prefix = task_instruction("molopt", use_chat_template=False)
        drd2 = property_search_task_description("DRD2")
        jnk3 = property_search_task_description("jnk3")
        dual = property_search_task_description("jnk3*gsk3b")
        self.assertEqual(drd2, f"The task is to optimize for DRD2 binding.\n{prefix}")
        self.assertEqual(jnk3, f"The task is to optimize for JNK3 inhibition.\n{prefix}")
        self.assertEqual(
            dual,
            "The task is to optimize the product of GSK3β (GSK3B) and JNK3 inhibition.\n"
            f"{prefix}",
        )
        self.assertTrue(drd2.endswith(":"))
        self.assertNotIn("Generate", prefix)


class ScoreMoleculesTests(unittest.TestCase):
    def test_invalid_scores_zero_and_skips_oracle(self):
        calls: list[list[str]] = []

        def oracle(smiles):
            calls.append(list(smiles))
            return [0.5] * len(smiles)

        rows = score_molecules(
            [ETHANOL, "not a molecule"], oracle, include_qed=False
        )
        self.assertEqual(calls, [[ETHANOL]])
        self.assertTrue(rows[0].valid)
        self.assertEqual(rows[0].score, 0.5)
        self.assertFalse(rows[1].valid)
        self.assertEqual(rows[1].score, INVALID_SCORE)

    def test_canonical_form_is_what_the_oracle_sees(self):
        seen: list[str] = []

        def oracle(smiles):
            seen.extend(smiles)
            return [0.9] * len(smiles)

        rows = score_molecules([ETHANOL_PERM], oracle, include_qed=False)
        self.assertEqual(seen, [ETHANOL])
        self.assertEqual(rows[0].canonical, ETHANOL)
        self.assertEqual(rows[0].score, 0.9)

    def test_batch_length_mismatch_falls_back_per_molecule(self):
        def oracle(smiles):
            if len(smiles) > 1:
                return [0.1]
            return [0.7]

        rows = score_molecules(
            [ETHANOL, BENZENE], oracle, include_qed=False
        )
        self.assertEqual([row.score for row in rows], [0.7, 0.7])

    def test_single_molecule_oracle_error_is_not_swallowed(self):
        def oracle(_smiles):
            raise RuntimeError("rf pickle")

        with self.assertRaises(RuntimeError):
            score_molecules([ETHANOL], oracle, include_qed=False)

    def test_non_finite_scores_become_invalid_score(self):
        def oracle(_smiles):
            return [float("nan"), float("inf")]

        rows = score_molecules(
            [ETHANOL, BENZENE], oracle, include_qed=False
        )
        self.assertEqual([row.score for row in rows], [0.0, 0.0])

    def test_scalar_oracle_for_one_molecule(self):
        def oracle(smiles):
            self.assertEqual(len(smiles), 1)
            return 0.6

        rows = score_molecules([ETHANOL], oracle, include_qed=False)
        self.assertEqual(rows[0].score, 0.6)


class CoerceOracleScoresTests(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "returned 1 scores for 2"):
            coerce_oracle_scores([0.1], n=2)

    def test_scalar_matches_n_one(self):
        self.assertEqual(coerce_oracle_scores(0.4, n=1), [0.4])

    def test_nan_becomes_invalid_score(self):
        self.assertEqual(
            coerce_oracle_scores([float("nan")], n=1), [INVALID_SCORE]
        )


class SummarizeScoresTests(unittest.TestCase):
    def test_valid_and_all_differ_when_invalids_are_present(self):
        rows = score_molecules(
            [ETHANOL, "junk"], FakeOracle(), include_qed=False
        )
        stats = summarize_scores(rows)
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["n_valid"], 1)
        self.assertEqual(stats["max_valid"], 0.8)
        self.assertEqual(stats["max_all"], 0.8)
        self.assertEqual(stats["mean_all"], 0.4)
        self.assertEqual(stats["frac_ge_0p5_valid"], 1.0)
        self.assertEqual(stats["frac_ge_0p5_all"], 0.5)
        self.assertEqual(stats["p05_valid"], 0.8)
        self.assertEqual(stats["p95_valid"], 0.8)

    def test_empty_valid_stats_are_none(self):
        rows = score_molecules(["??"], FakeOracle(), include_qed=False)
        stats = summarize_scores(rows)
        self.assertIsNone(stats["max_valid"])
        self.assertEqual(stats["max_all"], 0.0)
        self.assertEqual(stats["validity"], 0.0)
        self.assertIsNone(stats["p05_valid"])
        self.assertEqual(stats["p05_all"], 0.0)

    def test_quantiles_interpolate(self):
        rows = [
            MoleculeScore(
                smiles=str(i), canonical=str(i), valid=True, score=float(i)
            )
            for i in range(5)
        ]
        stats = summarize_scores(rows)
        self.assertAlmostEqual(stats["p05_valid"], 0.2)
        self.assertEqual(stats["p25_valid"], 1.0)
        self.assertEqual(stats["p75_valid"], 3.0)
        self.assertAlmostEqual(stats["p95_valid"], 3.8)


class ProductOracleTests(unittest.TestCase):
    def test_product_multiplies_factor_oracles(self):
        def gsk(smiles):
            return [0.5] * len(smiles)

        def jnk(smiles):
            return [0.4] * len(smiles)

        fn = load_property_oracle(
            "GSK3B_JNK3",
            factor_oracles={"GSK3B": gsk, "JNK3": jnk},
        )
        self.assertEqual(fn(["CCO", "c1ccccc1"]), [0.2, 0.2])

    def test_score_molecules_zeros_invalid_before_product(self):
        calls: list[list[str]] = []

        def gsk(smiles):
            calls.append(["gsk", *smiles])
            return [0.5] * len(smiles)

        def jnk(smiles):
            calls.append(["jnk", *smiles])
            return [0.4] * len(smiles)

        fn = load_property_oracle(
            DUAL_KINASE_ORACLE,
            factor_oracles={"GSK3B": gsk, "JNK3": jnk},
        )
        rows = score_molecules([ETHANOL, "not a molecule"], fn, include_qed=False)
        self.assertEqual(rows[0].score, 0.2)
        self.assertEqual(rows[1].score, INVALID_SCORE)
        self.assertFalse(rows[1].valid)
        self.assertEqual(calls, [["gsk", ETHANOL], ["jnk", ETHANOL]])


class LoadTdcOracleTests(unittest.TestCase):
    def test_unknown_name_is_rejected_before_importing_tdc(self):
        with self.assertRaisesRegex(ValueError, "unknown TDC oracle"):
            load_tdc_oracle("QED")

    def test_probe_rejects_silent_zero(self):
        from boreft.oracles import _assert_tdc_oracle_alive

        def oracle(_smiles):
            return [0.0]

        with self.assertRaisesRegex(RuntimeError, "GSK3B scored 0"):
            _assert_tdc_oracle_alive("GSK3B", oracle)

    def test_probe_accepts_docs_score(self):
        from boreft.oracles import _assert_tdc_oracle_alive

        def oracle(_smiles):
            return [0.03]

        _assert_tdc_oracle_alive("GSK3B", oracle)

    def test_default_names(self):
        self.assertEqual(TDC_ORACLE_NAMES, ("DRD2", "GSK3B", "JNK3"))
        self.assertEqual(
            MOLOPT_SEARCH_ORACLE_NAMES,
            ("DRD2", "GSK3B", "JNK3", DUAL_KINASE_ORACLE),
        )

    def test_prepare_dir_is_idempotent(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True)
        )
        path = os.path.join(tmpdir, "oracle")
        self.assertEqual(prepare_tdc_oracle_dir(path), path)
        self.assertEqual(prepare_tdc_oracle_dir(path), path)
        self.assertTrue(os.path.isdir(path))

    def test_mkdir_of_existing_dir_is_ok(self):
        from boreft.oracles import _tdc_mkdir_exist_ok

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True)
        )
        path = os.path.join(tmpdir, "oracle")
        os.mkdir(path)
        with _tdc_mkdir_exist_ok():
            os.mkdir(path)
        self.assertTrue(os.path.isdir(path))

    def test_prepare_dir_rejects_a_file_in_the_way(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        with self.assertRaisesRegex(RuntimeError, "is a file"):
            prepare_tdc_oracle_dir(tmp.name)

    def test_legacy_tree_nodes_gain_missing_go_to_left(self):
        import numpy as np
        from sklearn.tree._tree import NODE_DTYPE

        old_dtype = np.dtype(
            {
                "names": [
                    "left_child",
                    "right_child",
                    "feature",
                    "threshold",
                    "impurity",
                    "n_node_samples",
                    "weighted_n_node_samples",
                ],
                "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8"],
                "offsets": [0, 8, 16, 24, 32, 40, 48],
                "itemsize": 56,
            }
        )
        nodes = np.zeros(2, dtype=old_dtype)
        nodes["threshold"] = [0.5, 1.5]
        upgraded = _upgrade_sklearn_tree_nodes(nodes)
        self.assertEqual(upgraded.dtype, NODE_DTYPE)
        self.assertEqual(list(upgraded["threshold"]), [0.5, 1.5])
        self.assertTrue((upgraded["missing_go_to_left"] == 0).all())

    def test_new_tree_nodes_are_left_alone(self):
        import numpy as np
        from sklearn.tree._tree import NODE_DTYPE

        nodes = np.zeros(1, dtype=NODE_DTYPE)
        self.assertIs(_upgrade_sklearn_tree_nodes(nodes), nodes)

    def test_legacy_setstate_roundtrip_with_compat_patch(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree._tree import Tree

        from boreft.oracles import _sklearn_legacy_tree_pickle_compat

        X, y = make_classification(n_samples=24, n_features=4, random_state=0)
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        state = clf.tree_.__getstate__()
        old_dtype = np.dtype(
            {
                "names": [
                    "left_child",
                    "right_child",
                    "feature",
                    "threshold",
                    "impurity",
                    "n_node_samples",
                    "weighted_n_node_samples",
                ],
                "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8"],
                "offsets": [0, 8, 16, 24, 32, 40, 48],
                "itemsize": 56,
            }
        )
        old_nodes = np.empty(state["nodes"].shape, dtype=old_dtype)
        for name in old_dtype.names:
            old_nodes[name] = state["nodes"][name]
        legacy = dict(state)
        legacy["nodes"] = old_nodes
        rebuilt = Tree(4, np.array([2], dtype=np.intp), 1)
        with _sklearn_legacy_tree_pickle_compat():
            rebuilt.__setstate__(legacy)
        x = np.asarray(X[:3], dtype=np.float32)
        np.testing.assert_allclose(clf.tree_.predict(x), rebuilt.predict(x))

    def test_legacy_forest_gains_estimator_attr(self):
        import pickle
        from io import BytesIO

        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        from boreft.oracles import (
            _sklearn_legacy_tree_pickle_compat,
            _upgrade_sklearn_forest,
        )

        X, y = make_classification(n_samples=24, n_features=4, random_state=0)
        clf = RandomForestClassifier(
            n_estimators=3, max_depth=1, random_state=0
        ).fit(X, y)
        proto = clf.estimator
        clf.base_estimator = proto
        del clf.estimator
        with self.assertRaises(AttributeError):
            clf.predict_proba(X[:2])
        _upgrade_sklearn_forest(clf)
        self.assertIsNotNone(clf.estimator)
        proba = clf.predict_proba(np.asarray(X[:2], dtype=np.float32))
        self.assertEqual(proba.shape[0], 2)

        clf2 = RandomForestClassifier(
            n_estimators=3, max_depth=1, random_state=0
        ).fit(X, y)
        clf2.base_estimator = clf2.estimator
        del clf2.estimator
        buf = BytesIO()
        pickle.dump(clf2, buf)
        buf.seek(0)
        with _sklearn_legacy_tree_pickle_compat():
            loaded = pickle.load(buf)
        self.assertIsNotNone(getattr(loaded, "estimator", None))
        loaded.predict_proba(np.asarray(X[:2], dtype=np.float32))

    def test_row_normalize_turns_leaf_counts_into_probabilities(self):
        import numpy as np

        from boreft.oracles import _row_normalize_predict_proba

        np.testing.assert_allclose(
            _row_normalize_predict_proba(np.array([[922.0, 0.0], [10.0, 90.0]])),
            [[1.0, 0.0], [0.1, 0.9]],
        )
        np.testing.assert_allclose(
            _row_normalize_predict_proba(np.array([50.0, 50.0])),
            [0.5, 0.5],
        )

    def test_predict_proba_shim_keeps_fitted_trees_as_probabilities(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.tree import DecisionTreeClassifier

        from boreft.oracles import _sklearn_legacy_tree_pickle_compat

        X, y = make_classification(n_samples=24, n_features=4, random_state=0)
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        x = np.asarray(X[:4], dtype=np.float32)
        with _sklearn_legacy_tree_pickle_compat():
            proba = clf.predict_proba(x)
        self.assertEqual(proba.shape[0], 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))


class LibrarySplitTests(unittest.TestCase):
    def write_csv(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        tmp.write(text)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_prefix_is_train_and_tail_is_held_out(self):
        path = self.write_csv(
            "inchikey,smiles\n"
            "A,CCO\n"
            "B,c1ccccc1\n"
            "C,CCN\n"
            "D,CCC\n"
        )
        splits = load_library_splits(path, train_top_k=2)
        self.assertEqual(splits["train"], [ETHANOL, BENZENE])
        self.assertEqual(len(splits["held_out"]), 2)
        self.assertNotIn(ETHANOL, splits["held_out"])

    def test_equivalent_spelling_of_train_is_not_held_out(self):
        path = self.write_csv("inchikey,smiles\nA,CCO\nB,OCC\nC,CCC\n")
        splits = load_library_splits(path, train_smiles=[ETHANOL])
        self.assertEqual(splits["held_out"], ["CCC"])

    def test_held_out_n_subsamples_deterministically(self):
        path = self.write_csv(
            "inchikey,smiles\nA,CCO\nB,c1ccccc1\nC,CCN\nD,CCC\nE,CCCC\n"
        )
        a = load_library_splits(path, train_top_k=1, held_out_n=2, seed=0)
        b = load_library_splits(path, train_top_k=1, held_out_n=2, seed=0)
        self.assertEqual(a["held_out"], b["held_out"])
        self.assertEqual(len(a["held_out"]), 2)


class ItemsJsonTests(unittest.TestCase):
    def write_json(self, payload) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_reads_word_and_unwraps_tags(self):
        path = self.write_json(
            [
                {
                    "id": 0,
                    "prompt": "p",
                    "target": "<SMILES>CCO</SMILES>",
                    "word": "<SMILES>CCO</SMILES>",
                },
                {"id": 1, "prompt": "p", "target": "<SMILES>c1ccccc1</SMILES>"},
            ]
        )
        self.assertEqual(smiles_from_items_json(path), [ETHANOL, BENZENE])

    def test_rejects_object_payload(self):
        path = self.write_json({"items": [{"word": ETHANOL}]})
        with self.assertRaises(ValueError):
            smiles_from_items_json(path)


class SobolDumpTests(unittest.TestCase):
    def test_reads_greedy_samples(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(
            {
                "greedy": {
                    "per_sample": [
                        {"samples": [ETHANOL]},
                        {"sample_mode": BENZENE, "samples": []},
                    ]
                }
            },
            tmp,
        )
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        self.assertEqual(
            smiles_from_sobol_results(tmp.name), [ETHANOL, BENZENE]
        )

    def test_legacy_flat_dump_is_greedy(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"per_sample": [{"samples": [ETHANOL]}]}, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        self.assertEqual(smiles_from_sobol_results(tmp.name), [ETHANOL])

    def test_json_list_dump(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump([ETHANOL, f"<SMILES>{BENZENE}</SMILES>"], tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        self.assertEqual(
            smiles_from_sobol_results(tmp.name), [ETHANOL, BENZENE]
        )

    def test_write_decoded_smiles_roundtrip(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True)
        )
        path = os.path.join(tmpdir, "sobol_decoded.json")
        write_decoded_smiles(path, [ETHANOL, BENZENE])
        self.assertEqual(smiles_from_sobol_results(path), [ETHANOL, BENZENE])


class RepairInvalidSmilesTests(unittest.TestCase):
    def test_repairs_only_invalid(self):
        from unittest.mock import patch

        from boreft.oracle_screen import repair_invalid_smiles

        with (
            patch("boreft.oracle_screen.require_smiself"),
            patch(
                "boreft.oracle_screen.repair_smiles", return_value=ETHANOL
            ) as mock_repair,
        ):
            out, stats = repair_invalid_smiles(
                [ETHANOL, "not a molecule", BENZENE]
            )
        self.assertEqual(out, [ETHANOL, ETHANOL, BENZENE])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["n_invalid"], 1)
        self.assertEqual(stats["n_repaired"], 1)
        self.assertEqual(stats["n_still_invalid"], 0)
        mock_repair.assert_called_once_with("not a molecule")

    def test_failed_repair_keeps_original(self):
        from unittest.mock import patch

        from boreft.oracle_screen import repair_invalid_smiles

        with (
            patch("boreft.oracle_screen.require_smiself"),
            patch("boreft.oracle_screen.repair_smiles", return_value=None),
        ):
            out, stats = repair_invalid_smiles(["??"])
        self.assertEqual(out, ["??"])
        self.assertEqual(stats["n_repaired"], 0)
        self.assertEqual(stats["n_still_invalid"], 1)

    def test_missing_smiself_raises(self):
        import sys
        from unittest.mock import patch

        from boreft.oracle_screen import repair_invalid_smiles

        with patch.dict(sys.modules, {"smiself": None}):
            with self.assertRaises(ImportError) as ctx:
                repair_invalid_smiles(["??"])
        self.assertIn("install_smiself.sh", str(ctx.exception))


class GenerationPromptOverlayTests(unittest.TestCase):
    def test_generation_from_task_config_keeps_checkpoint_system(self):
        from boreft.task_config import task_instruction

        old = {
            "task": "molopt",
            "use_chat_template": True,
            "chat_instruction": "OLD USER PROMPT",
            "system_prompt": "OLD SYSTEM",
        }
        cfg = overlay_generation_prompt_cfg(
            old, generation_prompt_from_task_config=True
        )
        self.assertEqual(
            cfg["chat_instruction"],
            task_instruction("molopt", use_chat_template=True),
        )
        self.assertEqual(cfg["system_prompt"], "OLD SYSTEM")
        self.assertTrue(cfg["use_chat_template"])

    def test_system_from_task_config_keeps_checkpoint_instruction(self):
        from boreft.task_config import task_system_prompt

        old = {
            "task": "molopt",
            "chat_instruction": "OLD USER PROMPT",
            "system_prompt": "OLD SYSTEM",
        }
        cfg = overlay_generation_prompt_cfg(
            old, system_prompt_from_task_config=True
        )
        self.assertEqual(cfg["chat_instruction"], "OLD USER PROMPT")
        self.assertEqual(cfg["system_prompt"], task_system_prompt("molopt"))

    def test_both_from_task_config(self):
        from boreft.task_config import task_instruction, task_system_prompt

        old = {
            "task": "molopt",
            "chat_instruction": "OLD USER PROMPT",
            "system_prompt": "OLD SYSTEM",
        }
        cfg = overlay_generation_prompt_cfg(
            old,
            generation_prompt_from_task_config=True,
            system_prompt_from_task_config=True,
        )
        self.assertEqual(
            cfg["chat_instruction"],
            task_instruction("molopt", use_chat_template=True),
        )
        self.assertEqual(cfg["system_prompt"], task_system_prompt("molopt"))

    def test_custom_prompt_keeps_checkpoint_system(self):
        old = {
            "task": "molopt",
            "chat_instruction": "OLD",
            "system_prompt": "KEEP ME",
        }
        cfg = overlay_generation_prompt_cfg(
            old, generation_prompt="  Generate a molecule.  "
        )
        self.assertEqual(cfg["chat_instruction"], "Generate a molecule.")
        self.assertEqual(cfg["system_prompt"], "KEEP ME")

    def test_custom_prompt_with_system_from_task_config(self):
        from boreft.task_config import task_system_prompt

        old = {
            "task": "molopt",
            "chat_instruction": "OLD",
            "system_prompt": "OLD SYSTEM",
        }
        cfg = overlay_generation_prompt_cfg(
            old,
            generation_prompt="Generate a molecule.",
            system_prompt_from_task_config=True,
        )
        self.assertEqual(cfg["chat_instruction"], "Generate a molecule.")
        self.assertEqual(cfg["system_prompt"], task_system_prompt("molopt"))

    def test_no_overlay_copies_cfg(self):
        old = {"chat_instruction": "OLD"}
        cfg = overlay_generation_prompt_cfg(old)
        self.assertEqual(cfg, old)
        self.assertIsNot(cfg, old)

    def test_cannot_combine_custom_and_generation_from_task_config(self):
        with self.assertRaises(ValueError):
            overlay_generation_prompt_cfg(
                {},
                generation_prompt="x",
                generation_prompt_from_task_config=True,
            )


class ScreenReportTests(unittest.TestCase):
    def test_split_summary_and_table(self):
        scored = score_split([ETHANOL, BENZENE], {"DRD2": FakeOracle()})
        summary = summarize_split(scored)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["oracles"]["DRD2"]["max_valid"], 0.8)
        self.assertIsNotNone(summary["oracles"]["DRD2"]["mean_qed_valid"])
        table = format_screen_table({"train": summary}, ["DRD2"])
        self.assertIn("train", table)
        self.assertIn("0.8000", table)

    def test_three_way_screen_and_wandb_flatten(self):
        summaries, scores = run_three_way_screen(
            {"train": [ETHANOL], "test": [BENZENE], "sobol": [ASPIRIN]},
            {"DRD2": FakeOracle()},
        )
        self.assertEqual(set(summaries), {"train", "test", "sobol"})
        self.assertIn("DRD2", scores["test"])
        flat = screen_metrics_for_wandb(summaries)
        self.assertAlmostEqual(flat["oracle_screen/train/DRD2/max_valid"], 0.8)
        self.assertAlmostEqual(flat["oracle_screen/test/DRD2/max_valid"], 0.2)
        self.assertIn("oracle_screen/sobol/validity", flat)

    def test_cache_hits_do_not_call_live_oracle(self):
        from boreft.chem import canonical_target_key
        from boreft.oracle_screen import score_split_with_cache

        cache = {
            canonical_target_key(ETHANOL): {"DRD2": 0.42},
        }
        scored = score_split_with_cache(
            [ETHANOL], cache, {"DRD2": FakeOracle()}, oracles=["DRD2"]
        )
        self.assertAlmostEqual(scored["DRD2"][0].score, 0.42)

    def test_cache_miss_uses_live_oracle(self):
        from boreft.oracle_screen import score_split_with_cache

        scored = score_split_with_cache(
            [ETHANOL], {}, {"DRD2": FakeOracle()}, oracles=["DRD2"]
        )
        self.assertAlmostEqual(scored["DRD2"][0].score, 0.8)

    def test_plot_and_json_roundtrip(self):
        scored = {
            "train": score_split([ETHANOL, ASPIRIN], {"DRD2": FakeOracle()}),
            "held_out": score_split([BENZENE], {"DRD2": FakeOracle()}),
        }
        scores = {name: score_payload(block) for name, block in scored.items()}
        splits = {name: summarize_split(block) for name, block in scored.items()}
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True)
        )
        json_path = os.path.join(tmpdir, "oracle_screen.json")
        write_screen_report(
            json_path,
            meta={"oracles": ["DRD2"]},
            splits=splits,
            scores=scores,
        )
        plot_oracle_histograms(
            scores,
            oracle_names=["DRD2"],
            out_path=os.path.join(tmpdir, "oracle_screen_hist"),
        )
        self.assertTrue(os.path.isfile(json_path))
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, "oracle_screen_hist.pdf")))


class SobolDecodeModeTests(unittest.TestCase):
    def test_temperature_zero_is_greedy(self):
        self.assertEqual(sobol_generation_kwargs(0.0), {"use_sample": False})
        self.assertEqual(sobol_decode_tag(temperature=0.0), "")

    def test_temperature_one_matches_genz(self):
        self.assertEqual(
            sobol_generation_kwargs(1.0),
            {"use_sample": True, "temperature": 1.0, "top_p": 1.0},
        )
        self.assertEqual(sobol_decode_tag(temperature=1.0), "temp1")

    def test_top_p_suffix_when_not_one(self):
        self.assertEqual(
            sobol_decode_tag(temperature=1.0, top_p=0.9), "temp1_topp0.9"
        )

    def test_rejects_invalid_sampling_args(self):
        with self.assertRaises(ValueError):
            sobol_generation_kwargs(-0.1)
        with self.assertRaises(ValueError):
            sobol_generation_kwargs(1.0, 0.0)
        with self.assertRaises(ValueError):
            sobol_generation_kwargs(1.0, 1.1)


class WandbSummaryTests(unittest.TestCase):
    def test_named_summary_flattens_split_oracle_stats(self):
        from boreft.search_wandb import _named_summary

        out = _named_summary(
            {
                "splits": {
                    "train": {
                        "n": 10,
                        "validity": 1.0,
                        "oracles": {"DRD2": {"max_valid": 0.8}},
                    }
                }
            }
        )
        self.assertEqual(out["train/n"], 10.0)
        self.assertEqual(out["train/DRD2/max_valid"], 0.8)


if __name__ == "__main__":
    unittest.main()
