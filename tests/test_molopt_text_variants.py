"""Tests for ``scripts/analyze_molopt_text_variants.py``.

Encoders are stubbed. The experiment's job is to load Qwen; the test suite's
job is to pin the seven strings and the purity numbers against inputs whose
answers are known by construction. RDKit InChI is exercised on ethanol.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_molopt_text_variants.py")
    spec = importlib.util.spec_from_file_location("analyze_molopt_text_variants", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mtv = _load_script()

ETHANOL = "CCO"
ETHANOL_DEFN = "a primary alcohol that is ethane substituted by a hydroxy group."
ETHANOL_IUPAC = "ethanol"
ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
ETHANOL_RDKIT = (46.069, -0.0014, 20.23, 1, 1, 0, 0, 1.0, 0, 0)

BENZENE = "c1ccccc1"
BENZENE_DEFN = "the simplest aromatic hydrocarbon, consisting of a six-membered ring."
BENZENE_IUPAC = "benzene"
BENZENE_RDKIT = (78.114, 1.6866, 0.0, 0, 0, 0, 0, 0.0, 1, 0)


def _mol(
    smiles=ETHANOL,
    definition=ETHANOL_DEFN,
    category="alcohol",
    category_normalized="organooxygen",
    rdkit=ETHANOL_RDKIT,
    iupac=ETHANOL_IUPAC,
):
    return mtv.Molecule(
        smiles=smiles,
        definition=definition,
        category=category,
        category_normalized=category_normalized,
        rdkit_values=rdkit,
        iupac=iupac,
    )


class VariantTextTests(unittest.TestCase):
    def test_smiles_is_bare(self):
        self.assertEqual(mtv.variant_text(_mol(), "smiles"), ETHANOL)

    def test_defn_is_bare_description(self):
        self.assertEqual(mtv.variant_text(_mol(), "defn"), ETHANOL_DEFN)

    def test_smiles_defn_uses_production_template(self):
        self.assertEqual(
            mtv.variant_text(_mol(), "smiles_defn"),
            f"The molecule '{ETHANOL}' is: {ETHANOL_DEFN}",
        )

    def test_iupac_is_the_name(self):
        self.assertEqual(mtv.variant_text(_mol(), "iupac"), ETHANOL_IUPAC)

    def test_iupac_defn_puts_the_name_in_the_molecule_slot(self):
        self.assertEqual(
            mtv.variant_text(_mol(), "iupac_defn"),
            f"The molecule '{ETHANOL_IUPAC}' is: {ETHANOL_DEFN}",
        )

    def test_rdkit_matches_omit_molt5_suffix(self):
        text = mtv.variant_text(_mol(), "rdkit")
        self.assertTrue(text.startswith("2D properties: "))
        self.assertIn("average molecular weight 46.069", text)
        self.assertNotIn(ETHANOL_DEFN, text)

    def test_smiles_rdkit_prefixes_the_smiles(self):
        text = mtv.variant_text(_mol(), "smiles_rdkit")
        self.assertTrue(text.startswith(f"{ETHANOL} 2D properties: "))
        self.assertIn("average molecular weight 46.069", text)

    def test_iupac_variants_require_a_name(self):
        mol = _mol(iupac=None)
        with self.assertRaises(ValueError):
            mtv.variant_text(mol, "iupac")
        with self.assertRaises(ValueError):
            mtv.variant_text(mol, "iupac_defn")


class PurityMetricTests(unittest.TestCase):
    def test_cluster_purity_is_one_when_each_bin_is_pure(self):
        true = ["a", "a", "b", "b"]
        pred = ["0", "0", "1", "1"]
        self.assertEqual(mtv.cluster_purity(true, pred), 1.0)

    def test_cluster_purity_counts_the_majority_in_mixed_bins(self):
        # One mixed cluster of 3: majority a (2/3). Purity = 2/3.
        true = ["a", "a", "b"]
        pred = ["0", "0", "0"]
        self.assertAlmostEqual(mtv.cluster_purity(true, pred), 2.0 / 3.0)

    def test_knn_purity_is_one_on_separated_class_centroids(self):
        labels = ["a", "a", "b", "b"]
        emb = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        self.assertEqual(mtv.knn_purity(emb, labels, k=1), 1.0)
        # Each class has only one other member, so @2 half the neighbours are the other class.
        self.assertEqual(mtv.knn_purity(emb, labels, k=2), 0.5)

    def test_chance_knn_purity_is_leave_one_out_class_prior(self):
        labels = ["a", "a", "a", "b"]
        # P(same | random other) = (3*2 + 1*0) / (4*3) = 0.5
        self.assertAlmostEqual(mtv.chance_knn_purity(labels), 0.5)

    def test_kmeans_purity_recovers_two_blobs(self):
        labels = ["a", "a", "a", "b", "b", "b"]
        emb = np.vstack(
            [
                np.repeat([[1.0, 0.0]], 3, axis=0),
                np.repeat([[0.0, 1.0]], 3, axis=0),
            ]
        )
        purity, pred = mtv.kmeans_purity(emb, labels, seed=0)
        self.assertEqual(purity, 1.0)
        self.assertEqual(len(set(pred.tolist())), 2)

    def test_silhouette_is_positive_on_separated_blobs(self):
        labels = ["a", "a", "a", "b", "b", "b"]
        emb = np.vstack(
            [
                np.repeat([[1.0, 0.0]], 3, axis=0),
                np.repeat([[0.0, 1.0]], 3, axis=0),
            ]
        )
        sil = mtv.cosine_silhouette(emb, labels)
        self.assertGreater(sil["overall"], 0.5)
        self.assertEqual(sil["n_dropped_singleton"], 0)


class SampleAndIupacTests(unittest.TestCase):
    def test_sample_drops_unknown_and_is_seeded(self):
        rows = [
            _mol(smiles="CCO", category_normalized="organooxygen"),
            _mol(smiles="c1ccccc1", category_normalized="benzenoid", rdkit=BENZENE_RDKIT),
            _mol(smiles="O", category_normalized="unknown"),
        ]
        a = mtv.sample_molecules(rows, n=2, seed=0, label_field="category_normalized")
        b = mtv.sample_molecules(rows, n=2, seed=0, label_field="category_normalized")
        self.assertEqual([m.smiles for m in a], [m.smiles for m in b])
        self.assertEqual(len(a), 2)
        self.assertTrue(all(m.category_normalized != "unknown" for m in a))

    def test_sample_always_drops_missing_rdkit(self):
        rows = [
            _mol(smiles="CCO", category_normalized="organooxygen"),
            _mol(smiles="[Na+]", category_normalized="inorganic", rdkit=None),
        ]
        kept = mtv.sample_molecules(
            rows, n=2, seed=0, label_field="category_normalized", drop_unknown=False
        )
        self.assertEqual([m.smiles for m in kept], ["CCO"])

    def test_n_zero_keeps_every_eligible_molecule(self):
        rows = [
            _mol(smiles="CCO", category_normalized="organooxygen"),
            _mol(smiles="c1ccccc1", category_normalized="benzenoid", rdkit=BENZENE_RDKIT),
            _mol(smiles="O", category_normalized="unknown"),
        ]
        kept = mtv.sample_molecules(rows, n=0, seed=0, label_field="category_normalized")
        self.assertEqual(sorted(m.smiles for m in kept), ["CCO", "c1ccccc1"])

    def test_rdkit_iupac_is_inchi(self):
        self.assertEqual(mtv.rdkit_iupac(ETHANOL), ETHANOL_INCHI)

    def test_assign_iupac_names_writes_through(self):
        mols = [_mol(iupac=None), _mol(smiles=BENZENE, rdkit=BENZENE_RDKIT, iupac=None)]
        mtv.assign_iupac_names(mols, lambda s: f"name-{s}")
        self.assertEqual(mols[0].iupac, f"name-{ETHANOL}")
        self.assertEqual(mols[1].iupac, f"name-{BENZENE}")

    def test_empty_variants_means_all(self):
        names = [v.name for v in mtv.selected_variants([])]
        self.assertEqual(names, [v.name for v in mtv.VARIANTS])


class EndToEndStubTests(unittest.TestCase):
    def test_run_scores_label_one_hots_at_purity_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = os.path.join(tmp, "definitions.jsonl")
            rdkit_path = os.path.join(tmp, "rdkit.jsonl")
            smiles = [
                "CCO",
                "CCOC",
                "CCOCC",
                "CCCO",
                "CCCCO",
                "CO",
                "c1ccccc1",
                "Cc1ccccc1",
                "CCc1ccccc1",
                "CCCc1ccccc1",
                "CCCCc1ccccc1",
                "c1ccc(O)cc1",
            ]
            cats = ["organooxygen"] * 6 + ["benzenoid"] * 6
            defns = [
                ETHANOL_DEFN if cat == "organooxygen" else BENZENE_DEFN for cat in cats
            ]
            with open(defs_path, "w", encoding="utf-8") as f:
                for s, cat, defn in zip(smiles, cats, defns):
                    f.write(
                        json.dumps(
                            {
                                "target": s,
                                "definition": defn,
                                "category": cat,
                                "category_normalized": cat,
                            }
                        )
                        + "\n"
                    )
            rdkit_by = {
                s: list(ETHANOL_RDKIT if cat == "organooxygen" else BENZENE_RDKIT)
                for s, cat in zip(smiles, cats)
            }
            with open(rdkit_path, "w", encoding="utf-8") as f:
                for s, values in rdkit_by.items():
                    f.write(json.dumps({"target": s, "definition": values}) + "\n")

            iupac_by = {
                "CCO": "ethanol",
                "CCOC": "methoxyethane",
                "CCOCC": "ethoxyethane",
                "CCCO": "propan-1-ol",
                "CCCCO": "butan-1-ol",
                "CO": "methanol",
                "c1ccccc1": "benzene",
                "Cc1ccccc1": "toluene",
                "CCc1ccccc1": "ethylbenzene",
                "CCCc1ccccc1": "propylbenzene",
                "CCCCc1ccccc1": "butylbenzene",
                "c1ccc(O)cc1": "phenol",
            }
            mols_for_map = [
                _mol(
                    smiles=s,
                    definition=defn,
                    category_normalized=cat,
                    rdkit=tuple(rdkit_by[s]),
                    iupac=iupac_by[s],
                )
                for s, cat, defn in zip(smiles, cats, defns)
            ]
            vec = {
                "organooxygen": np.array([1.0, 0.0], dtype=np.float64),
                "benzenoid": np.array([0.0, 1.0], dtype=np.float64),
            }
            text_to_vec = {}
            for mol, cat in zip(mols_for_map, cats):
                for name in (v.name for v in mtv.VARIANTS):
                    text_to_vec[mtv.variant_text(mol, name)] = vec[cat]

            def encode(texts):
                return np.stack([text_to_vec[t] for t in texts])

            args = mtv.parse_args(
                [
                    "--definitions",
                    defs_path,
                    "--rdkit-definitions",
                    rdkit_path,
                    "--n",
                    "12",
                    "--seed",
                    "0",
                    "--out-dir",
                    tmp,
                ]
            )
            report = mtv.run(args, encode, lambda s: iupac_by[s])
            self.assertEqual(report["n"], 12)
            self.assertEqual(report["n_requested"], 12)
            for name, scores in report["scores"].items():
                self.assertEqual(scores["knn_purity_at_5"], 1.0, msg=name)
                self.assertEqual(scores["kmeans_purity"], 1.0, msg=name)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "text_variants.json")))


if __name__ == "__main__":
    unittest.main()
