from __future__ import annotations

import unittest
from unittest.mock import patch

from boreft.catalog_decode import (
    catalog_maxima,
    compare_decode,
    train_smiles_from_words,
)


class CatalogDecodeTest(unittest.TestCase):
    def test_train_smiles_unwrap_mist_tags(self):
        self.assertEqual(
            train_smiles_from_words(
                ["[START_SMILES]CCO[END_SMILES]", "CCC"]
            ),
            ["CCO", "CCC"],
        )

    def test_catalog_maxima_picks_first_index_on_ties(self):
        maxima = catalog_maxima(
            ["CCO", "CCC", "CC"],
            {"DRD2": [0.1, 0.9, 0.9], "GSK3B": [1.0, 0.2, 0.3], "JNK3": [0.0, 0.0, 0.4]},
        )
        by_oracle = {row.oracle: row for row in maxima}
        self.assertEqual(by_oracle["DRD2"].index, 1)
        self.assertEqual(by_oracle["DRD2"].smiles, "CCC")
        self.assertEqual(by_oracle["GSK3B"].index, 0)
        self.assertEqual(by_oracle["JNK3"].index, 2)
        self.assertAlmostEqual(by_oracle["JNK3"].score, 0.4)

    def test_catalog_maxima_normalizes_score_keys(self):
        maxima = catalog_maxima(
            ["CCO", "CCC"],
            {"drd2": [0.1, 0.8], "gsk3beta": [0.9, 0.2], "jnk3": [0.0, 0.4]},
            oracles=["DRD2", "GSK3β", "JNK3"],
        )
        by_oracle = {row.oracle: row for row in maxima}
        self.assertEqual(by_oracle["DRD2"].index, 1)
        self.assertEqual(by_oracle["GSK3B"].index, 0)
        self.assertEqual(by_oracle["JNK3"].index, 1)

    def test_compare_decode_exact_and_tanimoto(self):
        hit = compare_decode("CCO", "OCC", gold_score=1.0, decoded_score=1.0)
        self.assertTrue(hit.exact)
        self.assertTrue(hit.valid)
        self.assertAlmostEqual(hit.tanimoto or 0.0, 1.0)

        miss = compare_decode("CCO", "CCC", gold_score=1.0, decoded_score=0.2)
        self.assertFalse(miss.exact)
        self.assertTrue(miss.valid)
        self.assertGreater(miss.tanimoto or 0.0, 0.0)
        self.assertLess(miss.tanimoto or 1.0, 1.0)
        self.assertAlmostEqual(miss.decoded_score or 0.0, 0.2)

    def test_compare_decode_invalid_has_no_tanimoto(self):
        with patch("boreft.chem.repair_smiles", return_value=None):
            row = compare_decode("CCO", "not-a-molecule", gold_score=1.0)
        self.assertFalse(row.valid)
        self.assertFalse(row.exact)
        self.assertIsNone(row.tanimoto)
