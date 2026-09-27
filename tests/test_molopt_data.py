"""Tests for molopt CSV loading (MolOptItem.load_csv, read_csv_smiles)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from boreft.chem import RDKIT_DESCRIPTOR_SCHEMA_VERSION
from boreft.data.base import item_target
from boreft.data.molopt import (
    MolOptItem,
    apply_mist_smiles_tags,
    apply_smiles_tags,
    read_csv_smiles,
)
from boreft.data_utils import apply_chat_format

CSV_TEXT = """inchikey,smiles,qed
KEY1,CCO,0.41
KEY2,c1ccccc1,0.35
KEY3,CC(=O)Oc1ccccc1C(=O)O,0.55
KEY4,CCN(CC)CC,0.29
"""

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOLOPT_TRAIN = os.path.join(_REPO_ROOT, "data", "molopt", "train")
CORPUS_CSV = os.path.join(_MOLOPT_TRAIN, "chebi20.csv")
DEFINITIONS_JSONL = os.path.join(_MOLOPT_TRAIN, "definitions.jsonl")
RDKIT_DEFINITIONS_JSONL = os.path.join(
    _MOLOPT_TRAIN, "definitions_rdkit.jsonl"
)
RDKIT_DEFINITIONS_MAP = os.path.join(
    _MOLOPT_TRAIN, "definitions_rdkit_map.json"
)

# Validating every molecule in the committed corpus with RDKit is slower than it is
# informative; a prefix is enough to catch a malformed or wrongly-columned file.
_CORPUS_CHECK_HEAD = 200


class WriteCsvMixin:
    def write_csv(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        tmp.write(text)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name


class LoadCsvTests(WriteCsvMixin, unittest.TestCase):
    def test_loads_all_rows_in_file_order(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT))
        self.assertEqual(
            [it.target for it in items],
            ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"],
        )
        self.assertEqual([it.inchikey for it in items], ["KEY1", "KEY2", "KEY3", "KEY4"])

    def test_ids_are_contiguous_from_zero(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT))
        self.assertEqual([it.id for it in items], [0, 1, 2, 3])

    def test_top_k_takes_a_prefix(self):
        """--train-top-k must actually shrink the vocabulary (it used to be dropped)."""
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=2)
        self.assertEqual([it.target for it in items], ["CCO", "c1ccccc1"])
        self.assertEqual([it.id for it in items], [0, 1])

    def test_top_k_larger_than_file_keeps_everything(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=99)
        self.assertEqual(len(items), 4)

    def test_prompt_is_the_molopt_instruction(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        self.assertIn("SMILES", items[0].prompt)

    def test_blank_smiles_rows_skipped_without_gaps_in_ids(self):
        text = "inchikey,smiles\nA,CCO\nB,\nC,CCC\n"
        items = MolOptItem.load_csv(self.write_csv(text))
        self.assertEqual([it.target for it in items], ["CCO", "CCC"])
        self.assertEqual([it.id for it in items], [0, 1])

    def test_missing_inchikey_column_is_tolerated(self):
        items = MolOptItem.load_csv(self.write_csv("smiles\nCCO\n"))
        self.assertEqual(items[0].inchikey, "")

    def test_missing_smiles_column_raises(self):
        with self.assertRaisesRegex(ValueError, "smiles"):
            MolOptItem.load_csv(self.write_csv("inchikey,name\nA,ethanol\n"))


class _ChatTok:
    chat_template = "yes"
    eos_token = "</s>"

    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=False, **_kwargs
    ):
        user = conversation[0]["content"]
        if add_generation_prompt:
            return f"<user>{user}<asst>"
        asst = conversation[1]["content"]
        return f"<user>{user}<asst>{asst}</s>"


class SmilesTagsTests(WriteCsvMixin, unittest.TestCase):
    def test_wraps_generation_target_and_keeps_identity_bare(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        apply_smiles_tags(items)
        self.assertEqual(items[0]._raw_word, "CCO")
        self.assertEqual(items[0].target, "<SMILES>CCO</SMILES>")
        self.assertEqual(item_target(items[0]), "CCO")

    def test_chat_format_keeps_tags_in_assistant_suffix(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        apply_smiles_tags(items)
        apply_chat_format(items, _ChatTok(), "Generate a SMILES string.")
        self.assertEqual(items[0]._raw_word, "CCO")
        self.assertTrue(items[0].target.startswith("<SMILES>CCO</SMILES>"))


class MistSmilesTagsTests(WriteCsvMixin, unittest.TestCase):
    def test_wraps_generation_target_as_close_tag_and_keeps_identity_bare(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        apply_mist_smiles_tags(items)
        self.assertEqual(items[0]._raw_word, "CCO")
        self.assertEqual(items[0].target, "CCO [END_SMILES]")
        self.assertEqual(item_target(items[0]), "CCO")

    def test_chat_format_keeps_full_pair_in_assistant_suffix(self):
        items = MolOptItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        apply_mist_smiles_tags(items, include_open=True)
        apply_chat_format(items, _ChatTok(), "Generate a SMILES string.")
        self.assertEqual(items[0]._raw_word, "CCO")
        self.assertTrue(
            items[0].target.startswith("[START_SMILES] CCO [END_SMILES]")
        )


class ReadCsvSmilesTests(WriteCsvMixin, unittest.TestCase):
    def test_reads_whole_file_for_the_held_out_pool(self):
        smiles = read_csv_smiles(self.write_csv(CSV_TEXT))
        self.assertEqual(len(smiles), 4)

    def test_top_k_matches_the_training_head(self):
        path = self.write_csv(CSV_TEXT)
        head = read_csv_smiles(path, top_k=2)
        self.assertEqual(head, [it.target for it in MolOptItem.load_csv(path, top_k=2)])

    def test_missing_smiles_column_raises(self):
        with self.assertRaisesRegex(ValueError, "smiles"):
            read_csv_smiles(self.write_csv("Word,Similarity\ncat,0.9\n"))


class CommittedCorpusTests(unittest.TestCase):
    """The committed ChEBI-20 corpus must stay loadable, since scripts point at it."""

    def test_corpus_loads_with_valid_molecules(self):
        from boreft.chem import is_valid_smiles

        self.assertTrue(os.path.isfile(CORPUS_CSV), CORPUS_CSV)
        items = MolOptItem.load_csv(CORPUS_CSV, top_k=_CORPUS_CHECK_HEAD)
        self.assertEqual(len(items), _CORPUS_CHECK_HEAD)
        self.assertTrue(all(it.inchikey for it in items))
        self.assertTrue(all(is_valid_smiles(it.target) for it in items))

    def test_targets_are_unique(self):
        targets = read_csv_smiles(CORPUS_CSV)
        self.assertGreater(len(targets), 1000)
        self.assertEqual(len(targets), len(set(targets)))

    def test_every_target_has_a_definition(self):
        """SDPO and --use-definition-embeds both refuse to start otherwise."""
        self.assertTrue(os.path.isfile(DEFINITIONS_JSONL), DEFINITIONS_JSONL)
        with open(DEFINITIONS_JSONL, encoding="utf-8") as f:
            defined = {json.loads(line)["target"] for line in f}
        missing = [t for t in read_csv_smiles(CORPUS_CSV) if t not in defined]
        self.assertEqual(missing[:5], [], f"{len(missing)} targets lack definitions")

    def test_rdkit_definitions_match_corpus_and_map(self):
        self.assertTrue(
            os.path.isfile(RDKIT_DEFINITIONS_JSONL), RDKIT_DEFINITIONS_JSONL
        )
        self.assertTrue(os.path.isfile(RDKIT_DEFINITIONS_MAP), RDKIT_DEFINITIONS_MAP)

        with open(RDKIT_DEFINITIONS_MAP, encoding="utf-8") as f:
            descriptor_map = json.load(f)
        positions = descriptor_map["positions"]
        expected_length = descriptor_map["definition_length"]
        self.assertEqual(
            descriptor_map["schema_version"], RDKIT_DESCRIPTOR_SCHEMA_VERSION
        )
        self.assertEqual(expected_length, 10)
        self.assertEqual([p["index"] for p in positions], list(range(expected_length)))
        self.assertTrue(all(p.get("definition_label") for p in positions))
        self.assertTrue(all("meaning" not in p for p in positions))
        self.assertEqual(
            [p["name"] for p in positions],
            [
                "molecular_weight",
                "clogp",
                "tpsa",
                "hydrogen_bond_donors",
                "hydrogen_bond_acceptors",
                "rotatable_bonds",
                "formal_charge",
                "fraction_csp3",
                "aromatic_rings",
                "atom_stereocenters",
            ],
        )
        normalization = descriptor_map["normalization"]
        self.assertEqual(normalization["method"], "median_mad")
        self.assertEqual(normalization["population"], "full_corpus")
        self.assertEqual(normalization["count"], len(read_csv_smiles(CORPUS_CSV)))
        self.assertEqual(len(normalization["median"]), expected_length)
        self.assertEqual(len(normalization["scale"]), expected_length)
        self.assertTrue(all(scale > 0 for scale in normalization["scale"]))

        with open(RDKIT_DEFINITIONS_JSONL, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        corpus_targets = read_csv_smiles(CORPUS_CSV)
        self.assertEqual([r["target"] for r in records], corpus_targets)
        self.assertTrue(
            all(len(r["definition"]) == expected_length for r in records)
        )
        self.assertTrue(
            all(
                isinstance(value, (int, float))
                for record in records
                for value in record["definition"]
            )
        )


if __name__ == "__main__":
    unittest.main()
