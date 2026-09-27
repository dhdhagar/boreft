"""Tests for hypogen CSV loading and the MCTS prepare script."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

from boreft.data.hypogen import HypoGenItem, read_csv_hypotheses
from boreft.eval.eval_suite import read_pool_csv_targets
from boreft.task_config import task_instruction
from boreft.text_similarity import format_text_for_embedding

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prepare_script():
    path = os.path.join(_REPO_ROOT, "scripts", "prepare_hypogen.py")
    spec = importlib.util.spec_from_file_location("prepare_hypogen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare_script()
collect_rows = prepare.collect_rows
collect_rows_from_generator_messages = prepare.collect_rows_from_generator_messages
hypothesis_identity_key = prepare.hypothesis_identity_key

CSV_TEXT = """hypothesis
Temperature raises diversification.
Soil diversity predicts species richness.

Area correlates with DR.
"""

_HYPOGEN_DIR = os.path.join(_REPO_ROOT, "data", "hypogen", "evo-fresh-fish")
CORPUS_CSV = os.path.join(_HYPOGEN_DIR, "train.csv")
DEFINITIONS_JSONL = os.path.join(_HYPOGEN_DIR, "definitions.jsonl")
_WARMSTART_DIR = os.path.join(
    _REPO_ROOT, "data", "hypogen", "evo-fresh-fish-warmstart"
)
WARMSTART_CSV = os.path.join(_WARMSTART_DIR, "train.csv")
WARMSTART_DEFINITIONS = os.path.join(_WARMSTART_DIR, "definitions.jsonl")


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
        items = HypoGenItem.load_csv(self.write_csv(CSV_TEXT))
        self.assertEqual(
            [it.target for it in items],
            [
                "Temperature raises diversification.",
                "Soil diversity predicts species richness.",
                "Area correlates with DR.",
            ],
        )

    def test_ids_are_contiguous_from_zero(self):
        items = HypoGenItem.load_csv(self.write_csv(CSV_TEXT))
        self.assertEqual([it.id for it in items], [0, 1, 2])

    def test_top_k_takes_a_prefix(self):
        items = HypoGenItem.load_csv(self.write_csv(CSV_TEXT), top_k=2)
        self.assertEqual(
            [it.target for it in items],
            [
                "Temperature raises diversification.",
                "Soil diversity predicts species richness.",
            ],
        )

    def test_prompt_is_the_hypogen_instruction(self):
        items = HypoGenItem.load_csv(self.write_csv(CSV_TEXT), top_k=1)
        self.assertEqual(
            items[0].prompt, task_instruction("hypogen", use_chat_template=False)
        )
        self.assertIn("hypothesis", items[0].prompt.lower())

    def test_blank_rows_skipped_without_gaps_in_ids(self):
        text = "hypothesis\nA.\n\nB.\n"
        items = HypoGenItem.load_csv(self.write_csv(text))
        self.assertEqual([it.target for it in items], ["A.", "B."])
        self.assertEqual([it.id for it in items], [0, 1])

    def test_missing_hypothesis_column_raises(self):
        with self.assertRaisesRegex(ValueError, "hypothesis"):
            HypoGenItem.load_csv(self.write_csv("Word,Similarity\ncat,1.0\n"))

    def test_read_csv_hypotheses_matches_load_csv(self):
        path = self.write_csv(CSV_TEXT)
        self.assertEqual(
            read_csv_hypotheses(path),
            [it.target for it in HypoGenItem.load_csv(path)],
        )
        self.assertEqual(read_csv_hypotheses(path, top_k=1), [
            "Temperature raises diversification."
        ])

    def test_pool_reader_keeps_file_order(self):
        path = self.write_csv(CSV_TEXT)
        self.assertEqual(
            read_pool_csv_targets(path, task="hypogen"),
            read_csv_hypotheses(path),
        )

    def test_comma_and_quote_roundtrip(self):
        text = (
            "hypothesis\n"
            "\"Larger basins ('area') correlate with diversity, not DR.\"\n"
        )
        path = self.write_csv(text)
        items = HypoGenItem.load_csv(path)
        self.assertEqual(
            items[0].target,
            "Larger basins ('area') correlate with diversity, not DR.",
        )

    def test_embedding_prompt_keeps_braces_in_hypothesis(self):
        text = "Larger basins ({area}) have lower speciation."
        out = format_text_for_embedding(text, task="hypogen")
        self.assertIn("{area}", out)
        self.assertIn(text, out)


class TrainConfigTests(unittest.TestCase):
    def test_hypogen_csv_is_required(self):
        from boreft.train_args import TrainConfig

        with self.assertRaisesRegex(ValueError, "--hypogen-csv"):
            TrainConfig(task="hypogen")

    def test_definitions_path_defaults_to_csv_sibling(self):
        from boreft.train_args import TrainConfig

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        csv_path = os.path.join(tmp.name, "train.csv")
        defs_path = os.path.join(tmp.name, "definitions.jsonl")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("hypothesis\nH.\n")
        with open(defs_path, "w", encoding="utf-8") as f:
            f.write("{}\n")
        cfg = TrainConfig(task="hypogen", hypogen_csv=csv_path)
        self.assertEqual(os.path.abspath(cfg.definitions_path), os.path.abspath(defs_path))


class CorpusLockstepTests(unittest.TestCase):
    @unittest.skipUnless(
        os.path.isfile(CORPUS_CSV) and os.path.isfile(DEFINITIONS_JSONL),
        "hypogen corpus not generated",
    )
    def test_every_csv_row_has_a_definition(self):
        hypotheses = read_csv_hypotheses(CORPUS_CSV)
        lookup = {}
        with open(DEFINITIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lookup[row["target"]] = row["definition"]
                self.assertNotIn("category", row)
        self.assertEqual(len(hypotheses), len(lookup))
        for hypothesis in hypotheses:
            self.assertIn(hypothesis, lookup)
            self.assertTrue(lookup[hypothesis].strip())

    @unittest.skipUnless(
        os.path.isfile(WARMSTART_CSV) and os.path.isfile(WARMSTART_DEFINITIONS),
        "hypogen warmstart corpus not generated",
    )
    def test_warmstart_has_eight_generator_hypotheses(self):
        hypotheses = read_csv_hypotheses(WARMSTART_CSV)
        self.assertEqual(len(hypotheses), 8)
        self.assertTrue(
            hypotheses[0].startswith("Higher rates of speciation (BAMM_speciation)")
        )
        lookup = {}
        with open(WARMSTART_DEFINITIONS, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lookup[row["target"]] = row["definition"]
        self.assertEqual(list(lookup), hypotheses)


class PrepareCollectRowsTests(unittest.TestCase):
    def test_walks_nested_experiments_and_keeps_first_objective(self):
        nodes = [
            {
                "hypothesis": "H1",
                "experiment_plan": {"objective": "O1"},
                "untried_experiments": [
                    {
                        "hypothesis": "H2",
                        "experiment_plan": {"objective": "O2"},
                    }
                ],
            },
            {
                "hypothesis": "h1",
                "experiment_plan": {"objective": "O1-other"},
            },
        ]
        rows, n_collisions = collect_rows(nodes)
        self.assertEqual(
            rows,
            [
                {"hypothesis": "H1", "definition": "O1"},
                {"hypothesis": "H2", "definition": "O2"},
            ],
        )
        self.assertEqual(n_collisions, 1)
        self.assertEqual(hypothesis_identity_key("H1"), hypothesis_identity_key(" h1 "))

    def test_missing_objective_raises(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            collect_rows([{"hypothesis": "H1", "experiment_plan": {}}])

    def test_reads_experiment_generator_payload(self):
        messages = [
            {"name": "user_proxy", "content": "ignore"},
            {
                "name": "experiment_generator",
                "content": json.dumps(
                    {
                        "experiments": [
                            {
                                "hypothesis": "H-warm",
                                "experiment_plan": {"objective": "O-warm"},
                            }
                        ]
                    }
                ),
            },
        ]
        rows, n_collisions = collect_rows_from_generator_messages(messages)
        self.assertEqual(n_collisions, 0)
        self.assertEqual(
            rows, [{"hypothesis": "H-warm", "definition": "O-warm"}]
        )

    def test_generator_messages_require_the_named_role(self):
        with self.assertRaisesRegex(ValueError, "experiment_generator"):
            collect_rows_from_generator_messages(
                [{"name": "user_proxy", "content": "{}"}]
            )


if __name__ == "__main__":
    unittest.main()
