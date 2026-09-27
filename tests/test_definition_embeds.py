#!/usr/bin/env python3
"""Tests for definition-based embedding text formatting."""

from __future__ import annotations

import unittest

from boreft.task_config import definition_embedding_text
from boreft.text_similarity import (
    definition_embed_text,
    embedding_provenance,
    eval_embed_cache_provenance,
    format_text_for_cache_embedding,
    format_text_for_embedding,
    load_category_normalized_lookup,
    warn_if_embedding_provenance_stale,
)


class DefinitionEmbedTextTest(unittest.TestCase):
    def test_definition_embed_text_format(self) -> None:
        text = definition_embed_text("Computer", "An Electronic Machine.")
        self.assertEqual(
            text,
            "The meaning of 'Computer' is: An Electronic Machine.",
        )

    def test_definition_embedding_text_from_task_config(self) -> None:
        text = definition_embedding_text("semantle", "laptop", "a portable machine.")
        self.assertEqual(
            text,
            "The meaning of 'laptop' is: a portable machine.",
        )

    def test_cache_format_uses_lookup(self) -> None:
        lookup = {"laptop": "The meaning of 'laptop' is: a portable machine."}
        self.assertEqual(
            format_text_for_cache_embedding("laptop", lookup),
            "The meaning of 'laptop' is: a portable machine.",
        )

    def test_cache_format_falls_back_to_prompt(self) -> None:
        lookup = {"laptop": "The meaning of 'laptop' is: a portable machine."}
        self.assertEqual(
            format_text_for_cache_embedding("unknown", lookup),
            "The meaning of 'unknown'.",
        )

    def test_eval_format_always_uses_prompt(self) -> None:
        self.assertEqual(
            format_text_for_embedding("computer"),
            "The meaning of 'computer'.",
        )

    def test_provenance_differs_for_train_vs_eval_cache(self) -> None:
        train = embedding_provenance(
            use_definition_embeds=True, definitions_path="/tmp/d.jsonl"
        )
        eval_prov = eval_embed_cache_provenance()
        self.assertTrue(train["use_definition_embeds"])
        self.assertFalse(eval_prov["use_definition_embeds"])
        self.assertIn("embedding_prompt_defn", train)
        self.assertNotIn("embedding_prompt_defn", eval_prov)
        self.assertEqual(train["task"], "semantle")

    def test_warn_stale_definition_embed_provenance(self) -> None:
        import warnings

        saved = {
            "task": "semantle",
            "use_definition_embeds": True,
            "definitions_path": "/tmp/d.jsonl",
            "embedding_prompt_defn": "stale template",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_embedding_provenance_stale(saved, context="eval")
        self.assertEqual(len(caught), 1)
        self.assertIn("embedding_prompt_defn", str(caught[0].message))

    def test_warn_no_stale_when_definition_embed_provenance_matches(self) -> None:
        import warnings

        expected = embedding_provenance(
            use_definition_embeds=True, definitions_path="/tmp/d.jsonl"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_embedding_provenance_stale(expected, context="eval")
        self.assertEqual(len(caught), 0)

    def test_load_category_normalized_lookup(self) -> None:
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"target": "laptop", "category_normalized": "computing"}))
            f.write("\n")
            path = f.name
        lookup = load_category_normalized_lookup(path)
        self.assertEqual(lookup["laptop"], "computing")


if __name__ == "__main__":
    unittest.main()
