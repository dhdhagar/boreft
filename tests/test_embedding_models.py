"""Tests for the per-task embedding model wiring in ``boreft.text_similarity``.

The pretrained encoders are stubbed: what matters here is that every task follows
the one sentence-transformers path (which model, which prompt decoration, one cached
model per model name) and that provenance still records what a checkpoint was built
with, so a changed model invalidates the embed cache instead of silently mixing
spaces.
"""

from __future__ import annotations

import hashlib
import unittest
import warnings
from unittest import mock

import numpy as np

from boreft import text_similarity
from boreft.task_config import (
    DEFAULT_EMBEDDING_MODEL,
    task_config,
    task_embedding_model,
)
from boreft.text_similarity import (
    embedding_model_name,
    embedding_provenance,
    encode_texts_as_is,
    encode_texts_normalized,
    format_text_for_embedding,
    warn_if_embedding_provenance_stale,
)

SEMANTLE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Both tasks currently share one encoder (see notes/embedding_models.md), so the
# per-task-model machinery is exercised against a fake task rather than molopt.
MOLOPT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
OTHER_MODEL = "some-org/other-encoder"
OTHER_TASK = {
    "target_kind": "text",
    "embedding_model": OTHER_MODEL,
    "embedding_prompt": "The meaning of '{text}'.",
}


class _FakeSentenceTransformer:
    """Records the strings it was asked to encode; returns deterministic unit rows."""

    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.encoded: list[str] = []
        type(self).instances.append(self)

    def encode(self, texts, normalize_embeddings=True):
        self.encoded.extend(texts)
        rows = []
        for text in texts:
            digest = hashlib.sha1(text.encode("utf-8")).digest()[:8]
            vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) + 1.0
            rows.append(vec / np.linalg.norm(vec))
        return np.asarray(rows, dtype=np.float32)


class _FakeModelMixin:
    """Patches the encoder class and clears the per-model cache around each test."""

    def setUp(self):
        super().setUp()
        _FakeSentenceTransformer.instances = []
        patcher = mock.patch.object(
            text_similarity, "SentenceTransformer", _FakeSentenceTransformer
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        previous = dict(text_similarity._EMBED_MODEL_CACHE)
        text_similarity._EMBED_MODEL_CACHE.clear()
        self.addCleanup(
            lambda: (
                text_similarity._EMBED_MODEL_CACHE.clear(),
                text_similarity._EMBED_MODEL_CACHE.update(previous),
            )
        )


class TaskModelSelectionTests(unittest.TestCase):
    def test_each_task_names_its_own_encoder(self):
        self.assertEqual(embedding_model_name("semantle"), SEMANTLE_MODEL)
        self.assertEqual(embedding_model_name("molopt"), MOLOPT_MODEL)

    def test_both_tasks_currently_name_the_same_encoder(self):
        """Pins the decision to score molecules with the semantle text encoder."""
        self.assertEqual(
            embedding_model_name("molopt"), embedding_model_name("semantle")
        )

    def test_a_task_can_still_override_the_model(self):
        """The per-task lookup must keep working now that nothing exercises it."""
        with mock.patch.dict(task_config, {"other": OTHER_TASK}):
            self.assertEqual(embedding_model_name("other"), OTHER_MODEL)

    def test_unknown_tasks_fall_back_to_the_default_model(self):
        self.assertEqual(task_embedding_model("arc"), DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, SEMANTLE_MODEL)

    def test_both_tasks_decorate_their_target_with_a_prompt(self):
        """molopt SMILES go through the same prompt path as Semantle words."""
        word = format_text_for_embedding("cat", task="semantle")
        smiles = format_text_for_embedding("CCO", task="molopt")
        self.assertIn("cat", word)
        self.assertNotEqual(word, "cat")
        self.assertIn("CCO", smiles)
        self.assertNotEqual(smiles, "CCO")

    def test_decoration_strips_surrounding_whitespace(self):
        self.assertEqual(
            format_text_for_embedding("  CCO  ", task="molopt"),
            format_text_for_embedding("CCO", task="molopt"),
        )


class EncodingTests(_FakeModelMixin, unittest.TestCase):
    def test_a_task_encodes_with_its_own_model(self):
        encode_texts_normalized(["CCO"], task="molopt")
        self.assertEqual(
            [m.model_name for m in _FakeSentenceTransformer.instances], [MOLOPT_MODEL]
        )

    def test_tasks_sharing_a_model_load_it_once(self):
        encode_texts_normalized(["cat"], task="semantle")
        encode_texts_normalized(["CCO"], task="molopt")
        encode_texts_normalized(["dog"], task="semantle")
        self.assertEqual(
            [m.model_name for m in _FakeSentenceTransformer.instances], [SEMANTLE_MODEL]
        )

    def test_distinct_models_are_cached_separately(self):
        with mock.patch.dict(task_config, {"other": OTHER_TASK}):
            encode_texts_normalized(["cat"], task="semantle")
            encode_texts_normalized(["cat"], task="other")
            encode_texts_normalized(["dog"], task="semantle")
        self.assertEqual(
            sorted(m.model_name for m in _FakeSentenceTransformer.instances),
            sorted([OTHER_MODEL, SEMANTLE_MODEL]),
        )

    def test_normalized_encoding_passes_the_decorated_string(self):
        encode_texts_normalized(["CCO"], task="molopt")
        (model,) = _FakeSentenceTransformer.instances
        self.assertEqual(
            model.encoded, [format_text_for_embedding("CCO", task="molopt")]
        )

    def test_as_is_encoding_passes_the_string_verbatim(self):
        encode_texts_as_is(["CCO"], task="molopt")
        (model,) = _FakeSentenceTransformer.instances
        self.assertEqual(model.encoded, ["CCO"])

    def test_rows_are_l2_normalized(self):
        emb = encode_texts_normalized(["CCO", "c1ccccc1"], task="molopt")
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), [1.0, 1.0], atol=1e-6)

    def test_empty_input_returns_no_rows_without_loading_a_model(self):
        self.assertEqual(encode_texts_normalized([], task="molopt").shape[0], 0)
        self.assertEqual(encode_texts_as_is([], task="molopt").shape[0], 0)
        self.assertEqual(_FakeSentenceTransformer.instances, [])


class ProvenanceTests(unittest.TestCase):
    def test_each_task_records_the_model_it_embeds_with(self):
        self.assertEqual(
            embedding_provenance(task="semantle")["sentence_transformer_model"],
            SEMANTLE_MODEL,
        )
        self.assertEqual(
            embedding_provenance(task="molopt")["sentence_transformer_model"],
            MOLOPT_MODEL,
        )

    def test_both_tasks_record_the_prompt_they_embed_with(self):
        for task in ("semantle", "molopt", "hypogen"):
            prov = embedding_provenance(task=task)
            self.assertIn("{text}", prov["embedding_prompt"], msg=task)

    def test_backend_dispatch_fields_are_gone(self):
        """The dispatch was removed; leftover keys would break cache validation."""
        prov = embedding_provenance(task="molopt")
        self.assertNotIn("embedding_backend", prov)
        self.assertNotIn("embedding_model", prov)

    def test_matching_config_is_not_reported_stale(self):
        for task in ("semantle", "molopt", "hypogen"):
            saved = embedding_provenance(task=task)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                warn_if_embedding_provenance_stale(saved, context="eval")

    def test_changed_model_is_reported_stale(self):
        saved = {
            **embedding_provenance(task="molopt"),
            "sentence_transformer_model": "other/model",
        }
        with self.assertWarnsRegex(UserWarning, "sentence_transformer_model"):
            warn_if_embedding_provenance_stale(saved, context="eval")

    def test_changed_prompt_is_reported_stale(self):
        saved = {
            **embedding_provenance(task="molopt"),
            "embedding_prompt": "{text}",
        }
        with self.assertWarnsRegex(UserWarning, "embedding_prompt"):
            warn_if_embedding_provenance_stale(saved, context="eval")


class DefinitionsPathTests(unittest.TestCase):
    def test_each_task_has_its_own_default_definitions_file(self):
        semantle = text_similarity.default_definitions_path("semantle")
        molopt = text_similarity.default_definitions_path("molopt")
        self.assertTrue(semantle.endswith("data/semantle/train/definitions.jsonl"))
        self.assertTrue(molopt.endswith("data/molopt/train/definitions.jsonl"))
        hypogen = text_similarity.default_definitions_path("hypogen")
        self.assertTrue(
            hypogen.endswith("data/hypogen/evo-fresh-fish/definitions.jsonl")
        )

    def test_the_default_task_is_semantle(self):
        self.assertEqual(
            text_similarity.default_definitions_path(),
            text_similarity.default_definitions_path("semantle"),
        )


if __name__ == "__main__":
    unittest.main()
