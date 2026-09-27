"""Tests for embed_cache resolve/save/load."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from boreft.embed_cache import (
    EMBED_CACHE_INDEX_FILENAME,
    EMBED_CACHE_PT_FILENAME,
    bias_network_input_dim_from_cfg,
    cache_entry_dir,
    compute_vocab_key,
    expected_texts_for_rows,
    load_embed_cache_tensor,
    load_index_for_pt,
    resolve_or_build_embed_cache,
    save_embed_cache,
    validate_index,
)


class EmbedCacheTest(unittest.TestCase):
    def test_vocab_key_changes_with_texts(self):
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": True,
        }
        k1 = compute_vocab_key(["a", "b"], prov)
        k2 = compute_vocab_key(["a", "c"], prov)
        self.assertNotEqual(k1, k2)

    def test_cache_entry_dir_uses_full_vocab_key(self):
        key = "a" * 64
        self.assertTrue(cache_entry_dir(key).endswith(key))

    def test_expected_texts_for_rows_indices(self):
        words = ["w0", "w1", "w2", "w3"]
        self.assertEqual(
            expected_texts_for_rows(words, indices=[2, 0]),
            ["w2", "w0"],
        )

    def test_save_load_roundtrip(self):
        texts = ["alpha", "beta", "gamma"]
        emb = torch.randn(3, 8)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            index = save_embed_cache(tmp, emb, texts, provenance=prov)
            pt_path = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            with mock.patch(
                "boreft.embed_cache.embedding_provenance", return_value=prov
            ):
                loaded = load_embed_cache_tensor(
                    pt_path, expected_texts=texts, index=index
                )
            self.assertTrue(torch.allclose(emb, loaded, atol=1e-6))
            json_path = os.path.join(tmp, EMBED_CACHE_INDEX_FILENAME)
            self.assertTrue(os.path.isfile(json_path))
            with open(json_path, encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["texts"], texts)
            self.assertIsNone(validate_index(on_disk, texts, prov))

    def test_load_requires_sibling_index(self):
        texts = ["one"]
        emb = torch.randn(1, 4)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_embed_cache(tmp, emb, texts, provenance=prov)
            pt_path = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            os.remove(os.path.join(tmp, EMBED_CACHE_INDEX_FILENAME))
            with self.assertRaises(FileNotFoundError):
                load_index_for_pt(pt_path, required=True)

    def test_load_rejects_text_mismatch(self):
        texts = ["alpha", "beta"]
        emb = torch.randn(2, 4)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            index = save_embed_cache(tmp, emb, texts, provenance=prov)
            pt_path = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            with self.assertRaises(ValueError):
                load_embed_cache_tensor(
                    pt_path,
                    expected_texts=["alpha", "gamma"],
                    index=index,
                )

    def test_load_with_indices_slice(self):
        texts = ["a", "b", "c", "d"]
        emb = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            index = save_embed_cache(tmp, emb, texts, provenance=prov)
            pt_path = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            with mock.patch(
                "boreft.embed_cache.embedding_provenance", return_value=prov
            ):
                loaded = load_embed_cache_tensor(
                    pt_path,
                    indices=[3, 1],
                    expected_texts=["d", "b"],
                    index=index,
                )
            self.assertTrue(torch.equal(loaded, emb[[3, 1]]))

    def test_validate_index_detects_text_mismatch(self):
        index = {
            "texts": ["one", "two"],
            "sentence_transformer_model": "m",
            "embedding_prompt": "p {text}",
            "normalize_embeddings": True,
            "vocab_key": compute_vocab_key(
                ["one", "two"],
                {
                    "sentence_transformer_model": "m",
                    "embedding_prompt": "p {text}",
                    "use_definition_embeds": False,
                    "definitions_path": "",
                    "normalize_embeddings": True,
                },
            ),
        }
        reason = validate_index(index, ["one", "three"])
        self.assertEqual(reason, "texts differ")

    def test_resolve_or_build_cache_hit(self):
        texts = ["cat", "dog"]
        emb = torch.randn(2, 6)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as cache_dir:
            vocab_key = compute_vocab_key(texts, prov)
            entry_dir = cache_entry_dir(vocab_key, cache_dir)
            save_embed_cache(entry_dir, emb, texts, provenance=prov)

            with (
                mock.patch(
                    "boreft.embed_cache.encode_reference_embeddings"
                ) as mock_encode,
            ):
                tensor, path, index = resolve_or_build_embed_cache(
                    texts, cache_dir=cache_dir, provenance=prov
                )
                mock_encode.assert_not_called()

            self.assertTrue(os.path.isfile(path))
            self.assertEqual(index["texts"], texts)
            self.assertEqual(tensor.shape[0], 2)

    def test_resolve_or_build_explicit_path_requires_index(self):
        texts = ["x"]
        emb = torch.randn(1, 3)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            pt_only = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            torch.save({"embeddings": emb, "num_rows": 1, "embed_dim": 3}, pt_only)
            with self.assertRaises(FileNotFoundError):
                resolve_or_build_embed_cache(
                    texts, explicit_path=pt_only, cache_dir=tmp
                )


class BiasNetworkInputDimTests(unittest.TestCase):
    def test_prefers_input_dim_then_embed_dim(self):
        self.assertEqual(
            bias_network_input_dim_from_cfg({"bias_network_embed_dim": 1024}),
            1024,
        )
        self.assertEqual(
            bias_network_input_dim_from_cfg(
                {
                    "bias_network_input_dim": 2048,
                    "bias_network_embed_dim": 1024,
                }
            ),
            2048,
        )
        self.assertIsNone(bias_network_input_dim_from_cfg({}))

    def test_reads_on_disk_cache_width(self):
        texts = ["one"]
        emb = torch.randn(1, 12)
        prov = {
            "sentence_transformer_model": "test-model",
            "embedding_prompt": "q {text}",
            "use_definition_embeds": False,
            "definitions_path": "",
            "normalize_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_embed_cache(tmp, emb, texts, provenance=prov)
            pt_path = os.path.join(tmp, EMBED_CACHE_PT_FILENAME)
            self.assertEqual(
                bias_network_input_dim_from_cfg({}, embed_cache_path=pt_path),
                12,
            )


if __name__ == "__main__":
    unittest.main()
