"""Tests for ``--bias-network-embed-model``.

The flag lets the bias network read a different sentence-transformer than the one
the task's similarity metrics use, so the invariant under test throughout is that
the *training* embed cache follows the override while everything reported as
``embed_sim`` stays on the task's own model.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from boreft.bias_tables import bias_predict_kwargs
from boreft.embed_cache import resolve_or_build_embed_cache
from boreft.text_similarity import (
    bias_network_embed_model_from_cfg,
    embedding_model_name,
    embedding_provenance,
    training_cache_params,
    training_embed_cache_provenance_from_cfg,
    warn_if_embedding_provenance_stale,
)
from boreft.train_args import TrainConfig

CHEMATE = "SchwallerGroup/CheMatE-v0"


def _molopt_config(**kwargs) -> TrainConfig:
    base = dict(
        task="molopt",
        molopt_csv="mols.csv",
        add_bias_network=True,
        use_word_bias=True,
    )
    base.update(kwargs)
    return TrainConfig(**base)


class FlagValidationTests(unittest.TestCase):
    def test_accepted_with_sentence_transformer_bias_network(self):
        cfg = _molopt_config(bias_network_embed_model=CHEMATE)
        self.assertEqual(cfg.bias_network_embed_model, CHEMATE)

    def test_defaults_to_none(self):
        self.assertIsNone(_molopt_config().bias_network_embed_model)

    def test_rejected_without_bias_network(self):
        with self.assertRaisesRegex(ValueError, "--add-bias-network"):
            _molopt_config(
                add_bias_network=False, bias_network_embed_model=CHEMATE
            )

    def test_rejected_for_llm_encoder(self):
        with self.assertRaisesRegex(ValueError, "sentence_transformer"):
            _molopt_config(
                bias_network_encoder="llm_encoder",
                bias_network_embed_model=CHEMATE,
            )

    def test_rejected_when_blank(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _molopt_config(bias_network_embed_model="   ")


class ProvenanceTests(unittest.TestCase):
    def test_training_cache_provenance_records_the_override(self):
        prov, _, _ = training_cache_params(
            use_definition_embeds=False,
            task="molopt",
            words=["CCO"],
            model_name=CHEMATE,
        )
        self.assertEqual(prov["sentence_transformer_model"], CHEMATE)

    def test_training_cache_provenance_defaults_to_task_model(self):
        prov, _, _ = training_cache_params(
            use_definition_embeds=False, task="molopt", words=["CCO"]
        )
        self.assertEqual(
            prov["sentence_transformer_model"], embedding_model_name("molopt")
        )

    def test_eval_provenance_ignores_the_override(self):
        """Similarity metrics must stay comparable to runs without the flag."""
        prov = embedding_provenance(task="molopt")
        self.assertEqual(
            prov["sentence_transformer_model"], embedding_model_name("molopt")
        )

    def test_override_changes_the_cache_identity(self):
        default, _, _ = training_cache_params(
            use_definition_embeds=False, task="molopt", words=["CCO"]
        )
        overridden, _, _ = training_cache_params(
            use_definition_embeds=False,
            task="molopt",
            words=["CCO"],
            model_name=CHEMATE,
        )
        self.assertNotEqual(default, overridden)


class ProvenanceRoundTripTests(unittest.TestCase):
    """A checkpoint must rebuild the exact cache its bias network was trained on."""

    def test_saved_override_is_reconstructed(self):
        cfg = {"task": "molopt", "bias_network_embed_model": CHEMATE}
        self.assertEqual(bias_network_embed_model_from_cfg(cfg), CHEMATE)
        self.assertEqual(
            training_embed_cache_provenance_from_cfg(cfg)[
                "sentence_transformer_model"
            ],
            CHEMATE,
        )

    def test_absent_override_falls_back_to_the_task_model(self):
        cfg = {"task": "molopt"}
        self.assertIsNone(bias_network_embed_model_from_cfg(cfg))
        self.assertEqual(
            training_embed_cache_provenance_from_cfg(cfg)[
                "sentence_transformer_model"
            ],
            embedding_model_name("molopt"),
        )

    def test_predict_kwargs_carry_the_override(self):
        cfg = {"task": "molopt", "bias_network_embed_model": CHEMATE}
        self.assertEqual(
            bias_predict_kwargs(cfg, tokenizer=None), {"embed_model": CHEMATE}
        )


class StalenessWarningTests(unittest.TestCase):
    def test_override_run_is_not_reported_stale(self):
        """The saved model is the bias network's, so eval must not expect the task's."""
        cfg = {
            "task": "molopt",
            "bias_network_embed_model": CHEMATE,
            "sentence_transformer_model": CHEMATE,
            "embedding_prompt": embedding_provenance(task="molopt")[
                "embedding_prompt"
            ],
            "normalize_embeddings": True,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_embedding_provenance_stale(cfg)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_genuine_drift_is_still_reported(self):
        cfg = {
            "task": "molopt",
            "bias_network_embed_model": CHEMATE,
            "sentence_transformer_model": CHEMATE,
            "embedding_prompt": "stale prompt {text}",
            "normalize_embeddings": True,
        }
        with self.assertWarnsRegex(UserWarning, "embedding_prompt"):
            warn_if_embedding_provenance_stale(cfg)


class CacheEncoderTests(unittest.TestCase):
    def test_cache_is_encoded_with_the_model_it_is_labelled_with(self):
        """A cache tagged with one model but encoded by another would be silent poison."""
        prov = embedding_provenance(model_name=CHEMATE, task="molopt")
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch(
            "boreft.embed_cache.encode_reference_embeddings",
            return_value=np.zeros((1, 4)),
        ) as mock_encode:
            resolve_or_build_embed_cache(
                ["CCO"], cache_dir=cache_dir, provenance=prov
            )
        self.assertEqual(mock_encode.call_args.kwargs["model_name"], CHEMATE)

    def test_task_default_cache_passes_no_model_override(self):
        prov = embedding_provenance(task="molopt")
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch(
            "boreft.embed_cache.encode_reference_embeddings",
            return_value=np.zeros((1, 4)),
        ) as mock_encode:
            resolve_or_build_embed_cache(
                ["CCO"], cache_dir=cache_dir, provenance=prov
            )
        self.assertEqual(
            mock_encode.call_args.kwargs["model_name"], embedding_model_name("molopt")
        )


class EncodeDispatchTests(unittest.TestCase):
    def test_model_name_selects_the_loaded_encoder(self):
        from boreft import text_similarity

        loaded: list[str] = []

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                return np.zeros((len(texts), 3), dtype=np.float32)

        def fake_get(name: str):
            loaded.append(name)
            return FakeModel()

        with mock.patch.object(text_similarity, "_get_embed_model", fake_get):
            text_similarity.encode_texts_normalized(["CCO"], task="molopt")
            text_similarity.encode_reference_embeddings(
                ["CCO"], task="molopt", model_name=CHEMATE
            )

        self.assertEqual(loaded, [embedding_model_name("molopt"), CHEMATE])


if __name__ == "__main__":
    unittest.main()
