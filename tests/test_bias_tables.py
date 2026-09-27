"""Tests for bias-network materialization and eval table lookup."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from boreft.bias_tables import (
    _intervention_from_reft_model,
    attach_materialized_bias_tables,
    bias_predict_kwargs,
    detach_materialized_bias_tables,
    export_bias_tables_for_checkpoint,
    load_bias_tables,
    materialize_bias_tables,
    predict_bias_vectors_for_words,
    predict_bias_vectors_from_raw_texts,
    save_bias_tables,
    stack_bias_mu_std,
    stack_bias_vectors,
    training_search_domain,
)
from boreft.pyreft import ReftConfig, build_reft_representation, get_reft_model
from boreft.pyreft.interventions import DistributionalWordIntervention
from transformers import AutoModelForCausalLM


def _tiny_reft_with_bias_network(*, variance: str | float = "learnable"):
    embed_cache = torch.randn(8, 16)
    intervention = DistributionalWordIntervention(
        embed_dim=32,
        low_rank_dimension=4,
        num_words=8,
        add_bias_network=True,
        embed_cache=embed_cache,
        variance=variance,
        dtype=torch.float32,
        device="cpu",
    )
    base = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    reft_config = ReftConfig(
        representations=[build_reft_representation(0, 4, intervention)]
    )
    return get_reft_model(base, reft_config, set_device=True)


class BiasTablesTest(unittest.TestCase):
    def test_materialized_mu_matches_network(self):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)

        mu_tables, logvar_tables, _ = materialize_bias_tables(iv, num_words=8)
        attach_materialized_bias_tables(iv, mu_tables, logvar_tables)

        ids = torch.tensor([0, 3, 7], dtype=torch.long)
        mu_net = iv.bias_network.mu(iv.embed_cache[ids])
        mu_tab = iv.get_bias_mu(ids)
        self.assertTrue(torch.allclose(mu_net, mu_tab, atol=1e-5))

        mu_lv, logvar_lv = iv.get_bias_mu_logvar(ids)
        _, logvar_net = iv.bias_network(iv.embed_cache[ids])
        self.assertTrue(torch.allclose(mu_net, mu_lv, atol=1e-5))
        self.assertTrue(torch.allclose(logvar_net, logvar_lv, atol=1e-5))

    def test_stack_bias_mu_std_matches_clamped_logvar(self):
        from boreft.pyreft.losses import clamp_logvar

        reft_model = _tiny_reft_with_bias_network()
        ids = [0, 3, 7]
        mu, std = stack_bias_mu_std(reft_model, ids)
        np.testing.assert_allclose(
            mu, stack_bias_vectors(reft_model, ids), atol=1e-5
        )
        iv = _intervention_from_reft_model(reft_model)
        _, logvar = iv.get_bias_mu_logvar(torch.tensor(ids, dtype=torch.long))
        expected = torch.exp(0.5 * clamp_logvar(logvar.float())).detach().cpu().numpy()
        np.testing.assert_allclose(std, expected, atol=1e-5)
        self.assertTrue(np.all(std > 0))

    def test_stack_bias_mu_std_restores_train_mode(self):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        iv.train()
        stack_bias_mu_std(reft_model, [0, 1])
        self.assertTrue(iv.training)

    def test_training_search_domain_std_k_zero_matches_mean_box(self):
        from boreft.bo.acquisition import latent_bounds

        reft_model = _tiny_reft_with_bias_network()
        ids = list(range(8))
        mu, bounds = training_search_domain(reft_model, ids, aabb_std_k=0.0)
        np.testing.assert_allclose(mu, stack_bias_vectors(reft_model, ids), atol=1e-5)
        np.testing.assert_allclose(bounds, latent_bounds(mu), atol=1e-5)

    def test_training_search_domain_expands_with_std_k(self):
        from boreft.bo.acquisition import latent_bounds

        reft_model = _tiny_reft_with_bias_network()
        ids = list(range(8))
        mu, std = stack_bias_mu_std(reft_model, ids)
        _, bounds = training_search_domain(reft_model, ids, aabb_std_k=2.0)
        np.testing.assert_allclose(
            bounds, latent_bounds(mu, std=std, std_k=2.0), atol=1e-5
        )
        mean_box = latent_bounds(mu)
        self.assertTrue(np.all(bounds[0] <= mean_box[0] + 1e-6))
        self.assertTrue(np.all(bounds[1] >= mean_box[1] - 1e-6))

    def test_training_search_domain_ellipsoid_encloses_means(self):
        from boreft.bo.acquisition import latent_ellipsoid

        reft_model = _tiny_reft_with_bias_network()
        ids = list(range(8))
        mu, bounds = training_search_domain(
            reft_model, ids, search_domain="ellipsoid"
        )
        ell = latent_ellipsoid(mu)
        np.testing.assert_allclose(bounds, ell.aabb(), atol=1e-5)
        self.assertTrue(np.all(ell.contains(mu)))
        with self.assertRaisesRegex(ValueError, "search_domain"):
            training_search_domain(reft_model, ids, search_domain="sphere")

    def test_export_without_attach_leaves_network_path(self):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)

        with tempfile.TemporaryDirectory() as tmp:
            path = export_bias_tables_for_checkpoint(
                reft_model,
                tmp,
                num_words=8,
                update_config=False,
                attach_in_memory=False,
            )
            self.assertIsNotNone(path)
            self.assertFalse(hasattr(iv, "materialized_mu"))

            ids = torch.tensor([1, 2], dtype=torch.long)
            mu_before = iv.bias_network.mu(iv.embed_cache[ids]).clone()
            mu_lookup = iv.get_bias_mu(ids)
            self.assertTrue(torch.allclose(mu_before, mu_lookup, atol=1e-5))

    def test_fixed_variance_skips_logvar_in_file(self):
        reft_model = _tiny_reft_with_bias_network(variance=0.01)
        iv = _intervention_from_reft_model(reft_model)

        mu, logvar, meta = materialize_bias_tables(iv, num_words=8)
        self.assertIsNone(logvar)
        self.assertIn("fixed_logvar", meta)

        with tempfile.TemporaryDirectory() as tmp:
            fpath = f"{tmp}/bias_tables.pt"
            save_bias_tables(fpath, mu, logvar, metadata=meta)
            mu_l, logvar_l, meta_l = load_bias_tables(fpath)
            self.assertIsNone(logvar_l)
            self.assertIn("fixed_logvar", meta_l)

    def test_detach_clears_materialized_buffers(self):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        mu, logvar, _ = materialize_bias_tables(iv, num_words=8)
        attach_materialized_bias_tables(iv, mu, logvar)
        self.assertTrue(hasattr(iv, "materialized_mu"))
        detach_materialized_bias_tables(iv)
        self.assertFalse(hasattr(iv, "materialized_mu"))

    @patch("boreft.bias_tables.encode_reference_embeddings")
    def test_predict_bias_vectors_matches_network(self, mock_encode):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        words = ["w0", "w3"]
        mock_encode.return_value = iv.embed_cache[[0, 3]].numpy()

        vecs = predict_bias_vectors_for_words(reft_model, words)
        expected = iv.bias_network.mu(iv.embed_cache[[0, 3]]).detach().numpy()
        np.testing.assert_allclose(np.stack(vecs), expected, atol=1e-5)

    @patch("boreft.bias_tables.encode_texts_as_is")
    def test_predict_bias_vectors_from_raw_texts(self, mock_encode):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        raw = "arbitrary phrase"
        mock_encode.return_value = iv.embed_cache[[1]].numpy()

        vecs = predict_bias_vectors_from_raw_texts(reft_model, [raw])
        mock_encode.assert_called_once_with([raw], task="semantle", model_name=None)
        expected = iv.bias_network.mu(iv.embed_cache[[1]]).detach().numpy()
        np.testing.assert_allclose(np.stack(vecs), expected, atol=1e-5)

    @patch("boreft.bias_tables.encode_texts_as_is")
    def test_predict_bias_vectors_from_raw_texts_uses_task_model(self, mock_encode):
        """The bias network only accepts embeddings from the model it trained on."""
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        mock_encode.return_value = iv.embed_cache[[1]].numpy()

        predict_bias_vectors_from_raw_texts(reft_model, ["CCO"], task="molopt")
        mock_encode.assert_called_once_with(["CCO"], task="molopt", model_name=None)

    @patch("boreft.bias_tables.encode_texts_as_is")
    def test_raw_texts_embed_model_overrides_task_model(self, mock_encode):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        mock_encode.return_value = iv.embed_cache[[1]].numpy()

        predict_bias_vectors_from_raw_texts(
            reft_model, ["CCO"], task="molopt", embed_model="SchwallerGroup/CheMatE-v0"
        )
        mock_encode.assert_called_once_with(
            ["CCO"], task="molopt", model_name="SchwallerGroup/CheMatE-v0"
        )

    @patch("boreft.bias_tables.encode_reference_embeddings")
    def test_words_embed_model_overrides_task_model(self, mock_encode):
        reft_model = _tiny_reft_with_bias_network()
        iv = _intervention_from_reft_model(reft_model)
        mock_encode.return_value = iv.embed_cache[[0]].numpy()

        predict_bias_vectors_for_words(
            reft_model, ["CCO"], task="molopt", embed_model="SchwallerGroup/CheMatE-v0"
        )
        self.assertEqual(
            mock_encode.call_args.kwargs["model_name"], "SchwallerGroup/CheMatE-v0"
        )


class BiasPredictKwargsTest(unittest.TestCase):
    """bias_predict_kwargs must reproduce each run's bias-network input space."""

    def test_plain_sentence_transformer_run_needs_nothing(self):
        self.assertEqual(bias_predict_kwargs({"task": "molopt"}, tokenizer=None), {})

    def test_overridden_encoder_is_passed_through(self):
        cfg = {"task": "molopt", "bias_network_embed_model": "SchwallerGroup/CheMatE-v0"}
        self.assertEqual(
            bias_predict_kwargs(cfg, tokenizer=None),
            {"embed_model": "SchwallerGroup/CheMatE-v0"},
        )

    def test_llm_encoder_run_ignores_embed_model(self):
        cfg = {
            "task": "semantle",
            "bias_input_source": "llm_encoder",
            "bias_encoder_max_length": 32,
            "bias_encoder_layer_index": 5,
        }
        kwargs = bias_predict_kwargs(
            cfg, tokenizer="tok", raw_definition_lookup={"a": "b"}
        )
        self.assertNotIn("embed_model", kwargs)
        self.assertEqual(kwargs["encoder_max_length"], 32)
        self.assertEqual(kwargs["encoder_layer_index"], 5)

    def test_predict_bias_vectors_from_raw_texts_llm_encoder(self):
        from unittest.mock import MagicMock, patch

        penult = torch.randn(1, 5, 16)
        attn = torch.ones(1, 5)
        encoder = MagicMock(return_value=torch.randn(1, 8))
        encoder.pooling = "last_instruction"
        iv = DistributionalWordIntervention(
            embed_dim=16,
            low_rank_dimension=4,
            num_words=3,
            add_bias_network=True,
            bias_input_source="llm_encoder",
            bias_network_input_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        iv.set_semantic_encoder(encoder)
        iv.set_encoder_inputs(penult, attn)
        base = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        reft_model = get_reft_model(
            base,
            ReftConfig(representations=[build_reft_representation(0, 4, iv)]),
            set_device=True,
        )
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kw: {
            "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
        }

        with patch(
            "boreft.pyreft.semantic_encoder.capture_penultimate",
            return_value=penult[:1],
        ), patch(
            "boreft.pyreft.semantic_encoder.encode_definition_inputs",
            return_value=(
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                None,
            ),
        ):
            vecs = predict_bias_vectors_from_raw_texts(
                reft_model,
                ["probe text"],
                tokenizer=tokenizer,
            )
        encoder.assert_called_once()
        self.assertEqual(len(vecs), 1)
        self.assertEqual(vecs[0].shape, (4,))


if __name__ == "__main__":
    unittest.main()
