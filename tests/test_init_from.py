"""Tests for --init-from learnable-parameter initialization."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from boreft.data_utils import (
    apply_init_from_state_dict,
    init_learnable_parameters_from_run,
    learnable_parameter_spec,
    load_intervenable_state_dict,
)
from boreft.pyreft.interventions import (
    DistributionalWordIntervention,
    LoreftPerWordBiasIntervention,
)
from boreft.train_args import TrainConfig


def _vae_intervention(
    *,
    num_words: int = 8,
    low_rank: int = 4,
    embed_dim: int = 32,
    add_bias_network: bool = False,
    variance="learnable",
    embed_cache=None,
):
    kwargs = dict(
        embed_dim=embed_dim,
        low_rank_dimension=low_rank,
        num_words=num_words,
        dtype=torch.float32,
        device="cpu",
        variance=variance,
    )
    if add_bias_network:
        cache = embed_cache if embed_cache is not None else torch.randn(num_words, 16)
        kwargs.update(add_bias_network=True, embed_cache=cache)
    return DistributionalWordIntervention(**kwargs)


def _linear_intervention(*, num_words: int = 8, low_rank: int = 4, embed_dim: int = 32):
    return LoreftPerWordBiasIntervention(
        embed_dim=embed_dim,
        low_rank_dimension=low_rank,
        num_words=num_words,
        dtype=torch.float32,
        device="cpu",
    )


def _write_dummy_init_from_dir(root: str, *, extra_bins: int = 0) -> str:
    os.makedirs(os.path.join(root, "intervenable_model"), exist_ok=True)
    torch.save(
        {},
        os.path.join(root, "intervenable_model", "intkey_dummy.bin"),
    )
    for i in range(extra_bins):
        torch.save(
            {},
            os.path.join(root, "intervenable_model", f"intkey_extra_{i}.bin"),
        )
    return root


def _perturb(module, scale: float = 0.5) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if param.requires_grad:
                param.add_(scale)


class ApplyInitFromStateDictTests(unittest.TestCase):
    def test_copies_learnable_params_and_rotation(self):
        src = _vae_intervention()
        dst = _vae_intervention()
        _perturb(src)
        n = apply_init_from_state_dict(dst, src.state_dict(), source="src")
        self.assertEqual(n, len(learnable_parameter_spec(src)))
        for name, param in src.named_parameters():
            if not param.requires_grad:
                continue
            other = dict(dst.named_parameters())[name]
            self.assertTrue(torch.allclose(param, other), name)
        self.assertTrue(
            torch.allclose(src.rotate_layer.weight, dst.rotate_layer.weight, atol=1e-5)
        )

    def test_does_not_copy_embed_cache(self):
        src_cache = torch.randn(8, 16)
        dst_cache = torch.zeros(8, 16)
        src = _vae_intervention(add_bias_network=True, embed_cache=src_cache)
        dst = _vae_intervention(
            add_bias_network=True, embed_cache=dst_cache.clone()
        )
        apply_init_from_state_dict(dst, src.state_dict(), source="src")
        self.assertTrue(torch.equal(dst.embed_cache, dst_cache))
        self.assertFalse(torch.equal(dst.embed_cache, src.embed_cache))

    def test_incompatible_rank(self):
        src = _vae_intervention(low_rank=4)
        dst = _vae_intervention(low_rank=8)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_init_from_state_dict(dst, src.state_dict(), source="src")

    def test_incompatible_vocab_size(self):
        src = _vae_intervention(num_words=8)
        dst = _vae_intervention(num_words=16)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_init_from_state_dict(dst, src.state_dict(), source="src")

    def test_incompatible_bias_network_vs_tables(self):
        src = _vae_intervention(add_bias_network=True)
        dst = _vae_intervention(add_bias_network=False)
        with self.assertRaisesRegex(ValueError, "incompatible") as ctx:
            apply_init_from_state_dict(dst, src.state_dict(), source="src")
        err = str(ctx.exception)
        self.assertIn("missing from source", err)
        self.assertIn("word_mu.weight", err)
        self.assertIn("extra in source", err)
        self.assertIn("bias_network.fc1.weight", err)

    def test_incompatible_linear_vs_vae(self):
        src = _linear_intervention()
        dst = _vae_intervention()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_init_from_state_dict(dst, src.state_dict(), source="src")

    def test_incompatible_learnable_vs_fixed_variance(self):
        src = _vae_intervention(variance="learnable")
        dst = _vae_intervention(variance=0.01)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_init_from_state_dict(dst, src.state_dict(), source="src")

    def test_source_embed_cache_is_not_extra_learnable(self):
        src = _vae_intervention()
        _perturb(src)
        dst = _vae_intervention()
        saved = dict(src.state_dict())
        saved["embed_cache"] = torch.randn(8, 16)
        apply_init_from_state_dict(dst, saved, source="src")
        self.assertTrue(torch.equal(src.word_mu.weight, dst.word_mu.weight))
        self.assertIsNone(getattr(dst, "embed_cache", None))

    def test_missing_rotation_buffer_is_incompatible(self):
        src = _vae_intervention()
        dst = _vae_intervention()
        saved = dict(src.state_dict())
        rot_name = next(n for n, _ in src.named_buffers() if "parametrizations" in n)
        del saved[rot_name]
        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_init_from_state_dict(dst, saved, source="src")

    def test_extra_unexpected_key_is_incompatible(self):
        src = _vae_intervention()
        dst = _vae_intervention()
        saved = dict(src.state_dict())
        saved["unexpected.weight"] = torch.randn(2, 2)
        with self.assertRaisesRegex(ValueError, "extra in source"):
            apply_init_from_state_dict(dst, saved, source="src")


class LoadIntervenableStateDictTests(unittest.TestCase):
    def test_roundtrip_via_intkey_bin(self):
        src = _vae_intervention()
        _perturb(src)
        dst = _vae_intervention()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "intervenable_model"))
            torch.save(
                src.state_dict(),
                os.path.join(tmp, "intervenable_model", "intkey_layer.0.comp.block_output.bin"),
            )
            loaded = load_intervenable_state_dict(tmp)
            n = apply_init_from_state_dict(dst, loaded, source=tmp)
        self.assertGreater(n, 0)
        self.assertTrue(
            torch.allclose(src.rotate_layer.weight, dst.rotate_layer.weight, atol=1e-5)
        )
        self.assertTrue(torch.equal(src.word_mu.weight, dst.word_mu.weight))

    def test_missing_intervenable_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_intervenable_state_dict(tmp)

    def test_multiple_bins_are_rejected(self):
        src = _vae_intervention()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "intervenable_model"))
            torch.save(
                src.state_dict(),
                os.path.join(tmp, "intervenable_model", "intkey_a.bin"),
            )
            torch.save(
                src.state_dict(),
                os.path.join(tmp, "intervenable_model", "intkey_b.bin"),
            )
            with self.assertRaisesRegex(ValueError, "expected one intkey"):
                load_intervenable_state_dict(tmp)

    def test_init_learnable_parameters_from_run_uses_reft_model(self):
        src_iv = _vae_intervention()
        _perturb(src_iv)
        dst_iv = _vae_intervention()
        reft = type("Reft", (), {"interventions": {"layer.0": dst_iv}})()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "intervenable_model"))
            torch.save(
                src_iv.state_dict(),
                os.path.join(tmp, "intervenable_model", "intkey_dummy.bin"),
            )
            n = init_learnable_parameters_from_run(reft, tmp)
        self.assertGreater(n, 0)
        self.assertTrue(torch.equal(src_iv.word_mu.weight, dst_iv.word_mu.weight))


class TrainConfigInitFromTests(unittest.TestCase):
    def test_resolves_abspath_when_intervenable_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dummy_init_from_dir(tmp)
            cfg = TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                init_from=tmp,
            )
            self.assertEqual(cfg.init_from, os.path.abspath(tmp))

    def test_rejects_missing_directory(self):
        with self.assertRaisesRegex(ValueError, "--init-from is not a directory"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                init_from="/no/such/init-from-dir",
            )

    def test_rejects_directory_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "intervenable_model"):
                TrainConfig(
                    task="semantle",
                    semantle_csv=("data.csv",),
                    init_from=tmp,
                )

    def test_rejects_empty_intervenable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "intervenable_model"))
            with self.assertRaisesRegex(ValueError, "intkey_"):
                TrainConfig(
                    task="semantle",
                    semantle_csv=("data.csv",),
                    init_from=tmp,
                )

    def test_rejects_multiple_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dummy_init_from_dir(tmp, extra_bins=1)
            with self.assertRaisesRegex(ValueError, "expected one intkey"):
                TrainConfig(
                    task="semantle",
                    semantle_csv=("data.csv",),
                    init_from=tmp,
                )


if __name__ == "__main__":
    unittest.main()
