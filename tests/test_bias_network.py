"""Tests that TargetBiasNetwork receives non-zero gradients during backprop."""

from __future__ import annotations

import unittest

import torch

from boreft.pyreft.bias_network import TargetBiasNetwork
from boreft.pyreft.interventions import DistributionalWordIntervention


def _assert_nonzero_grads(module: torch.nn.Module, *, msg_prefix: str = "") -> None:
    saw_grad = False
    for name, param in module.named_parameters():
        prefix = f"{msg_prefix}{name}: " if msg_prefix else f"{name}: "
        self_msg = prefix
        assert param.grad is not None, f"{self_msg}expected grad, got None"
        assert param.grad.abs().sum().item() > 0, f"{self_msg}expected non-zero grad"
        saw_grad = True
    assert saw_grad, f"{msg_prefix}module has no parameters"


class BiasNetworkGradTest(unittest.TestCase):
    def test_mu_and_logvar_heads_receive_grads(self):
        net = TargetBiasNetwork(embed_dim=16, bias_dim=4, learnable_logvar=True)
        e = torch.randn(3, 16)
        mu, logvar = net(e)
        self.assertIsNotNone(logvar)
        loss = mu.pow(2).sum() + logvar.pow(2).sum()
        loss.backward()
        _assert_nonzero_grads(net)

    def test_mu_only_head_receives_grads(self):
        net = TargetBiasNetwork(embed_dim=8, bias_dim=3, learnable_logvar=False)
        e = torch.randn(2, 8)
        mu, logvar = net(e)
        self.assertIsNone(logvar)
        mu.sum().backward()
        _assert_nonzero_grads(net)

    def test_residual_skip_receives_grads(self):
        net = TargetBiasNetwork(
            embed_dim=12, bias_dim=5, learnable_logvar=False, residual=True
        )
        e = torch.randn(4, 12)
        net.mu(e).pow(2).sum().backward()
        _assert_nonzero_grads(net)
        self.assertIsNotNone(net.mu_skip)
        self.assertIsNotNone(net.mu_skip.weight.grad)
        self.assertGreater(net.mu_skip.weight.grad.abs().sum().item(), 0)

    def test_intervention_lookup_backprops_to_bias_network(self):
        embed_cache = torch.randn(8, 16)
        intervention = DistributionalWordIntervention(
            embed_dim=32,
            low_rank_dimension=4,
            num_words=8,
            add_bias_network=True,
            embed_cache=embed_cache,
            dtype=torch.float32,
            device="cpu",
        )
        word_ids = torch.tensor([0, 2, 5], dtype=torch.long)
        mu, logvar = intervention.get_bias_mu_logvar(word_ids)
        (mu.pow(2).sum() + logvar.pow(2).sum()).backward()
        _assert_nonzero_grads(intervention.bias_network)
        self.assertFalse(intervention.embed_cache.requires_grad)

    def test_load_ignores_saved_embed_cache_when_module_has_none(self):
        src = DistributionalWordIntervention(
            embed_dim=32,
            low_rank_dimension=4,
            num_words=4,
            add_bias_network=True,
            embed_cache=torch.randn(4, 8),
            dtype=torch.float32,
            device="cpu",
        )
        sd = dict(src.state_dict())
        sd["embed_cache"] = torch.randn(4, 8)
        dst = DistributionalWordIntervention(
            embed_dim=32,
            low_rank_dimension=4,
            num_words=4,
            add_bias_network=True,
            bias_network_input_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertIsNone(dst.embed_cache)
        dst.load_state_dict(sd)
        self.assertIsNone(dst.embed_cache)
        sd["not_a_real_param"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            dst.load_state_dict(sd)


if __name__ == "__main__":
    unittest.main()
