import math
import unittest

import torch

from boreft.pyreft.losses import kl_divergence


def _old_kl(mu, logvar, prior_var=1.0):
    """Reference KL (pre-free-bits): sum over dims, mean over batch."""
    lv = torch.clamp(logvar, -14.0, 10.0)
    log_prior_var = math.log(prior_var)
    kl = -0.5 * torch.sum(
        1 + lv - log_prior_var - (mu.pow(2) + lv.exp()) / prior_var,
        dim=-1,
    )
    return kl.mean()


class TestFreeBitsKL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mu = torch.randn(16, 8)
        self.logvar = torch.randn(16, 8)

    def test_free_bits_zero_matches_old(self):
        """free_bits=0 must be numerically identical to the original KL."""
        self.assertTrue(
            torch.allclose(
                kl_divergence(self.mu, self.logvar, free_bits=0.0),
                _old_kl(self.mu, self.logvar),
            )
        )

    def test_free_bits_zero_matches_old_prior_var(self):
        self.assertTrue(
            torch.allclose(
                kl_divergence(self.mu, self.logvar, prior_var=2.0, free_bits=0.0),
                _old_kl(self.mu, self.logvar, prior_var=2.0),
            )
        )

    def test_floor_engages_near_prior(self):
        """At the prior (mu=0, logvar=0) per-dim KL=0, so the result is floor*dim."""
        mu = torch.zeros(16, 8)
        logvar = torch.zeros(16, 8)
        fb = 0.1
        self.assertAlmostEqual(
            float(kl_divergence(mu, logvar, free_bits=fb)), fb * 8, places=5
        )

    def test_free_bits_lower_bounded(self):
        """Free-bits KL can never drop below floor * num_dims."""
        fb = 0.5
        kl = float(kl_divergence(self.mu, self.logvar, free_bits=fb))
        self.assertGreaterEqual(kl + 1e-6, fb * 8)

    def test_floor_inactive_when_kl_large(self):
        """When every per-dim KL exceeds the floor, free-bits is a no-op."""
        mu = 5.0 * torch.ones(16, 8)  # large mu -> large per-dim KL
        logvar = torch.zeros(16, 8)
        self.assertTrue(
            torch.allclose(
                kl_divergence(mu, logvar, free_bits=0.01),
                _old_kl(mu, logvar),
            )
        )

    def test_negative_free_bits_rejected(self):
        with self.assertRaisesRegex(ValueError, "free_bits"):
            kl_divergence(self.mu, self.logvar, free_bits=-0.1)

    def test_gradient_zero_below_floor(self):
        """No gradient flows through a dimension whose batch-mean KL is below floor."""
        mu = torch.zeros(16, 8, requires_grad=True)
        logvar = torch.zeros(16, 8, requires_grad=True)
        kl_divergence(mu, logvar, free_bits=0.5).backward()
        self.assertTrue(torch.allclose(mu.grad, torch.zeros_like(mu.grad)))
        self.assertTrue(torch.allclose(logvar.grad, torch.zeros_like(logvar.grad)))


if __name__ == "__main__":
    unittest.main()
