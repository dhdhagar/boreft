from __future__ import annotations

import unittest

import numpy as np
import torch
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from boreft.bo.surrogate import (
    BiasExactGP,
    SurrogateFitConfig,
    fit_surrogate,
)


class BiasGPSurrogateTest(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        self.y = -np.square(self.x - 0.7).sum(axis=1)
        self.bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    def test_static_gp_fits_and_predicts(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(kind="static"),
        )
        posterior = surrogate.model.posterior(
            torch.tensor([[0.5, 0.5]], dtype=torch.double)
        )
        self.assertEqual(tuple(posterior.mean.shape), (1, 1))
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertEqual(surrogate.metadata()["feature_dim"], 2)
        self.assertFalse(surrogate.metadata()["use_ard"])
        self.assertEqual(surrogate.metadata()["kernel"], "matern-2.5")
        self.assertIsInstance(
            surrogate.model.covar_module.base_kernel, MaternKernel
        )
        self.assertEqual(surrogate.model.covar_module.base_kernel.nu, 2.5)
        self.assertEqual(
            surrogate.model.covar_module.base_kernel.lengthscale.numel(), 1
        )

    def test_ard_fits_one_lengthscale_per_feature(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(kind="static", use_ard=True),
        )
        lengthscale = surrogate.model.covar_module.base_kernel.lengthscale
        self.assertTrue(surrogate.metadata()["use_ard"])
        self.assertEqual(tuple(lengthscale.shape[-1:]), (2,))
        self.assertTrue(torch.isfinite(lengthscale).all())

    def test_projected_ard_uses_projection_dimension(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(
                kind="projected",
                use_ard=True,
                projection_dim=3,
                steps=4,
                patience=2,
            ),
        )
        lengthscale = surrogate.model.covar_module.base_kernel.lengthscale
        self.assertEqual(surrogate.metadata()["feature_dim"], 3)
        self.assertEqual(tuple(lengthscale.shape[-1:]), (3,))
        self.assertTrue(surrogate.checkpoint()["use_ard"])

    def test_rbf_kernel_is_squared_exponential(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(kind="static", kernel="rbf"),
        )
        self.assertEqual(surrogate.metadata()["kernel"], "rbf")
        self.assertEqual(surrogate.checkpoint()["kernel"], "rbf")
        self.assertIsInstance(surrogate.model.covar_module.base_kernel, RBFKernel)
        posterior = surrogate.model.posterior(
            torch.tensor([[0.5, 0.5]], dtype=torch.double)
        )
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_matern_nu_options(self):
        for kind, nu in (("matern-0.5", 0.5), ("matern-1.5", 1.5)):
            surrogate = fit_surrogate(
                self.x,
                self.y,
                self.bounds,
                SurrogateFitConfig(kind="static", kernel=kind),
            )
            kernel = surrogate.model.covar_module.base_kernel
            self.assertEqual(surrogate.metadata()["kernel"], kind)
            self.assertIsInstance(kernel, MaternKernel)
            self.assertEqual(kernel.nu, nu)

    def test_rbf_ard_fits_one_lengthscale_per_feature(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(kind="static", kernel="rbf", use_ard=True),
        )
        lengthscale = surrogate.model.covar_module.base_kernel.lengthscale
        self.assertEqual(tuple(lengthscale.shape[-1:]), (2,))
        self.assertTrue(torch.isfinite(lengthscale).all())

    def test_rejects_unknown_kernel(self):
        with self.assertRaisesRegex(ValueError, "kernel"):
            fit_surrogate(
                self.x,
                self.y,
                self.bounds,
                SurrogateFitConfig(kind="static", kernel="linear"),  # type: ignore[arg-type]
            )

    def test_projected_mll_gradient_reaches_projection(self):
        x = torch.tensor(self.x, dtype=torch.double)
        y = torch.tensor(self.y, dtype=torch.double)
        bounds = torch.tensor(self.bounds, dtype=torch.double)
        model = BiasExactGP(x, y, bounds, projection_dim=3).double()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        loss = -mll(model(x), y)
        loss.backward()
        grad = model.projection.layers[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(torch.linalg.vector_norm(grad)), 0.0)
        self.assertEqual(len(model.projection.layers), 1)
        self.assertEqual(model.projection_layers, 1)

    def test_deeper_projection_stacks_linear_elu_blocks(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(
                kind="projected",
                projection_dim=3,
                projection_layers=2,
                steps=4,
                patience=2,
            ),
        )
        layers = surrogate.model.projection.layers
        self.assertEqual(len(layers), 2)
        self.assertEqual(tuple(layers[0].weight.shape), (3, 2))
        self.assertEqual(tuple(layers[1].weight.shape), (3, 3))
        self.assertEqual(surrogate.metadata()["projection_layers"], 2)
        self.assertEqual(surrogate.checkpoint()["projection_layers"], 2)

        x = torch.tensor(self.x, dtype=torch.double)
        y = torch.tensor(self.y, dtype=torch.double)
        bounds = torch.tensor(self.bounds, dtype=torch.double)
        model = BiasExactGP(
            x, y, bounds, projection_dim=3, projection_layers=2
        ).double()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        loss = -mll(model(x), y)
        loss.backward()
        for layer in model.projection.layers:
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(float(torch.linalg.vector_norm(layer.weight.grad)), 0.0)

    def test_projected_fit_loss_matches_returned_best_model(self):
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(
                kind="projected",
                projection_dim=3,
                steps=8,
                patience=3,
            ),
        )
        model = surrogate.model
        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with torch.no_grad():
            actual = float(
                (-mll(model(model.train_inputs[0]), model.train_targets)).item()
            )
        self.assertAlmostEqual(surrogate.fit_loss, actual, places=8)

    def test_rejects_nonfinite_observation(self):
        y = self.y.copy()
        y[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_surrogate(self.x, y, self.bounds)

    def test_known_observation_variance_uses_fixed_noise_likelihood(self):
        variances = np.array([0.01, 0.02, 0.03, 0.04])
        surrogate = fit_surrogate(
            self.x,
            self.y,
            self.bounds,
            SurrogateFitConfig(kind="static"),
            observation_variances=variances,
        )
        self.assertIsInstance(
            surrogate.model.likelihood,
            FixedNoiseGaussianLikelihood,
        )
        expected = variances / np.var(self.y)
        np.testing.assert_allclose(
            surrogate.model.likelihood.noise_covar.noise.detach().numpy(),
            expected,
        )

    def test_rejects_negative_observation_variance(self):
        with self.assertRaisesRegex(ValueError, "observation_variances"):
            fit_surrogate(
                self.x,
                self.y,
                self.bounds,
                observation_variances=[0.0, 0.0, -0.1, 0.0],
            )

    def test_rejects_invalid_projected_fit_config(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            fit_surrogate(
                self.x,
                self.y,
                self.bounds,
                SurrogateFitConfig(kind="projected", steps=0),
            )
        with self.assertRaisesRegex(ValueError, "projection_layers"):
            fit_surrogate(
                self.x,
                self.y,
                self.bounds,
                SurrogateFitConfig(kind="projected", projection_layers=0),
            )


if __name__ == "__main__":
    unittest.main()
