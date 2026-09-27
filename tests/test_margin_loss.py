import unittest

import torch

from boreft.pyreft.losses import token_greedy_margin_loss
from boreft.train_args import TrainConfig


class TokenGreedyMarginLossTest(unittest.TestCase):
    def test_causal_shift_ignore_mask_and_margin(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 1.30, 1.00],
                    [0.95, 0.0, 1.00],
                    [10.0, 0.0, 0.0],
                ]
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[-100, 1, 2]])

        loss = token_greedy_margin_loss(logits, labels, margin=0.1)

        # First target clears the margin; the second misses it by 0.05.
        self.assertAlmostEqual(loss.item(), 0.025, places=6)

    def test_zero_loss_and_gradient_after_margin_is_satisfied(self):
        logits = torch.tensor(
            [[[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[-100, 1]])

        loss = token_greedy_margin_loss(logits, labels, margin=0.5)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_sample_weights_apply_after_each_sample_token_mean(self):
        logits = torch.tensor(
            [
                [[1.0, 0.5], [0.0, 0.0]],
                [[1.5, 0.5], [0.0, 0.0]],
            ]
        )
        labels = torch.tensor([[-100, 1], [-100, 1]])

        loss = token_greedy_margin_loss(
            logits,
            labels,
            margin=0.0,
            sample_weights=torch.tensor([1.0, 3.0]),
        )

        self.assertAlmostEqual(loss.item(), 0.875, places=6)

    def test_all_ignored_returns_differentiable_zero(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        labels = torch.full((2, 3), -100)

        loss = token_greedy_margin_loss(logits, labels)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)

    def test_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            token_greedy_margin_loss(
                torch.zeros(1, 2, 3),
                torch.tensor([[-100, 1]]),
                margin=-0.1,
            )

    def test_train_config_rejects_negative_margin(self):
        with self.assertRaisesRegex(ValueError, "--margin-loss-margin"):
            TrainConfig(
                task="semantle",
                semantle_csv=("train.csv",),
                use_margin_loss=True,
                margin_loss_margin=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
