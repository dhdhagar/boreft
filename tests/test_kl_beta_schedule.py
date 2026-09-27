import unittest

from boreft.pyreft.losses import (
    align_linear_annealing_map,
    effective_loss_coeffs,
    legacy_kl_annealing_map_from_saved_cfg,
    linear_anneal,
    load_linear_annealing_map,
    parse_linear_annealing_map,
    resolve_linear_annealing_map,
    serialize_linear_annealing_map,
)
from boreft.train_args import TrainConfig, resolved_linear_annealing_map


class TestLinearAnneal(unittest.TestCase):
    def test_linear_anneal_helper(self):
        self.assertAlmostEqual(
            linear_anneal(0.0, 1.0, epochs=1, global_step=2, steps_per_epoch=4),
            0.5,
        )
        self.assertAlmostEqual(
            linear_anneal(0.0, 0.1, epochs=2, global_step=0, steps_per_epoch=4),
            0.0,
        )
        self.assertAlmostEqual(
            linear_anneal(0.0, 0.1, epochs=2, global_step=4, steps_per_epoch=4),
            0.05,
        )
        self.assertAlmostEqual(
            linear_anneal(0.0, 0.1, epochs=2, global_step=8, steps_per_epoch=4),
            0.1,
        )
        self.assertAlmostEqual(
            linear_anneal(0.0, 0.1, epochs=2, global_step=12, steps_per_epoch=4),
            0.1,
        )


class TestLinearAnnealMap(unittest.TestCase):
    def test_parse_full_and_shorthand(self):
        parsed = parse_linear_annealing_map(
            "kl_beta=0:0.1:50,lambda_sdpo=0:20,lambda_ce=1.0:0.5:10",
            defaults={"kl_beta": 0.1, "lambda_ce": 1.0, "lambda_sdpo": 1.0},
        )
        self.assertEqual(parsed["kl_beta"], (0.0, 0.1, 50))
        self.assertEqual(parsed["lambda_sdpo"], (0.0, 1.0, 20))
        self.assertEqual(parsed["lambda_ce"], (1.0, 0.5, 10))

    def test_rejects_unknown_key(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_linear_annealing_map(
                "lambda_mm=0:1:5",
                defaults={"kl_beta": 0.1, "lambda_ce": 1.0, "lambda_sdpo": 0.0},
            )

    def test_effective_loss_coeffs(self):
        defaults = {"kl_beta": 0.1, "lambda_ce": 1.0, "lambda_sdpo": 1.0}
        anneal_map = {
            "kl_beta": (0.0, 0.1, 2),
            "lambda_sdpo": (0.0, 1.0, 1),
        }
        mid = effective_loss_coeffs(
            defaults,
            anneal_map,
            global_step=4,
            steps_per_epoch=4,
        )
        self.assertAlmostEqual(mid["kl_beta"], 0.05)
        self.assertAlmostEqual(mid["lambda_sdpo"], 1.0)
        self.assertAlmostEqual(mid["lambda_ce"], 1.0)

        early = effective_loss_coeffs(
            defaults,
            anneal_map,
            global_step=0,
            steps_per_epoch=4,
        )
        self.assertAlmostEqual(early["lambda_sdpo"], 0.0)

    def test_serialize_roundtrip(self):
        original = {"lambda_ce": (0.5, 1.0, 3), "kl_beta": (0.0, 0.2, 10)}
        raw = serialize_linear_annealing_map(original)
        self.assertEqual(raw["kl_beta"], [0.0, 0.2, 10])
        loaded = load_linear_annealing_map(raw)
        self.assertEqual(loaded["kl_beta"], (0.0, 0.2, 10))
        self.assertEqual(loaded["lambda_ce"], (0.5, 1.0, 3))

    def test_resolve_passthrough(self):
        resolved = resolve_linear_annealing_map(
            raw_map="kl_beta=0:50",
            defaults={"kl_beta": 0.1, "lambda_ce": 1.0, "lambda_sdpo": 0.0},
        )
        self.assertEqual(resolved["kl_beta"], (0.0, 0.1, 50))

    def test_legacy_kl_from_saved_cfg(self):
        legacy = legacy_kl_annealing_map_from_saved_cfg(
            {
                "kl_beta_anneal": True,
                "kl_beta_start": 0.0,
                "kl_beta_anneal_epochs": 50,
            },
            kl_beta=0.1,
        )
        self.assertEqual(legacy["kl_beta"], (0.0, 0.1, 50))
        self.assertEqual(
            legacy_kl_annealing_map_from_saved_cfg({}, kl_beta=0.1),
            {},
        )

    def test_align_rewrites_ends_and_drops_disabled(self):
        aligned = align_linear_annealing_map(
            {
                "kl_beta": (0.0, 0.1, 50),
                "lambda_ce": (0.5, 1.0, 10),
                "lambda_sdpo": (0.0, 1.0, 20),
            },
            kl_beta=0.05,
            lambda_ce=0.0,
            lambda_sdpo=0.0,
            override_ends={"kl_beta"},
        )
        self.assertEqual(aligned["kl_beta"], (0.0, 0.05, 50))
        self.assertNotIn("lambda_ce", aligned)
        self.assertNotIn("lambda_sdpo", aligned)

    def test_align_preserves_ends_without_override(self):
        aligned = align_linear_annealing_map(
            {"lambda_ce": (1.0, 0.2, 20)},
            kl_beta=0.0,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(aligned["lambda_ce"], (1.0, 0.2, 20))

    def test_rejects_two_field_start_end_without_epochs(self):
        with self.assertRaisesRegex(ValueError, "three fields"):
            parse_linear_annealing_map(
                "kl_beta=0:0.1",
                defaults={"kl_beta": 0.1, "lambda_ce": 1.0, "lambda_sdpo": 0.0},
            )


class TestBatchAnnealMapResolve(unittest.TestCase):
    def test_inherit_checkpoint_map(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map

        resolved = resolve_batch_linear_annealing_map(
            BatchLearnConfig(),
            {"linear_annealing_map": {"kl_beta": [0.0, 0.1, 50]}},
            kl_beta=0.1,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(resolved["kl_beta"], (0.0, 0.1, 50))

    def test_inherit_preserves_cooldown_ends(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map

        # Static lambda_ce is the training gate (1.0); map cools to 0.2.
        resolved = resolve_batch_linear_annealing_map(
            BatchLearnConfig(),
            {"linear_annealing_map": {"lambda_ce": [1.0, 0.2, 20]}},
            kl_beta=0.0,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(resolved["lambda_ce"], (1.0, 0.2, 20))

    def test_coeff_override_retargets_inherited_end(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map

        resolved = resolve_batch_linear_annealing_map(
            BatchLearnConfig(lambda_ce=0.5),
            {"linear_annealing_map": {"lambda_ce": [1.0, 0.2, 20]}},
            kl_beta=0.0,
            lambda_ce=0.5,
            lambda_sdpo=0.0,
        )
        self.assertEqual(resolved["lambda_ce"], (1.0, 0.5, 20))

    def test_override_and_disable(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map

        overridden = resolve_batch_linear_annealing_map(
            BatchLearnConfig(linear_annealing_map="kl_beta=0:25"),
            {"linear_annealing_map": {"lambda_ce": [1.0, 0.5, 10]}},
            kl_beta=0.2,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(overridden, {"kl_beta": (0.0, 0.2, 25)})

        disabled = resolve_batch_linear_annealing_map(
            BatchLearnConfig(linear_annealing_map=""),
            {"linear_annealing_map": {"kl_beta": [0.0, 0.1, 50]}},
            kl_beta=0.1,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(disabled, {})

    def test_legacy_checkpoint_fallback(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map

        resolved = resolve_batch_linear_annealing_map(
            BatchLearnConfig(),
            {
                "kl_beta_anneal": True,
                "kl_beta_start": 0.0,
                "kl_beta_anneal_epochs": 40,
            },
            kl_beta=0.1,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(resolved["kl_beta"], (0.0, 0.1, 40))

    def test_null_map_key_disables_without_legacy_fallback(self):
        from boreft.learn_bias import BatchLearnConfig, resolve_batch_linear_annealing_map
        from boreft.pyreft.losses import annealing_map_from_saved_cfg

        saved = {
            "linear_annealing_map": None,
            "kl_beta_anneal": True,
            "kl_beta_start": 0.0,
            "kl_beta_anneal_epochs": 40,
            "kl_beta": 0.1,
        }
        self.assertEqual(annealing_map_from_saved_cfg(saved), {})
        resolved = resolve_batch_linear_annealing_map(
            BatchLearnConfig(),
            saved,
            kl_beta=0.1,
            lambda_ce=1.0,
            lambda_sdpo=0.0,
        )
        self.assertEqual(resolved, {})


class TestLinearAnnealValidation(unittest.TestCase):
    def test_map_config(self):
        cfg = TrainConfig(
            task="semantle",
            semantle_csv=("data.csv",),
            kl_beta=0.1,
            lambda_ce=1.0,
            linear_annealing_map="kl_beta=0:0.1:25,lambda_ce=0.5:10",
        )
        resolved = resolved_linear_annealing_map(cfg)
        self.assertEqual(resolved["kl_beta"], (0.0, 0.1, 25))
        self.assertEqual(resolved["lambda_ce"], (0.5, 1.0, 10))

    def test_map_requires_kl_beta_gate(self):
        with self.assertRaisesRegex(ValueError, "requires --kl-beta > 0"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                kl_beta=0.0,
                linear_annealing_map="kl_beta=0:0.1:10",
            )

    def test_map_requires_vae_bias(self):
        with self.assertRaisesRegex(ValueError, "--bias-type vae"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                bias_type="linear",
                kl_beta=0.1,
                linear_annealing_map="kl_beta=0:0.1:10",
            )

    def test_map_sdpo_requires_lambda_sdpo_gate(self):
        with self.assertRaisesRegex(ValueError, "requires --lambda-sdpo > 0"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                lambda_sdpo=0.0,
                linear_annealing_map="lambda_sdpo=0:1.0:10",
            )

    def test_rejects_negative_max_grad_norm(self):
        with self.assertRaisesRegex(ValueError, "--max-grad-norm"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                max_grad_norm=-1.0,
            )

    def test_rejects_nonpositive_grad_acc_steps(self):
        with self.assertRaisesRegex(ValueError, "--grad-acc-steps"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                grad_acc_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
