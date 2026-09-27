"""Tests for save-best-params helpers and end-of-train layout."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

from boreft.data import ReftItem
from boreft.data.eval_callback import TargetReconEval
from boreft.data_utils import (
    BEST_CHECKPOINT_DIRNAME,
    BEST_CHECKPOINT_INFO_NAME,
    LATEST_CHECKPOINT_DIRNAME,
    TRAINING_CONFIG_NAME,
    add_load_latest_argument,
    has_best_checkpoint,
    promote_best_checkpoint_to_root,
    resolve_checkpoint_dir,
    resolve_weight_dir_and_run_config,
)
from boreft.train import _finalize_training_checkpoint
from boreft.train_args import TrainConfig


def _touch_intervenable(root: str, marker: str = "root") -> str:
    interv = os.path.join(root, "intervenable_model")
    os.makedirs(interv, exist_ok=True)
    path = os.path.join(interv, "intkey_dummy.bin")
    with open(path, "wb") as f:
        f.write(marker.encode("utf-8"))
    return path


def _read_marker(root: str) -> str:
    path = os.path.join(root, "intervenable_model", "intkey_dummy.bin")
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


class ResolveCheckpointDirTests(unittest.TestCase):
    def test_default_returns_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch_intervenable(tmp, "root")
            latest = os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME)
            _touch_intervenable(latest, "latest")
            self.assertEqual(
                resolve_checkpoint_dir(tmp, load_latest=False),
                os.path.abspath(tmp),
            )

    def test_load_latest_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch_intervenable(tmp, "root")
            latest = os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME)
            _touch_intervenable(latest, "latest")
            self.assertEqual(
                resolve_checkpoint_dir(tmp, load_latest=True),
                os.path.abspath(latest),
            )

    def test_load_latest_falls_back_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch_intervenable(tmp, "root")
            os.makedirs(os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME), exist_ok=True)
            self.assertEqual(
                resolve_checkpoint_dir(tmp, load_latest=True),
                os.path.abspath(tmp),
            )

    def test_run_config_overlays_root_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch_intervenable(tmp, "root")
            with open(os.path.join(tmp, TRAINING_CONFIG_NAME), "w") as f:
                json.dump({"model_name": "root-model", "seed": 7}, f)
            latest = os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME)
            _touch_intervenable(latest, "latest")
            with open(os.path.join(latest, "intervention_config.json"), "w") as f:
                json.dump({"layer": 3, "model_name": "latest-model"}, f)
            weight_dir, cfg = resolve_weight_dir_and_run_config(
                tmp, load_latest=True
            )
            self.assertEqual(weight_dir, os.path.abspath(latest))
            self.assertEqual(cfg["layer"], 3)
            self.assertEqual(cfg["model_name"], "root-model")
            self.assertEqual(cfg["seed"], 7)


class PromoteBestCheckpointTests(unittest.TestCase):
    def test_promotes_weights_and_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch_intervenable(tmp, "stale-root")
            best = os.path.join(tmp, BEST_CHECKPOINT_DIRNAME)
            _touch_intervenable(best, "best")
            with open(os.path.join(best, "checkpoint_info.json"), "w") as f:
                json.dump(
                    {
                        "selection_metric": "embed_sim",
                        "mean_sim_at_save": 0.9,
                        "global_step": 12,
                    },
                    f,
                )
            info = promote_best_checkpoint_to_root(tmp)
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info["mean_sim_at_save"], 0.9)
            self.assertEqual(_read_marker(tmp), "best")
            root_info = os.path.join(tmp, BEST_CHECKPOINT_INFO_NAME)
            self.assertTrue(os.path.isfile(root_info))


class SaveBestCallbackTests(unittest.TestCase):
    def _make_callback(self, output_dir: str, *, save_best: bool = True):
        item = ReftItem(id=0, prompt="p", target="word")
        return TargetReconEval(
            intervenable=object(),
            tokenizer=object(),
            items=[item],
            eval_steps=1,
            save_best_params=save_best,
            output_dir=output_dir,
            intervention_config={},
        )

    def test_saves_on_first_and_strict_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = self._make_callback(tmp)
            calls: list[float] = []

            def fake_save(sub, **kwargs):
                calls.append(float(kwargs["checkpoint_info"]["mean_sim_at_save"]))
                _touch_intervenable(sub, f"m{len(calls)}")
                return dict(kwargs.get("intervention_config") or {})

            with mock.patch(
                "boreft.data_utils.save_reft_checkpoint_dir", side_effect=fake_save
            ):
                cb._maybe_save_best_checkpoint(0.5, global_step=1, epoch=0.0)
                cb._maybe_save_best_checkpoint(0.5, global_step=2, epoch=0.0)  # tie
                cb._maybe_save_best_checkpoint(0.4, global_step=3, epoch=0.0)  # worse
                cb._maybe_save_best_checkpoint(0.6, global_step=4, epoch=0.0)

            self.assertEqual(calls, [0.5, 0.6])
            self.assertEqual(cb._best_selection_mean, 0.6)
            self.assertTrue(has_best_checkpoint(tmp))

    def test_disabled_skips_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = self._make_callback(tmp, save_best=False)
            with mock.patch(
                "boreft.data_utils.save_reft_checkpoint_dir"
            ) as save_mock:
                cb._maybe_save_best_checkpoint(0.9, global_step=1, epoch=0.0)
            save_mock.assert_not_called()
            self.assertIsNone(cb._best_selection_mean)


class FinalizeTrainingCheckpointTests(unittest.TestCase):
    def test_with_best_writes_latest_and_promotes_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = os.path.join(tmp, BEST_CHECKPOINT_DIRNAME)
            _touch_intervenable(best, "best")
            with open(os.path.join(best, "checkpoint_info.json"), "w") as f:
                json.dump(
                    {"mean_sim_at_save": 0.8, "selection_metric": "embed_sim"}, f
                )
            with open(os.path.join(tmp, TRAINING_CONFIG_NAME), "w") as f:
                json.dump({"seed": 1}, f)

            class _Tok:
                def save_pretrained(self, path):
                    os.makedirs(path, exist_ok=True)
                    open(os.path.join(path, "tokenizer.json"), "w").close()

            saved_dirs: list[str] = []

            def fake_save(save_dir, **kwargs):
                saved_dirs.append(os.path.abspath(save_dir))
                _touch_intervenable(save_dir, "latest")
                src = kwargs.get("training_config_src")
                if src and os.path.isfile(src):
                    open(
                        os.path.join(save_dir, TRAINING_CONFIG_NAME), "w"
                    ).close()
                return dict(kwargs["intervention_config"])

            with mock.patch(
                "boreft.train.save_reft_checkpoint_dir", side_effect=fake_save
            ):
                _finalize_training_checkpoint(
                    output_dir=tmp,
                    reft_model=object(),
                    tokenizer=_Tok(),
                    items=[],
                    intervention_config={"task": "semantle"},
                    save_best_params=True,
                )

            latest = os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME)
            self.assertEqual(saved_dirs, [os.path.abspath(latest)])
            self.assertEqual(_read_marker(latest), "latest")
            self.assertEqual(_read_marker(tmp), "best")
            self.assertTrue(
                os.path.isfile(os.path.join(latest, TRAINING_CONFIG_NAME))
            )

    def test_without_best_saves_root_only(self):
        with tempfile.TemporaryDirectory() as tmp:

            class _Tok:
                def save_pretrained(self, path):
                    os.makedirs(path, exist_ok=True)

            saved_dirs: list[str] = []

            def fake_save(save_dir, **kwargs):
                saved_dirs.append(os.path.abspath(save_dir))
                _touch_intervenable(save_dir, "final")
                return dict(kwargs["intervention_config"])

            with mock.patch(
                "boreft.train.save_reft_checkpoint_dir", side_effect=fake_save
            ):
                _finalize_training_checkpoint(
                    output_dir=tmp,
                    reft_model=object(),
                    tokenizer=_Tok(),
                    items=[],
                    intervention_config={"task": "semantle"},
                    save_best_params=True,
                )

            self.assertEqual(saved_dirs, [os.path.abspath(tmp)])
            self.assertFalse(
                os.path.isdir(os.path.join(tmp, LATEST_CHECKPOINT_DIRNAME))
            )


class TrainConfigDefaultsTests(unittest.TestCase):
    def test_save_best_params_default_true(self):
        field = TrainConfig.__dataclass_fields__["save_best_params"]
        self.assertIs(field.default, True)
        load_field = TrainConfig.__dataclass_fields__["load_latest"]
        self.assertIs(load_field.default, False)


class LoadLatestCliTests(unittest.TestCase):
    def test_boolean_optional_action(self):
        p = argparse.ArgumentParser()
        add_load_latest_argument(p)
        self.assertFalse(p.parse_args([]).load_latest)
        self.assertTrue(p.parse_args(["--load-latest"]).load_latest)
        self.assertFalse(p.parse_args(["--no-load-latest"]).load_latest)


if __name__ == "__main__":
    unittest.main()
