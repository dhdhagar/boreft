"""Tests for the interactive bias-subspace explorer."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from boreft.learn_bias import (
    BatchLearnControl,
    _install_batch_bias_network_inputs,
    delete_learned_biases,
    load_learned_biases,
)
from boreft.subspace_viz import (
    bias_l2_norm,
    fit_projection,
    mean_bias_l2_norm,
    plot_bounds,
    projected_point,
)
from boreft.viz_server import (
    BrowserSession,
    VizRuntime,
    recovery_parameter_changes,
    snapshot_module_state_dict,
    snapshot_recovery_parameters,
    state_dict_l2_delta,
)


class Projection2DTests(unittest.TestCase):
    def test_project_inverse_round_trip_on_fitted_plane(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [-1.0, 0.0, 2.0],
                [0.0, -1.0, 2.0],
            ]
        )
        projection = fit_projection(vectors)
        for vector in vectors:
            x, y = projection.project(vector)
            np.testing.assert_allclose(
                projection.inverse(x, y), vector, atol=1e-10
            )

    def test_single_point_projection_is_finite(self) -> None:
        projection = fit_projection([[2.0, -3.0, 4.0]])
        coordinates = projection.project([2.0, -3.0, 4.0])
        np.testing.assert_array_equal(coordinates, [0.0, 0.0])
        np.testing.assert_array_equal(
            projection.inverse(0.0, 0.0), [2.0, -3.0, 4.0]
        )
        self.assertTrue(np.isfinite(projection.inverse(20.0, 30.0)).all())

    def test_plot_bounds_pad_degenerate_axes(self) -> None:
        bounds = plot_bounds([{"x": 2.0, "y": -1.0}])
        self.assertLess(bounds["min_x"], 2.0)
        self.assertGreater(bounds["max_x"], 2.0)
        self.assertLess(bounds["min_y"], -1.0)
        self.assertGreater(bounds["max_y"], -1.0)

    def test_bias_norm_helpers_and_projected_point(self) -> None:
        self.assertAlmostEqual(bias_l2_norm([3.0, 4.0]), 5.0)
        self.assertAlmostEqual(
            mean_bias_l2_norm([[3.0, 4.0], [0.0, 0.0]]), 2.5
        )
        point = projected_point(
            point_id="train:0",
            kind="train",
            label="tablet",
            coordinates=[0.1, -0.2],
            cluster="devices",
            bias_norm=5.0,
        )
        self.assertEqual(point["bias_norm"], 5.0)


class BatchBiasNetworkInputTests(unittest.TestCase):
    def test_embed_cache_inputs_are_target_aligned_and_restored(self) -> None:
        class FakeIntervention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotate_layer = torch.nn.Linear(3, 2, bias=False)
                self.bias_input_source = "embed_cache"
                self.register_buffer("embed_cache", torch.ones(2, 4))

        intervention = FakeIntervention()
        original = intervention.embed_cache
        ckpt = SimpleNamespace(
            saved_cfg={"task": "semantle"},
            reft_model=SimpleNamespace(),
        )
        features = np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            dtype=np.float32,
        )
        with patch(
            "boreft.text_similarity.encode_reference_embeddings",
            return_value=features,
        ) as encode:
            restore = _install_batch_bias_network_inputs(
                ckpt,
                intervention,
                ["tablet", "notebook"],
                {
                    "tablet": "a touchscreen computer",
                    "notebook": "a book of blank pages",
                },
            )

        np.testing.assert_allclose(
            intervention.embed_cache.cpu().numpy(), features
        )
        encode.assert_called_once()
        restore()
        self.assertIs(intervention.embed_cache, original)


class ParameterChangeMetricTests(unittest.TestCase):
    def test_state_dict_l2_delta_relative_and_absolute(self) -> None:
        reference = {
            "weight": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
        }
        current = {
            "weight": torch.tensor([[3.0, 0.0], [0.0, 8.0]]),
        }
        delta = state_dict_l2_delta(current, reference)
        # ||ref|| = 5, ||Δ|| = 4 → relative = 4/5
        self.assertAlmostEqual(delta["absolute_l2"], 4.0)
        self.assertAlmostEqual(delta["relative_l2"], 0.8)

    def test_recovery_parameter_changes_marks_untrained_components(self) -> None:
        module = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        original = {"W": snapshot_module_state_dict(module), "R": None, "bias_network": None, "encoder": None}
        with torch.no_grad():
            module.weight.add_(1.0)
        current = {"W": snapshot_module_state_dict(module), "R": None, "bias_network": None, "encoder": None}
        previous = original
        changes = recovery_parameter_changes(
            current=current,
            previous=previous,
            original=original,
            learn_W=True,
            learn_R=False,
            learn_bias_network=False,
        )
        self.assertEqual(changes["vs_previous"]["W"]["status"], "changed")
        self.assertGreater(changes["vs_previous"]["W"]["relative_l2"], 0.0)
        self.assertEqual(changes["vs_previous"]["R"]["status"], "unchanged")
        self.assertIsNone(changes["vs_previous"]["R"]["relative_l2"])
        self.assertEqual(
            changes["vs_original"]["bias_network"]["status"], "unchanged"
        )


class ActivationTests(unittest.TestCase):
    def _fake_intervention(self) -> torch.nn.Module:
        class FakeIntervention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.learned_source = torch.nn.Linear(4, 3, bias=False)
                self.rotate_layer = torch.nn.Linear(3, 3, bias=False)
                self.bias_network = torch.nn.Linear(4, 3, bias=False)

            def get_semantic_encoder(self):
                return None

        intervention = FakeIntervention()
        with torch.no_grad():
            intervention.learned_source.weight.fill_(0.1)
            intervention.rotate_layer.weight.copy_(torch.eye(3))
            intervention.bias_network.weight.fill_(0.2)
        return intervention

    def _runtime(self, checkpoint_dir: str) -> VizRuntime:
        runtime = object.__new__(VizRuntime)
        runtime.checkpoint_dir = checkpoint_dir
        runtime.lock = threading.RLock()
        runtime.sessions = {}
        runtime.task = "semantle"
        runtime.word_ids = [7]
        runtime.words = ["laptop"]
        runtime.item_original_splits = ["train"]
        runtime.word_clusters = {"laptop": "all"}
        runtime.projection_sets = ["train_seen"]
        runtime.train_biases = np.asarray(
            [[1.0, 0.0, 4.0]], dtype=np.float32
        )
        runtime.projection = fit_projection(
            [[1.0, 0.0, 4.0], [-1.0, 0.0, 4.0], [0.0, 1.0, 4.0]]
        )
        runtime.train_points = [
            {
                "id": "train:7",
                "kind": "train",
                "label": "laptop",
                "x": 0.0,
                "y": 0.0,
                "cluster": "all",
                "metadata": {
                    "word_id": 7,
                    "original_split": "train",
                    "seen": True,
                    "set_key": "train_seen",
                },
            }
        ]
        runtime.additional_points = []
        runtime.learned_records = []
        runtime.display_learned_records = []
        runtime.test_words = ["tablet"]
        runtime.test_biases = np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32)
        runtime.test_base_biases = runtime.test_biases.copy()
        runtime.test_item_ids = [None]
        runtime.test_seen = [False]
        runtime.test_points = [
            {
                "id": "test:0",
                "kind": "test",
                "label": "tablet",
                "x": 0.0,
                "y": 0.0,
                "cluster": "test-interp",
                "metadata": {
                    "split": "interp",
                    "original_split": "test",
                    "seen": False,
                    "set_key": "test_unseen",
                    "vector_index": 0,
                },
            }
        ]
        runtime.supports_text_activation = True
        runtime.model_name = "test-model"
        runtime.label_source = "none"
        runtime.test_error = None
        runtime.raw_text_cache = {}
        runtime.raw_definition_lookup = {
            "laptop": "a portable computer",
            "tablet": "a flat touchscreen computer",
        }
        runtime.raw_definition_by_normalized = dict(runtime.raw_definition_lookup)
        runtime.jobs_lock = threading.Lock()
        runtime.recovery_jobs = {}
        runtime.recovery_controls = {}
        runtime.active_recovery_job_id = None
        runtime.recovery_status = {}
        intervention = self._fake_intervention()
        runtime.ckpt = SimpleNamespace(
            saved_cfg={"seed": 42},
            tokenizer=object(),
            reft_model=SimpleNamespace(
                interventions={"test": intervention}
            ),
            prompt="prompt",
            assistant_suffix=None,
            from_chat_template=False,
            intervention_token_id=None,
            content_span=None,
        )
        runtime.original_param_snapshot = snapshot_recovery_parameters(
            intervention
        )
        return runtime

    def test_map_payload_includes_bias_norms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.train_points[0]["bias_norm"] = bias_l2_norm(
                runtime.train_biases[0]
            )
            payload = runtime.map_payload()
            expected = float(np.linalg.norm(runtime.train_biases[0]))
            self.assertAlmostEqual(
                payload["axes"]["avg_train_bias_norm"], expected, places=5
            )
            self.assertAlmostEqual(
                payload["metadata"]["avg_train_bias_norm"], expected, places=5
            )
            train = next(p for p in payload["points"] if p["kind"] == "train")
            self.assertAlmostEqual(train["bias_norm"], expected, places=5)
            self.assertIn(payload["metadata"]["bias_type"], {"vae", "linear"})
            self.assertIn("kl_beta", payload["metadata"])
            self.assertIn("lambda_ce", payload["metadata"])
            self.assertIn("lambda_sdpo", payload["metadata"])
            self.assertIn("supports_sdpo", payload["metadata"])
            self.assertIn("task", payload["metadata"])
            self.assertIn("supports_fingerprints", payload["metadata"])
            self.assertIn("weight_decay_mode", payload["metadata"])
            self.assertIn("lr_scheduler_type", payload["metadata"])
            self.assertIn("warmup_ratio", payload["metadata"])
            self.assertIn("linear_annealing_map", payload["metadata"])
            self.assertIn("linear_annealing_map_cli", payload["metadata"])
            self.assertTrue(payload["metadata"]["supports_sdpo"])
            self.assertEqual(payload["metadata"]["task"], "semantle")
            self.assertFalse(payload["metadata"]["supports_fingerprints"])
            self.assertEqual(payload["metadata"]["weight_decay_mode"], "none")
            self.assertEqual(payload["metadata"]["projection_sets"], ["train_seen"])
            self.assertEqual(payload["metadata"]["set_counts"]["train_seen"], 1)
            self.assertEqual(payload["metadata"]["set_counts"]["test_unseen"], 1)

    def test_legacy_extension_origin_uses_safe_default_split(self) -> None:
        self.assertEqual(
            VizRuntime._item_original_split({"origin": "old"}), "train"
        )
        self.assertEqual(
            VizRuntime._item_original_split({"origin": "new"}), "additional"
        )
        self.assertEqual(
            VizRuntime._item_original_split(
                {"origin": "old", "original_split": "test"}
            ),
            "test",
        )

    def test_projection_can_fit_selected_seen_and_unseen_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            result = runtime.set_projection_sets(
                ["train_seen", "test_unseen"]
            )
            self.assertEqual(
                result["sets"], ["test_unseen", "train_seen"]
            )
            self.assertEqual(result["n_points"], 2)
            self.assertEqual(runtime.projection_sets, result["sets"])

    def test_projection_does_not_fall_back_to_unselected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.projection_sets = ["test_unseen"]
            old_projection = runtime.projection
            with open(
                f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(
                        {"target": "tablet", "mu": [0.4, -0.2, 4.5]}
                    )
                    + "\n"
                )

            with patch.object(
                runtime,
                "_effective_train_biases",
                return_value=runtime.train_biases.copy(),
            ):
                refreshed = runtime._refresh_subspace_geometry(force=True)

            self.assertFalse(refreshed)
            self.assertIs(runtime.projection, old_projection)
            self.assertEqual(runtime.test_points[0]["metadata"]["set_key"], "test_seen")

    def test_training_set_selection_resolves_matching_point_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            self.assertEqual(
                runtime._point_ids_for_sets({"train_seen"}),
                ["train:7"],
            )
            self.assertEqual(
                runtime._point_ids_for_sets({"test_unseen"}),
                ["test:0"],
            )

    def _wait_job(self, runtime: VizRuntime, job_id: str) -> dict:
        deadline = time.time() + 3
        while time.time() < deadline:
            job = runtime.recovery_job(job_id)
            if job["status"] in {"completed", "failed"}:
                return job
            time.sleep(0.01)
        self.fail(f"recovery job {job_id} did not finish")

    def test_learned_point_uses_exact_full_rank_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exact_mu = [0.25, -0.5, 9.0]
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(json.dumps({"target": "gadget", "mu": exact_mu}) + "\n")
            runtime = self._runtime(tmp)

            result = runtime.activate(
                "session", kind="learned", point_id="learned:0"
            )

            self.assertEqual(result["label"], "gadget")
            np.testing.assert_array_equal(
                runtime.sessions["session"].subspace,
                np.asarray(exact_mu, dtype=np.float32),
            )

    def test_delete_learned_biases_rewrites_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/learned_biases.jsonl"
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps({"target": "alpha", "mu": [1, 0, 0]}) + "\n")
                file.write(json.dumps({"target": "beta", "mu": [0, 1, 0]}) + "\n")
                file.write(json.dumps({"target": "Alpha", "mu": [2, 0, 0]}) + "\n")
            removed = delete_learned_biases(tmp, ["alpha"])
            self.assertEqual(removed, 2)
            remaining = load_learned_biases(tmp)
            self.assertEqual([r["target"] for r in remaining], ["beta"])

    def test_delete_learned_biases_matches_molopt_targets_by_canonical_smiles(
        self,
    ) -> None:
        """Equivalent spellings are one molecule; case-distinct ones are not."""
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/learned_biases.jsonl"
            with open(path, "w", encoding="utf-8") as file:
                file.write(json.dumps({"target": "OCC", "mu": [1, 0, 0]}) + "\n")
                file.write(json.dumps({"target": "c1ccccc1", "mu": [0, 1, 0]}) + "\n")
            removed = delete_learned_biases(tmp, ["CCO"], task="molopt")
            self.assertEqual(removed, 1)
            remaining = load_learned_biases(tmp)
            self.assertEqual([r["target"] for r in remaining], ["c1ccccc1"])

            self.assertEqual(
                delete_learned_biases(tmp, ["C1CCCCC1"], task="molopt"), 0
            )

    def test_delete_learned_biases_reads_task_from_checkpoint_config(self) -> None:
        """Callers that do not know the task still get the run's own normalizer."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/training_config.json", "w", encoding="utf-8") as file:
                json.dump({"task": "molopt"}, file)
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(json.dumps({"target": "OCC", "mu": [1, 0, 0]}) + "\n")
            self.assertEqual(delete_learned_biases(tmp, ["CCO"]), 1)

    def test_delete_learned_target_removes_map_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "target": "gadget",
                            "mu": [0.5, -0.25, 3.0],
                            "learned_at": "2026-01-01T00:00:00Z",
                        }
                    )
                    + "\n"
                )
            runtime = self._runtime(tmp)
            runtime.activate("session", kind="learned", point_id="learned:0")
            before = runtime.map_payload()
            self.assertTrue(
                any(point["kind"] == "learned" for point in before["points"])
            )

            result = runtime.delete_learned_target(point_id="learned:0")
            self.assertEqual(result["target"], "gadget")
            self.assertEqual(result["removed_records"], 1)
            self.assertEqual(load_learned_biases(tmp), [])
            after = runtime.map_payload()
            self.assertFalse(
                any(point["kind"] == "learned" for point in after["points"])
            )
            self.assertEqual(runtime.sessions["session"].mode, "base")
            self.assertIsNone(runtime.sessions["session"].target)

    def test_delete_refits_pca_when_additional_set_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8"
            ) as file:
                for target, mu in (
                    ("gadget", [0.5, -0.25, 3.0]),
                    ("widget", [-0.5, 0.75, 4.0]),
                    ("device", [0.2, 0.4, 5.0]),
                ):
                    file.write(json.dumps({"target": target, "mu": mu}) + "\n")
            runtime = self._runtime(tmp)
            runtime.set_projection_sets(["additional_seen"])
            old_projection = runtime.projection

            with patch.object(
                runtime,
                "_effective_train_biases",
                return_value=runtime.train_biases.copy(),
            ):
                result = runtime.delete_learned_target(target="gadget")

            self.assertTrue(result["geometry_refreshed"])
            self.assertIsNot(runtime.projection, old_projection)
            self.assertEqual(
                {record["target"] for record in load_learned_biases(tmp)},
                {"widget", "device"},
            )

    def test_delete_rejects_invalidating_selected_pca_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps({"target": "gadget", "mu": [0.5, -0.25, 3.0]})
                    + "\n"
                )
                file.write(
                    json.dumps({"target": "widget", "mu": [-0.5, 0.75, 4.0]})
                    + "\n"
                )
            runtime = self._runtime(tmp)
            runtime.set_projection_sets(["additional_seen"])

            with self.assertRaisesRegex(ValueError, "fewer than two points"):
                runtime.delete_learned_target(target="gadget")

            self.assertEqual(
                {record["target"] for record in load_learned_biases(tmp)},
                {"gadget", "widget"},
            )

    def test_empty_space_uses_inverse_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.activate("session", kind="coordinate", x=0.3, y=-0.7)
            expected = runtime.projection.inverse(0.3, -0.7)
            np.testing.assert_allclose(
                runtime.sessions["session"].subspace, expected, rtol=1e-6
            )

    def test_train_point_keeps_word_id_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.activate("session", kind="train", point_id="train:7")
            self.assertEqual(runtime.sessions["session"].subspace, 7)
            self.assertEqual(runtime.sessions["session"].target, "laptop")

    def test_test_point_uses_exact_predicted_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.activate("session", kind="test", point_id="test:0")
            np.testing.assert_array_equal(
                runtime.sessions["session"].subspace,
                runtime.test_biases[0],
            )
            self.assertEqual(runtime.sessions["session"].target, "tablet")

    def test_train_and_test_points_prefer_latest_learned_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recovered = [0.9, -0.8, 0.7]
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "target": "tablet",
                            "mu": recovered,
                            "recovery_recall_converged": True,
                        }
                    )
                    + "\n"
                )
            runtime = self._runtime(tmp)
            runtime.activate("session", kind="test", point_id="test:0")
            np.testing.assert_array_equal(
                runtime.sessions["session"].subspace,
                np.asarray(recovered, dtype=np.float32),
            )

    def test_arbitrary_text_predicts_projects_and_caches_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            predicted = np.asarray([0.4, -0.2, 4.5], dtype=np.float32)
            with patch(
                "boreft.viz_server.predict_bias_vectors_from_raw_texts",
                return_value=[predicted],
            ) as predict:
                result = runtime.activate_text("session", "flat touchscreen")
                runtime.activate_text("session", "flat touchscreen")

            predict.assert_called_once()
            np.testing.assert_array_equal(
                runtime.sessions["session"].subspace,
                predicted,
            )
            expected = runtime.projection.project(predicted)
            np.testing.assert_allclose(
                [result["point"]["x"], result["point"]["y"]],
                expected,
            )

    def test_recovery_scan_marks_recall_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            with (
                patch("boreft.viz_server.generate_text", return_value="laptop"),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["not-tablet", "also-not-tablet"],
                ),
            ):
                started = runtime.start_recovery_scan(
                    scope="test",
                    n_samples=2,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=8,
                    seed=7,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(job["result"]["miss_point_ids"], ["test:0"])
            self.assertFalse(
                runtime.recovery_status["test:0"]["baseline_converged"]
            )

    def test_recovery_training_control_pause_resume_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            control = BatchLearnControl()
            runtime.recovery_jobs["job"] = {
                "id": "job",
                "type": "train",
                "status": "running",
                "phase": "training",
            }
            runtime.recovery_controls["job"] = control

            paused = runtime.control_recovery_training("job", "pause")
            self.assertTrue(control.paused)
            self.assertEqual(paused["training_state"], "pausing")

            resumed = runtime.control_recovery_training("job", "resume")
            self.assertFalse(control.paused)
            self.assertEqual(resumed["training_state"], "running")

            stopped = runtime.control_recovery_training("job", "stop")
            self.assertTrue(control.stop_requested)
            self.assertFalse(control.paused)
            self.assertEqual(stopped["training_state"], "stopping")

    def test_bias_network_learning_refreshes_train_rows_without_rehearsal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=5,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={"bias_learning_mode": "bias_network"},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.predict_bias_network_rows",
                    side_effect=[
                        (
                            np.asarray([[1.1, 0.1, 4.1]], dtype=np.float32),
                            None,
                        ),
                        (
                            np.asarray([[0.8, -0.3, 4.7]], dtype=np.float32),
                            None,
                        ),
                    ],
                ) as refresh,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet", "other"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
                patch("boreft.viz_server.save_bias_tables"),
                patch("boreft.viz_server.attach_materialized_bias_tables"),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=5,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=True,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(
                learn.call_args.kwargs["targets"], ["tablet"]
            )
            self.assertEqual(
                [call.args[1] for call in refresh.call_args_list],
                [["laptop"], ["tablet"]],
            )
            self.assertEqual(job["result"]["n_trained"], 1)
            self.assertFalse(job["result"]["geometry_refreshed"])
            np.testing.assert_allclose(
                runtime.train_biases,
                np.asarray([[1.1, 0.1, 4.1]], dtype=np.float32),
            )
            np.testing.assert_allclose(
                runtime.test_base_biases,
                np.asarray([[0.8, -0.3, 4.7]], dtype=np.float32),
            )

    def test_recovery_training_forwards_eval_and_stop_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=3,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={},
                eval_history=[
                    {"epoch": 2, "avg_embed_sim": 0.7, "min_embed_sim": 0.5}
                ],
                stopped_early=True,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=10,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=0.0,
                    eval_epochs=2,
                    stop_threshold=0.85,
                    stop_threshold_min=0.6,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            config = learn.call_args.kwargs["config"]
            self.assertEqual(config.eval_epochs, 2)
            self.assertEqual(config.stop_threshold, 0.85)
            self.assertEqual(config.stop_threshold_min, 0.6)
            self.assertEqual(config.kl_beta, 0.0)
            self.assertFalse(config.inherit_stop_threshold)
            self.assertEqual(
                job["result"]["eval_history"][0]["avg_embed_sim"], 0.7
            )
            self.assertTrue(job["result"]["stopped_early"])

    def test_recovery_training_forwards_loss_module_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.ckpt.saved_cfg = {
                "seed": 42,
                "lambda_ce": 1.0,
                "lambda_sdpo": 0.5,
                "kl_beta": 0.1,
                "weight_decay_mode": "W_and_b",
                "wd_W": 1e-4,
                "wd_b": 1e-4,
            }
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=2,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=3,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=0.0,
                    lambda_ce=0.0,
                    lambda_sdpo=0.0,
                    weight_decay_mode="none",
                    wd_W=None,
                    wd_b=None,
                    eval_epochs=None,
                    stop_threshold=None,
                    stop_threshold_min=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            config = learn.call_args.kwargs["config"]
            self.assertEqual(config.lambda_ce, 0.0)
            self.assertEqual(config.lambda_sdpo, 0.0)
            self.assertEqual(config.kl_beta, 0.0)
            self.assertEqual(config.weight_decay_mode, "none")

    def test_recovery_training_forwards_scheduler_and_wd_coeffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=2,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=3,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    lambda_ce=None,
                    lambda_sdpo=0.0,
                    weight_decay_mode="W_and_b",
                    wd_W=1e-3,
                    wd_b=2e-3,
                    lr_scheduler_type="cosine",
                    warmup_ratio=0.1,
                    eval_epochs=None,
                    stop_threshold=None,
                    stop_threshold_min=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                    training_sets=["test_unseen"],
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            config = learn.call_args.kwargs["config"]
            self.assertEqual(config.weight_decay_mode, "W_and_b")
            self.assertEqual(config.wd_W, 1e-3)
            self.assertEqual(config.wd_b, 2e-3)
            self.assertEqual(config.lr_scheduler_type, "cosine")
            self.assertEqual(config.warmup_ratio, 0.1)
            self.assertIn("loss_history", job["result"])
            self.assertEqual(
                job["result"]["training_sets"], ["test_unseen"]
            )

    def test_recovery_training_forwards_annealing_map_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.ckpt.saved_cfg = {
                **(runtime.ckpt.saved_cfg or {}),
                "kl_beta": 0.1,
                "linear_annealing_map": {"kl_beta": [0.0, 0.1, 50]},
            }
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=2,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={
                    "linear_annealing_map": {"kl_beta": [0.0, 0.1, 10]},
                    "kl_beta": 0.1,
                    "lambda_ce": 1.0,
                    "lambda_sdpo": 0.0,
                },
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=3,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=0.1,
                    lambda_ce=None,
                    lambda_sdpo=0.0,
                    linear_annealing_map="kl_beta=0:10",
                    weight_decay_mode=None,
                    wd_W=None,
                    wd_b=None,
                    eval_epochs=None,
                    stop_threshold=None,
                    stop_threshold_min=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            config = learn.call_args.kwargs["config"]
            self.assertEqual(config.linear_annealing_map, "kl_beta=0:10")
            self.assertEqual(
                runtime.ckpt.saved_cfg["linear_annealing_map"],
                {"kl_beta": [0.0, 0.1, 10]},
            )

            payload = runtime.map_payload()
            self.assertEqual(
                payload["metadata"]["linear_annealing_map"],
                {"kl_beta": [0.0, 0.1, 10]},
            )
            self.assertEqual(
                payload["metadata"]["linear_annealing_map_cli"],
                "kl_beta=0:0.1:10",
            )

    def test_recovery_training_rejects_invalid_annealing_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            with self.assertRaisesRegex(ValueError, "invalid linear_annealing_map"):
                runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=3,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                    linear_annealing_map="lambda_mm=0:1:5",
                )

    def test_recovery_training_rejects_stop_without_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            with self.assertRaisesRegex(ValueError, "eval every N epochs"):
                runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=10,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    stop_threshold=0.85,
                    stop_threshold_min=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )

    def test_recovery_training_persists_result_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.recovery_status["test:0"] = {
                "baseline_scanned": True,
                "baseline_converged": False,
            }
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=12,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.1,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )

            def learn_batch(*args, **kwargs):
                kwargs["progress_callback"](
                    {
                        "event": "log",
                        "step": 3,
                        "max_steps": 12,
                        "epoch": 1.0,
                        "loss": 0.25,
                        "learning_rate": 0.01,
                    }
                )
                return learned

            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    side_effect=learn_batch,
                ) as learn,
                patch("boreft.viz_server.generate_text", return_value="tablet"),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet", "other"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=20,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=5,
                    learn_W=True,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=False,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            self.assertTrue(
                runtime.recovery_status["test:0"]["learned_converged"]
            )
            learn.assert_called_once()
            self.assertEqual(job["result"]["mean_train_loss"], 0.1)
            self.assertTrue(
                learn.call_args.kwargs["config"].learn_W
            )
            self.assertEqual(
                learn.call_args.kwargs["config"].epochs, 20
            )
            self.assertIsInstance(
                learn.call_args.kwargs["training_control"],
                BatchLearnControl,
            )

    def test_batch_training_recall_miss_remains_unconverged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.recovery_status["test:0"] = {
                "baseline_scanned": True,
                "baseline_converged": False,
                "n_samples": 2,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 8,
                "seed": 7,
            }
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=12,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.1,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ),
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet-like",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["other", "computer"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=20,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=5,
                    learn_W=False,
                    learn_R=True,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=False,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["result"]["n_converged"], 0)
            self.assertFalse(
                runtime.recovery_status["test:0"]["learned_converged"]
            )
            self.assertNotIn(
                "optimizer_converged", runtime.recovery_status["test:0"]
            )

    def test_new_target_rehearses_all_previously_trained_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(
                        {
                            "target": "tablet",
                            "definition": "a flat touchscreen computer",
                            "mu": [0.2, -0.4, 5.0],
                            "new_target": False,
                        }
                    )
                    + "\n"
                )
            runtime = self._runtime(tmp)
            learned = SimpleNamespace(
                targets=["laptop", "tablet", "notebook"],
                steps=8,
                mu=np.asarray(
                    [
                        [1.1, 0.0, 4.0],
                        [0.3, -0.3, 4.8],
                        [0.5, 0.2, 4.2],
                    ],
                    dtype=np.float32,
                ),
                mu_pred=np.asarray(
                    [
                        [1.0, 0.0, 4.0],
                        [0.2, -0.4, 5.0],
                        [0.4, 0.1, 4.0],
                    ],
                    dtype=np.float32,
                ),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.predict_bias_vectors_for_words",
                    return_value=[
                        np.asarray([0.4, 0.1, 4.0], dtype=np.float32)
                    ],
                ),
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ) as learn,
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="notebook",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["notebook", "other"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    [],
                    new_target="notebook",
                    new_definition="a book of blank ruled pages",
                    epochs=10,
                    batch_size=8,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    learn_W=True,
                    learn_R=True,
                    learn_bias_network=False,
                    include_previous_targets=True,
                    force=False,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(
                learn.call_args.kwargs["targets"],
                ["laptop", "tablet", "notebook"],
            )
            self.assertEqual(job["result"]["n_trained"], 3)
            self.assertEqual(job["result"]["n_evaluated"], 1)
            self.assertTrue(job["result"]["learn_W"])
            self.assertTrue(job["result"]["learn_R"])
            records = load_learned_biases(tmp)
            self.assertTrue(records[-1]["new_target"])
            self.assertEqual(records[-1]["target"], "notebook")

    def test_new_target_with_test_projection_set_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.projection_sets = ["test_unseen"]
            learned = SimpleNamespace(
                targets=["notebook"],
                steps=2,
                mu=np.asarray([[0.5, 0.2, 4.2]], dtype=np.float32),
                mu_pred=np.asarray([[0.4, 0.1, 4.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.2,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )
            with (
                patch(
                    "boreft.viz_server.predict_bias_vectors_for_words",
                    return_value=[
                        np.asarray([0.4, 0.1, 4.0], dtype=np.float32)
                    ],
                ),
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    return_value=learned,
                ),
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="notebook",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["notebook"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ),
            ):
                started = runtime.start_recovery_training(
                    [],
                    new_target="notebook",
                    new_definition="a book of blank ruled pages",
                    epochs=2,
                    batch_size=1,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    learn_W=False,
                    learn_R=False,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(job["result"]["n_trained"], 1)

    def test_recovery_training_reports_parameter_change_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            intervention = runtime.ckpt.reft_model.interventions["test"]
            learned = SimpleNamespace(
                targets=["tablet"],
                steps=4,
                mu=np.asarray([[0.4, -0.2, 4.5]], dtype=np.float32),
                mu_pred=np.asarray([[0.2, -0.4, 5.0]], dtype=np.float32),
                logvar=None,
                bias_dim=3,
                mean_train_loss=0.15,
                loss_config={},
                resolved_train_config={},
                eval_history=[],
                stopped_early=False,
            )

            def mutate_and_return(*args, **kwargs):
                with torch.no_grad():
                    intervention.learned_source.weight.add_(0.5)
                    intervention.rotate_layer.weight.add_(0.25)
                return learned

            with (
                patch(
                    "boreft.viz_server.learn_biases_batched",
                    side_effect=mutate_and_return,
                ),
                patch(
                    "boreft.viz_server.generate_text",
                    return_value="tablet",
                ),
                patch(
                    "boreft.viz_server.generate_texts_batch",
                    return_value=["tablet", "other"],
                ),
                patch.object(
                    runtime, "_save_derived_checkpoint", return_value=tmp
                ) as save_ckpt,
            ):
                started = runtime.start_recovery_training(
                    ["test:0"],
                    new_target=None,
                    new_definition=None,
                    epochs=3,
                    batch_size=4,
                    lr=0.01,
                    kl_beta=None,
                    eval_epochs=None,
                    learn_W=True,
                    learn_R=True,
                    learn_bias_network=False,
                    include_previous_targets=False,
                    force=True,
                )
                job = self._wait_job(runtime, started["id"])

            self.assertEqual(job["status"], "completed", job)
            changes = job["result"]["parameter_changes"]
            self.assertIn("vs_previous", changes)
            self.assertIn("vs_original", changes)
            self.assertEqual(changes["vs_previous"]["W"]["status"], "changed")
            self.assertGreater(changes["vs_previous"]["W"]["relative_l2"], 0.0)
            self.assertEqual(changes["vs_previous"]["R"]["status"], "changed")
            self.assertEqual(
                changes["vs_previous"]["bias_network"]["status"], "unchanged"
            )
            self.assertIsNone(
                changes["vs_previous"]["bias_network"]["relative_l2"]
            )
            self.assertEqual(
                save_ckpt.call_args.kwargs["parameter_changes"], changes
            )

    def test_derived_checkpoint_preserves_source_and_writes_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = f"{tmp}/checkpoint"
            os.makedirs(f"{source}/intervenable_model")
            with open(
                f"{source}/intervenable_model/original.txt",
                "w",
                encoding="utf-8",
            ) as file:
                file.write("original")
            with open(
                f"{source}/intervention_config.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({"output_dir": source}, file)

            runtime = self._runtime(source)

            class FakeReftModel:
                def save_intervention(
                    self, *, save_directory, include_model
                ):
                    self.assert_include_model = include_model
                    os.makedirs(save_directory)
                    with open(
                        f"{save_directory}/updated.txt",
                        "w",
                        encoding="utf-8",
                    ) as file:
                        file.write("updated")

            runtime.ckpt.reft_model = FakeReftModel()
            destination = runtime._save_derived_checkpoint(
                learn_W=True,
                learn_R=False,
                learn_bias_network=False,
                include_previous_targets=True,
                n_targets=3,
            )

            self.assertTrue(
                os.path.isfile(
                    f"{source}/intervenable_model/original.txt"
                )
            )
            self.assertTrue(
                os.path.isfile(
                    f"{destination}/intervenable_model/updated.txt"
                )
            )
            with open(
                f"{destination}/recovery_info.json", encoding="utf-8"
            ) as file:
                info = json.load(file)
            self.assertEqual(info["parent_checkpoint"], source)
            self.assertTrue(info["learn_W"])
            self.assertEqual(info["n_targets"], 3)

    def test_refresh_subspace_geometry_refits_pca_when_train_bias_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime.word_ids = [7, 8]
            runtime.words = ["laptop", "desktop"]
            runtime.item_original_splits = ["train", "train"]
            runtime.train_biases = np.asarray(
                [[1.0, 0.0, 4.0], [-1.0, 0.0, 4.0]], dtype=np.float32
            )
            runtime.projection = fit_projection(runtime.train_biases)
            runtime.train_points = [
                {
                    "id": "train:7",
                    "kind": "train",
                    "label": "laptop",
                    "x": float(runtime.projection.project(runtime.train_biases[0])[0]),
                    "y": float(runtime.projection.project(runtime.train_biases[0])[1]),
                    "cluster": "all",
                    "metadata": {"word_id": 7},
                },
                {
                    "id": "train:8",
                    "kind": "train",
                    "label": "desktop",
                    "x": float(runtime.projection.project(runtime.train_biases[1])[0]),
                    "y": float(runtime.projection.project(runtime.train_biases[1])[1]),
                    "cluster": "all",
                    "metadata": {"word_id": 8},
                },
            ]
            old_projection = runtime.projection
            old_train_xy = (runtime.train_points[0]["x"], runtime.train_points[0]["y"])
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(
                    json.dumps({"target": "laptop", "mu": [0.0, 2.0, 4.0]}) + "\n"
                )
                file.write(
                    json.dumps({"target": "desktop", "mu": [0.0, -2.0, 4.0]}) + "\n"
                )
                file.write(
                    json.dumps({"target": "tablet", "mu": [0.5, -0.5, 5.0]}) + "\n"
                )

            refreshed = runtime._refresh_subspace_geometry()

            self.assertTrue(refreshed)
            np.testing.assert_array_equal(
                runtime.train_biases,
                np.asarray([[0.0, 2.0, 4.0], [0.0, -2.0, 4.0]], dtype=np.float32),
            )
            self.assertIsNot(runtime.projection, old_projection)
            new_train_xy = (
                runtime.train_points[0]["x"],
                runtime.train_points[0]["y"],
            )
            self.assertNotEqual(new_train_xy, old_train_xy)
            expected_test = runtime.projection.project(
                np.asarray([0.5, -0.5, 5.0], dtype=np.float32)
            )
            np.testing.assert_allclose(
                [runtime.test_points[0]["x"], runtime.test_points[0]["y"]],
                expected_test,
            )

    def test_refresh_subspace_geometry_skips_pca_when_train_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            # Learned overlay matches the current train bias → no PCA refit.
            with open(f"{tmp}/learned_biases.jsonl", "w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "target": "laptop",
                            "mu": [1.0, 0.0, 4.0],
                        }
                    )
                    + "\n"
                )
                file.write(
                    json.dumps(
                        {
                            "target": "tablet",
                            "mu": [9.0, 9.0, 9.0],
                        }
                    )
                    + "\n"
                )
            old_projection = runtime.projection

            refreshed = runtime._refresh_subspace_geometry()

            self.assertFalse(refreshed)
            self.assertIs(runtime.projection, old_projection)
            expected_test = old_projection.project(
                np.asarray([9.0, 9.0, 9.0], dtype=np.float32)
            )
            np.testing.assert_allclose(
                [runtime.test_points[0]["x"], runtime.test_points[0]["y"]],
                expected_test,
            )


class ChatSimilarityTests(unittest.TestCase):
    def _runtime(self, task: str = "semantle") -> VizRuntime:
        runtime = object.__new__(VizRuntime)
        runtime.task = task
        runtime.ckpt = SimpleNamespace(saved_cfg={"task": task})
        runtime.word_to_id = {"laptop": 0}
        runtime.lock = threading.RLock()
        runtime.sessions = {}
        runtime.model_name = "test-model"
        return runtime

    def test_similarity_vs_target_semantle_embed_only(self) -> None:
        runtime = self._runtime("semantle")
        with patch(
            "boreft.viz_server.embedding_sim_per_text",
            return_value=np.asarray([0.812], dtype=np.float64),
        ) as embed_mock:
            scores, error = runtime.similarity_vs_target(
                "<think>noise</think>laptop", "laptop"
            )
        embed_mock.assert_called_once_with(
            ["laptop"], ["laptop"], task="semantle"
        )
        self.assertEqual(scores, {"embed_sim": 0.812})
        self.assertIsNone(error)

    def test_similarity_vs_target_molopt_includes_structural(self) -> None:
        runtime = self._runtime("molopt")
        with (
            patch(
                "boreft.viz_server.embedding_sim_per_text",
                return_value=np.asarray([0.7], dtype=np.float64),
            ),
            patch(
                "boreft.chem.tanimoto_sim_per_text",
                return_value=np.asarray([0.45], dtype=np.float64),
            ) as tfs_mock,
            patch(
                "boreft.chem.rdkit_sim_per_text",
                return_value=np.asarray([0.55], dtype=np.float64),
            ) as rdkit_mock,
            patch(
                "boreft.viz_server.rdkit_map_path_for_cfg",
                return_value="/tmp/map.json",
            ),
        ):
            scores, error = runtime.similarity_vs_target("CCO", "CCO")
        tfs_mock.assert_called_once_with(["CCO"], ["CCO"])
        rdkit_mock.assert_called_once_with(
            ["CCO"], ["CCO"], map_path="/tmp/map.json"
        )
        self.assertEqual(
            scores,
            {"embed_sim": 0.7, "tfs": 0.45, "rdkit_sim": 0.55},
        )
        self.assertIsNone(error)

    def test_similarity_vs_target_molopt_keeps_partial_on_rdkit_error(
        self,
    ) -> None:
        runtime = self._runtime("molopt")
        with (
            patch(
                "boreft.viz_server.embedding_sim_per_text",
                return_value=np.asarray([0.7], dtype=np.float64),
            ),
            patch(
                "boreft.chem.tanimoto_sim_per_text",
                return_value=np.asarray([0.45], dtype=np.float64),
            ),
            patch(
                "boreft.viz_server.rdkit_map_path_for_cfg",
                side_effect=ValueError("map provenance mismatch"),
            ),
        ):
            scores, error = runtime.similarity_vs_target("CCO", "CCO")
        self.assertEqual(scores, {"embed_sim": 0.7, "tfs": 0.45})
        self.assertIn("rdkit_sim:", error)
        self.assertIn("map provenance mismatch", error)

    def test_chat_omits_similarity_when_toggle_off_or_no_target(self) -> None:
        runtime = self._runtime("semantle")
        runtime.ckpt = SimpleNamespace(
            saved_cfg={"position": "l1"},
            tokenizer=object(),
            reft_model=SimpleNamespace(),
            assistant_suffix=None,
            intervention_token_id=None,
        )
        session = BrowserSession()
        session.mode = "intervention"
        session.subspace = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        session.label = "laptop"
        session.target = "laptop"
        runtime.sessions["s"] = session

        with (
            patch(
                "boreft.viz_server.build_checkpoint_prompt",
                return_value=("prompt", False, None),
            ),
            patch("boreft.viz_server.generate_text", return_value="laptop"),
            patch(
                "boreft.viz_server.embedding_sim_per_text",
                return_value=np.asarray([1.0], dtype=np.float64),
            ) as embed_mock,
        ):
            off = runtime.chat(
                "s",
                text="hi",
                multi_turn=False,
                use_checkpoint_prompt=False,
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                compute_similarity=False,
            )
            self.assertNotIn("similarity", off)
            embed_mock.assert_not_called()

            session.target = None
            no_target = runtime.chat(
                "s",
                text="hi",
                multi_turn=False,
                use_checkpoint_prompt=False,
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                compute_similarity=True,
            )
            self.assertNotIn("similarity", no_target)
            embed_mock.assert_not_called()

            session.target = "laptop"
            on = runtime.chat(
                "s",
                text="hi",
                multi_turn=False,
                use_checkpoint_prompt=False,
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                compute_similarity=True,
            )
            self.assertEqual(on["similarity"], {"embed_sim": 1.0})
            embed_mock.assert_called_once()

    def test_chat_keeps_reply_when_similarity_fails(self) -> None:
        runtime = self._runtime("semantle")
        runtime.ckpt = SimpleNamespace(
            saved_cfg={"position": "l1"},
            tokenizer=object(),
            reft_model=SimpleNamespace(),
            assistant_suffix=None,
            intervention_token_id=None,
        )
        session = BrowserSession()
        session.mode = "intervention"
        session.subspace = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        session.label = "laptop"
        session.target = "laptop"
        runtime.sessions["s"] = session

        with (
            patch(
                "boreft.viz_server.build_checkpoint_prompt",
                return_value=("prompt", False, None),
            ),
            patch("boreft.viz_server.generate_text", return_value="laptop"),
            patch(
                "boreft.viz_server.embedding_sim_per_text",
                side_effect=RuntimeError("embed model missing"),
            ),
        ):
            result = runtime.chat(
                "s",
                text="hi",
                multi_turn=False,
                use_checkpoint_prompt=False,
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                compute_similarity=True,
            )
        self.assertEqual(result["text"], "laptop")
        self.assertNotIn("similarity", result)
        self.assertEqual(
            result["similarity_error"], "embed_sim: embed model missing"
        )

    def test_chat_returns_partial_similarity_when_rdkit_fails(self) -> None:
        runtime = self._runtime("molopt")
        runtime.ckpt = SimpleNamespace(
            saved_cfg={"position": "l1", "task": "molopt"},
            tokenizer=object(),
            reft_model=SimpleNamespace(),
            assistant_suffix=None,
            intervention_token_id=None,
        )
        session = BrowserSession()
        session.mode = "intervention"
        session.subspace = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        session.label = "CCO"
        session.target = "CCO"
        runtime.sessions["s"] = session

        with (
            patch(
                "boreft.viz_server.build_checkpoint_prompt",
                return_value=("prompt", False, None),
            ),
            patch("boreft.viz_server.generate_text", return_value="CCO"),
            patch(
                "boreft.viz_server.embedding_sim_per_text",
                return_value=np.asarray([0.8], dtype=np.float64),
            ),
            patch(
                "boreft.chem.tanimoto_sim_per_text",
                return_value=np.asarray([0.5], dtype=np.float64),
            ),
            patch(
                "boreft.viz_server.rdkit_map_path_for_cfg",
                side_effect=ValueError("map provenance mismatch"),
            ),
        ):
            result = runtime.chat(
                "s",
                text="hi",
                multi_turn=False,
                use_checkpoint_prompt=False,
                enable_thinking=False,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                compute_similarity=True,
            )
        self.assertEqual(result["text"], "CCO")
        self.assertEqual(result["similarity"], {"embed_sim": 0.8, "tfs": 0.5})
        self.assertIn("rdkit_sim:", result["similarity_error"])


if __name__ == "__main__":
    unittest.main()
