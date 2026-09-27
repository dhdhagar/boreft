"""Tests for LoRA SFT train-word draw (same as boreft.train)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from boreft.baselines.lora_sft import (
    SEARCH_TASK_DESCRIPTION,
    build_sft_rows,
    attach_lora_sft_adapter,
    discover_wandb_identity,
    epoch_adapter_path,
    load_semantle_train_words,
    resolve_sft_instruction,
    resolved_prompt_source,
    save_adapter_checkpoint,
    should_eval_epoch,
    parse_args,
    prepare_sft_items,
    proposal_eval_stats,
    resolve_train_word_draw,
    train,
    user_instruction,
    words_from_items_json,
)
from boreft.data.base import item_target
from boreft.data.semantle import SemantleItem
from boreft.data_utils import sample_reft_items
from boreft.task_config import task_instruction
from boreft.train import load_training_items
from boreft.train_args import TrainConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_CSV = _REPO_ROOT / "data" / "semantle" / "train" / "computer.csv"


def _write_csv(path: Path, words: list[str]) -> None:
    path.write_text(
        "Word,Similarity\n"
        + "".join(f"{word},{1.0 - i * 0.01}\n" for i, word in enumerate(words)),
        encoding="utf-8",
    )


class SemantleTrainWordDrawTest(unittest.TestCase):
    def test_matches_sample_reft_items_and_load_training_items(self):
        words = ["alpha", "bravo", "charlie", "delta", "echo"]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "puzzle.csv"
            _write_csv(csv_path, words)
            drawn = load_semantle_train_words(
                [str(csv_path)],
                train_top_k=5,
                train_n_samples=3,
                seed=42,
            )
            _, all_words, sim_map = SemantleItem.load_csvs([str(csv_path)], top_k=5)
            items = SemantleItem.load(all_words, sim_map)
            sampled, _ = sample_reft_items(items, 3, seed=42)
            self.assertEqual(drawn, [item_target(item) for item in sampled])

            config = TrainConfig(
                task="semantle",
                semantle_csv=(str(csv_path),),
                train_top_k=5,
                train_n_samples=3,
                seed=42,
                output_dir=str(Path(tmp) / "out"),
            )
            train_items, _ = load_training_items(config)
            self.assertEqual(drawn, [item_target(item) for item in train_items])
            self.assertEqual(len(drawn), 3)
            self.assertEqual(len(set(drawn)), 3)

    def test_top_k_then_sample_uses_the_truncated_csv(self):
        words = ["alpha", "bravo", "charlie", "delta", "echo"]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "puzzle.csv"
            _write_csv(csv_path, words)
            drawn = load_semantle_train_words(
                [str(csv_path)],
                train_top_k=3,
                train_n_samples=2,
                seed=0,
            )
            self.assertTrue(set(drawn).issubset({"alpha", "bravo", "charlie"}))
            self.assertEqual(len(drawn), 2)

    def test_checkpoint_items_must_match_the_draw(self):
        words = ["alpha", "bravo", "charlie", "delta"]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "puzzle.csv"
            _write_csv(csv_path, words)
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "training_config.json").write_text(
                json.dumps(
                    {
                        "semantle_csv": [str(csv_path)],
                        "train_top_k": 4,
                        "train_n_samples": 2,
                        "seed": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected = load_semantle_train_words(
                [str(csv_path)], train_top_k=4, train_n_samples=2, seed=1
            )
            (ckpt / "items.json").write_text(
                json.dumps([{"id": i, "word": w} for i, w in enumerate(expected)]),
                encoding="utf-8",
            )
            draw = resolve_train_word_draw(
                semantle_csv=None,
                train_top_k=None,
                train_n_samples=None,
                seed=None,
                reft_output_dir=str(ckpt),
                from_checkpoint_items=False,
                skip_items_check=False,
            )
            self.assertEqual(draw.words, expected)
            self.assertEqual(draw.train_n_samples, 2)
            self.assertTrue(draw.matched_checkpoint_items)

            with self.assertRaisesRegex(ValueError, "do not match checkpoint"):
                resolve_train_word_draw(
                    semantle_csv=None,
                    train_top_k=4000,
                    train_n_samples=3072,
                    seed=42,
                    reft_output_dir=str(ckpt),
                    from_checkpoint_items=False,
                    skip_items_check=False,
                )

            (ckpt / "items.json").write_text(
                json.dumps([{"id": 0, "word": "nope"}, {"id": 1, "word": "nah"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "do not match checkpoint"):
                resolve_train_word_draw(
                    semantle_csv=None,
                    train_top_k=4000,
                    train_n_samples=3072,
                    seed=42,
                    reft_output_dir=str(ckpt),
                    from_checkpoint_items=False,
                    skip_items_check=False,
                )

    def test_from_checkpoint_items_uses_items_json_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "items.json").write_text(
                json.dumps(
                    [
                        {"id": 0, "word": "zebra"},
                        {"id": 1, "target": "yak"},
                    ]
                ),
                encoding="utf-8",
            )
            (ckpt / "training_config.json").write_text("{}\n", encoding="utf-8")
            draw = resolve_train_word_draw(
                semantle_csv=None,
                train_top_k=None,
                train_n_samples=None,
                seed=None,
                reft_output_dir=str(ckpt),
                from_checkpoint_items=True,
                skip_items_check=False,
            )
            self.assertEqual(draw.words, ["zebra", "yak"])
            self.assertEqual(draw.source, "items.json")

    def test_words_from_items_json_prefers_word_over_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            path.write_text(
                json.dumps([{"word": "cat", "target": "  CAT\\n extra"}]),
                encoding="utf-8",
            )
            self.assertEqual(words_from_items_json(path), ["cat"])

    def test_search_instruction_is_the_random_task_description(self):
        text = "Generate an English word as a guess to find the hidden word."
        self.assertEqual(
            user_instruction(
                prompt_source="search",
                task_description=text,
                use_chat_template=True,
            ),
            text,
        )
        self.assertEqual(
            user_instruction(
                prompt_source="train",
                task_description=text,
                use_chat_template=True,
            ),
            task_instruction("semantle", use_chat_template=True),
        )

    def test_sft_items_use_search_prompt_not_train_prompt(self):
        train_prompt = task_instruction("semantle", use_chat_template=False)
        items = prepare_sft_items(
            ["alpha"],
            SEARCH_TASK_DESCRIPTION,
            use_chat_template=False,
            system_prompt=None,
        )
        self.assertEqual(items[0].prompt, SEARCH_TASK_DESCRIPTION)
        self.assertEqual(items[0].target, "alpha")
        self.assertNotEqual(items[0].prompt, train_prompt)

    def test_proposal_eval_stats_detects_shift_and_collapse(self):
        train = ["alpha", "bravo", "charlie"]
        shifted = proposal_eval_stats(
            ["Alpha", "bravo", "delta", "bravo"],
            train,
        )
        self.assertEqual(shifted["n"], 4)
        self.assertAlmostEqual(shifted["train_hit_rate"], 0.75)
        self.assertEqual(shifted["n_unique_in_train"], 2)
        self.assertAlmostEqual(shifted["train_coverage"], 2 / 3)

        collapsed = proposal_eval_stats(["alpha"] * 8, train)
        self.assertAlmostEqual(collapsed["train_hit_rate"], 1.0)
        self.assertAlmostEqual(collapsed["unique_rate"], 0.125)
        self.assertLess(collapsed["shift_score"], shifted["shift_score"])

        blank = proposal_eval_stats(["", "  "], train)
        self.assertAlmostEqual(blank["blank_rate"], 1.0)
        self.assertEqual(blank["train_hit_rate"], 0.0)

    def test_should_eval_epoch_hits_interval_and_last(self):
        self.assertTrue(should_eval_epoch(5, epochs=256, eval_epochs=5))
        self.assertTrue(should_eval_epoch(255, epochs=256, eval_epochs=5))
        self.assertTrue(should_eval_epoch(256, epochs=256, eval_epochs=5))
        self.assertFalse(should_eval_epoch(4, epochs=256, eval_epochs=5))
        self.assertFalse(should_eval_epoch(1, epochs=256, eval_epochs=5))
        self.assertTrue(should_eval_epoch(4, epochs=4, eval_epochs=5))
        self.assertFalse(should_eval_epoch(5, epochs=256, eval_epochs=0))
        self.assertTrue(should_eval_epoch(1, epochs=10, eval_epochs=1))
        self.assertTrue(should_eval_epoch(10, epochs=10, eval_epochs=1))
        self.assertFalse(should_eval_epoch(2, epochs=10, eval_epochs=0))

    def test_epoch_adapter_path_is_zero_padded(self):
        self.assertEqual(
            epoch_adapter_path(Path("/tmp/out"), 7),
            Path("/tmp/out") / "adapters" / "epoch007.pt",
        )

    def test_cli_omitted_knobs_resolve_to_the_canonical_draw(self):
        args = parse_args([])
        self.assertIsNone(args.train_top_k)
        self.assertIsNone(args.train_n_samples)
        self.assertIsNone(args.seed)
        self.assertIsNone(args.prompt_source)
        self.assertIsNone(args.task_description)
        self.assertIsNone(args.output_dir)
        self.assertEqual(resolved_prompt_source(args.prompt_source, "semantle"), "search")
        self.assertEqual(resolved_prompt_source(args.prompt_source, "molopt"), "train")
        self.assertEqual(args.n_eval, 500)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.eval_epochs, 1)
        self.assertEqual(args.save_epochs, 1)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "puzzle.csv"
            _write_csv(csv_path, ["alpha", "bravo", "charlie", "delta"])
            draw = resolve_train_word_draw(
                semantle_csv=[str(csv_path)],
                train_top_k=args.train_top_k,
                train_n_samples=args.train_n_samples,
                seed=args.seed,
                reft_output_dir=None,
                from_checkpoint_items=False,
                skip_items_check=False,
            )
        self.assertEqual(draw.train_top_k, 4000)
        self.assertEqual(draw.train_n_samples, 3072)
        self.assertEqual(draw.seed, 42)

    def test_sft_rows_mask_only_the_prompt_prefix(self):
        class _CharTok:
            eos_token = ""

            def __call__(self, text, add_special_tokens=True, return_tensors=None):
                ids = torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        rows = build_sft_rows(
            ["ab"],
            _CharTok(),
            "xy",
            use_chat_template=False,
            system_prompt=None,
        )
        self.assertEqual(len(rows), 1)
        labels = rows[0]["labels"]
        input_ids = rows[0]["input_ids"]
        self.assertTrue(torch.equal(input_ids[:2], torch.tensor([ord("x"), ord("y")])))
        self.assertTrue(torch.all(labels[:2] < 0))
        self.assertTrue(torch.all(labels[2:] >= 0))

    def test_dry_run_does_not_clobber_a_nonempty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "puzzle.csv"
            _write_csv(csv_path, ["alpha", "bravo", "charlie", "delta"])
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "adapter.pt").write_bytes(b"x")
            with self.assertRaisesRegex(FileExistsError, "is not empty"):
                train(
                    parse_args(
                        [
                            "--dry-run",
                            "--no-wandb",
                            "--output-dir",
                            str(out),
                            "--semantle-csv",
                            str(csv_path),
                            "--train-n-samples",
                            "2",
                            "--train-top-k",
                            "4",
                        ]
                    )
                )

    def test_canonical_3072_draw_matches_load_training_items(self):
        if not _CANONICAL_CSV.is_file():
            self.skipTest("data/semantle/train/computer.csv is missing")
        csv_path = str(_CANONICAL_CSV)
        drawn = load_semantle_train_words(
            [csv_path],
            train_top_k=4000,
            train_n_samples=3072,
            seed=42,
        )
        self.assertEqual(len(drawn), 3072)
        self.assertEqual(len(set(drawn)), 3072)

        config = TrainConfig(
            task="semantle",
            semantle_csv=(csv_path,),
            train_top_k=4000,
            train_n_samples=3072,
            seed=42,
            output_dir=str(_REPO_ROOT / "outputs" / "unused-lora-sft-test"),
        )
        train_items, _ = load_training_items(config)
        self.assertEqual(drawn, [item_target(item) for item in train_items])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            train(
                parse_args(
                    [
                        "--dry-run",
                        "--no-wandb",
                        "--overwrite",
                        "--output-dir",
                        str(out),
                        "--semantle-csv",
                        csv_path,
                    ]
                )
            )
            payload = json.loads((out / "words.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["words"], drawn)
            self.assertEqual(payload["n"], 3072)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["train_top_k"], 4000)
            self.assertEqual(payload["source"], "sample_reft_items")

    def test_eval_only_requires_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "words.json").write_text(
                json.dumps({"words": ["alpha"], "seed": 0}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "adapter.pt"):
                train(
                    parse_args(
                        [
                            "--eval-only",
                            "--no-wandb",
                            "--output-dir",
                            str(out),
                        ]
                    )
                )

    def test_eval_only_refuses_a_new_wandb_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "adapter.pt").write_bytes(b"x")
            (out / "words.json").write_text(
                json.dumps({"words": ["alpha"], "seed": 0}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "refusing to"):
                train(
                    parse_args(
                        [
                            "--eval-only",
                            "--output-dir",
                            str(out),
                        ]
                    )
                )


class DiscoverWandbIdentityTest(unittest.TestCase):
    def test_prefers_wandb_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "wandb_meta.json").write_text(
                json.dumps(
                    {"run_id": "abc123xy", "project": "boreft", "entity": "example"}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                discover_wandb_identity(out),
                {"run_id": "abc123xy", "project": "boreft", "entity": "example"},
            )

    def test_reads_run_id_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "wandb_run_id.txt").write_text("zz99run1\n", encoding="utf-8")
            self.assertEqual(discover_wandb_identity(out), {"run_id": "zz99run1"})

    def test_parses_wandb_run_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = out / "wandb" / "run-20260914_153159-puhyggoo"
            run_dir.mkdir(parents=True)
            self.assertEqual(
                discover_wandb_identity(out),
                {"run_id": "puhyggoo"},
            )

    def test_parses_wandb_url_from_debug_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = out / "wandb" / "run-oddname"
            run_dir.mkdir(parents=True)
            (run_dir / "debug.log").write_text(
                "View run at https://wandb.ai/example/boreft/runs/puhyggoo\n",
                encoding="utf-8",
            )
            self.assertEqual(
                discover_wandb_identity(out),
                {
                    "run_id": "puhyggoo",
                    "project": "boreft",
                    "entity": "example",
                },
            )


class AttachLoraSftAdapterTest(unittest.TestCase):
    def test_attach_reloads_weights_after_identity(self):
        import torch.nn as nn

        from boreft.baselines.sdpo_ttt import LoRALinear, _inject_lora

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)

        model = _Block()
        _inject_lora(model, 2, 2.0)
        adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
        self.assertTrue(adapters)
        for adapter in adapters:
            adapter.lora_B.data.fill_(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch001.pt"
            save_adapter_checkpoint(
                path,
                model=model,
                lora_rank=2,
                lora_alpha=2.0,
                model_name="dummy",
                use_chat_template=True,
                instruction="prompt",
                prompt_source="search",
                epoch=1,
                label="epoch1",
                shift_score=0.3,
            )
            for adapter in adapters:
                adapter.as_identity()
            self.assertTrue(torch.all(adapters[0].lora_B == 0))
            payload = attach_lora_sft_adapter(model, path)
            self.assertEqual(payload["epoch"], 1)
            self.assertTrue(torch.allclose(adapters[0].lora_B, torch.full_like(adapters[0].lora_B, 0.5)))


class MoloptLoraSftTest(unittest.TestCase):
    def test_from_checkpoint_items_uses_bare_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "items.json").write_text(
                json.dumps(
                    [
                        {"id": 0, "word": "CCO", "target": "CCO [END_SMILES]"},
                        {"id": 1, "word": "c1ccccc1", "target": "c1ccccc1 [END_SMILES]"},
                    ]
                ),
                encoding="utf-8",
            )
            (ckpt / "training_config.json").write_text(
                json.dumps(
                    {
                        "task": "molopt",
                        "mist_smiles_tags": True,
                        "use_chat_template": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            draw = resolve_train_word_draw(
                semantle_csv=None,
                train_top_k=None,
                train_n_samples=None,
                seed=None,
                reft_output_dir=str(ckpt),
                from_checkpoint_items=False,
                skip_items_check=False,
            )
            self.assertEqual(draw.task, "molopt")
            self.assertEqual(draw.source, "items.json")
            self.assertEqual(draw.words, ["CCO", "c1ccccc1"])

    def test_molopt_does_not_fall_back_to_semantle_csv(self):
        with self.assertRaisesRegex(FileNotFoundError, "items.json"):
            resolve_train_word_draw(
                semantle_csv=None,
                train_top_k=None,
                train_n_samples=None,
                seed=None,
                reft_output_dir=None,
                from_checkpoint_items=False,
                skip_items_check=False,
                task="molopt",
            )

    def test_train_instruction_appends_mist_open_tag(self):
        instruction = resolve_sft_instruction(
            prompt_source="train",
            task_description="unused",
            use_chat_template=False,
            task="molopt",
            mist_smiles_tags=True,
        )
        self.assertTrue(instruction.endswith("[START_SMILES]"))
        self.assertIn("valid SMILES string", instruction)
        self.assertEqual(
            user_instruction(
                prompt_source="train",
                task_description="unused",
                use_chat_template=False,
                task="molopt",
            ),
            task_instruction("molopt", use_chat_template=False),
        )

    def test_sft_gold_is_mist_completion_target(self):
        instruction = resolve_sft_instruction(
            prompt_source="train",
            task_description="unused",
            use_chat_template=False,
            task="molopt",
            mist_smiles_tags=True,
        )
        items = prepare_sft_items(
            ["CCO [END_SMILES]"],
            instruction,
            use_chat_template=False,
            system_prompt=None,
            task="molopt",
            mist_smiles_tags=True,
        )
        self.assertEqual(items[0].prompt, instruction)
        self.assertEqual(items[0].target, "CCO [END_SMILES]")
        self.assertEqual(items[0]._raw_word, "CCO")

    def test_proposal_eval_canonicalizes_molopt_hits(self):
        stats = proposal_eval_stats(
            ["OCC [END_SMILES]", "not a molecule"],
            ["CCO", "c1ccccc1"],
            task="molopt",
        )
        self.assertAlmostEqual(stats["train_hit_rate"], 0.5)
        self.assertAlmostEqual(stats["valid_rate"], 0.5)
        self.assertEqual(stats["n_unique_in_train"], 1)

    def test_dry_run_writes_checkpoint_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "items.json").write_text(
                json.dumps([{"id": 0, "word": "CCO"}, {"id": 1, "word": "c1ccccc1"}]),
                encoding="utf-8",
            )
            (ckpt / "training_config.json").write_text(
                json.dumps({"task": "molopt", "mist_smiles_tags": True}) + "\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "out"
            train(
                parse_args(
                    [
                        "--dry-run",
                        "--no-wandb",
                        "--output-dir",
                        str(out),
                        "--reft-output-dir",
                        str(ckpt),
                    ]
                )
            )
            payload = json.loads((out / "words.json").read_text(encoding="utf-8"))
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["words"], ["CCO", "c1ccccc1"])
            self.assertEqual(config["task"], "molopt")
            self.assertEqual(config["prompt_source"], "train")
            self.assertTrue(config["mist_smiles_tags"])
            self.assertTrue(config["instruction"].endswith("[START_SMILES]"))


if __name__ == "__main__":
    unittest.main()
