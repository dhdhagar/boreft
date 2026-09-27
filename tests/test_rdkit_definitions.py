"""Tests for opt-in RDKit scalar suffixes on molopt definitions."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from boreft.text_similarity import (
    append_rdkit_definition_values,
    definition_lookup_for_cfg,
    embedding_provenance,
    load_definition_embed_lookup,
    load_rdkit_definition_lookup,
    stringify_rdkit_definition,
    training_cache_params,
)
from boreft.train_args import TrainConfig
from boreft.train import _build_training_config_dict
from boreft.learn_bias import _build_batch_sdpo_kwargs
from boreft.chem import RDKIT_DESCRIPTOR_SCHEMA_VERSION


VALUES = [46.069, -0.0014, 20.23, 1, 1, 0, 0, 1.0, 0, 0]
PROPERTIES = (
    "average molecular weight 46.069; "
    "Wildman-Crippen octanol-water partition coefficient estimate -0.0014; "
    "topological polar surface area 20.23; "
    "number of hydrogen-bond donor groups 1; "
    "number of hydrogen-bond acceptor groups 1; "
    "number of rotatable bonds using RDKit's default definition 0; "
    "net formal charge of the represented molecular graph 0; "
    "fraction of carbon atoms that are sp3-hybridized 1.0; "
    "number of aromatic rings 0; "
    "number of tetrahedral atom stereocenters, specified or unspecified 0"
)


class TempDefinitionsMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.definitions_path = os.path.join(self.tmp.name, "definitions.jsonl")
        self.rdkit_path = os.path.join(
            self.tmp.name, "definitions_rdkit.jsonl"
        )
        self.map_path = os.path.join(
            self.tmp.name, "definitions_rdkit_map.json"
        )
        with open(self.definitions_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"target": "CCO", "definition": "ethanol"}) + "\n")
        with open(self.rdkit_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"target": "CCO", "definition": VALUES}) + "\n")
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": RDKIT_DESCRIPTOR_SCHEMA_VERSION}, f)


class DefinitionSuffixTests(TempDefinitionsMixin, unittest.TestCase):
    def test_stringifies_in_stored_order(self):
        self.assertEqual(
            stringify_rdkit_definition(VALUES),
            PROPERTIES,
        )

    def test_appends_exact_semicolon_format(self):
        lookup = load_rdkit_definition_lookup(self.rdkit_path)
        self.assertEqual(
            append_rdkit_definition_values(
                "CCO", "ethanol", rdkit_lookup=lookup, require_lookup=True
            ),
            f"ethanol 2D properties: {PROPERTIES}",
        )

    def test_missing_ad_hoc_target_is_computed_from_smiles(self):
        value = append_rdkit_definition_values(
            "CCN", "ethylamine", rdkit_lookup={}
        )
        self.assertTrue(value.startswith("ethylamine 2D properties: "))
        self.assertEqual(len(value.split("; ")), 10)

    def test_can_omit_natural_language_definition(self):
        lookup = load_rdkit_definition_lookup(self.rdkit_path)
        self.assertEqual(
            append_rdkit_definition_values(
                "CCO",
                "ethanol",
                rdkit_lookup=lookup,
                require_lookup=True,
                omit_base_definition=True,
            ),
            f"2D properties: {PROPERTIES}",
        )

    def test_strict_sidecar_lookup_rejects_missing_target(self):
        with self.assertRaisesRegex(ValueError, "missing from RDKit"):
            append_rdkit_definition_values(
                "CCN", "ethylamine", rdkit_lookup={}, require_lookup=True
            )

    def test_decorated_loader_appends_before_task_template(self):
        lookup = load_definition_embed_lookup(
            self.definitions_path,
            task="molopt",
            append_rdkit_definitions=True,
            rdkit_definitions_path=self.rdkit_path,
        )
        self.assertIn(
            "ethanol 2D properties: average molecular weight 46.069",
            lookup["CCO"],
        )


class ConfigAndProvenanceTests(TempDefinitionsMixin, unittest.TestCase):
    def test_flag_is_accepted_for_definition_embeds(self):
        cfg = TrainConfig(
            task="molopt",
            molopt_csv="molecules.csv",
            use_definition_embeds=True,
            append_rdkit_definitions=True,
        )
        self.assertTrue(cfg.append_rdkit_definitions)

    def test_flag_is_rejected_without_definition_consumer(self):
        with self.assertRaisesRegex(ValueError, "requires --use-definition-embeds"):
            TrainConfig(
                task="molopt",
                molopt_csv="molecules.csv",
                append_rdkit_definitions=True,
            )

    def test_smiles_tags_accepted_for_molopt(self):
        cfg = TrainConfig(
            task="molopt",
            molopt_csv="molecules.csv",
            smiles_tags=True,
        )
        self.assertTrue(cfg.smiles_tags)
        saved = _build_training_config_dict(
            cfg,
            batch_size=1,
            dtype_name="float32",
            num_anchors=1,
            num_training_examples=1,
        )
        self.assertTrue(saved["smiles_tags"])

    def test_mist_smiles_tags_accepted_for_molopt(self):
        cfg = TrainConfig(
            task="molopt",
            molopt_csv="molecules.csv",
            mist_smiles_tags=True,
        )
        self.assertTrue(cfg.mist_smiles_tags)
        saved = _build_training_config_dict(
            cfg,
            batch_size=1,
            dtype_name="float32",
            num_anchors=1,
            num_training_examples=1,
        )
        self.assertTrue(saved["mist_smiles_tags"])
        self.assertFalse(saved["smiles_tags"])

    def test_mist_smiles_tags_rejected_for_semantle(self):
        with self.assertRaisesRegex(ValueError, "only supported for task=molopt"):
            TrainConfig(
                task="semantle",
                semantle_csv=("words.csv",),
                mist_smiles_tags=True,
            )

    def test_mist_and_llasmol_tags_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "only one of"):
            TrainConfig(
                task="molopt",
                molopt_csv="molecules.csv",
                smiles_tags=True,
                mist_smiles_tags=True,
            )

    def test_smiles_tags_rejected_for_semantle(self):
        with self.assertRaisesRegex(ValueError, "only supported for task=molopt"):
            TrainConfig(
                task="semantle",
                semantle_csv=("words.csv",),
                smiles_tags=True,
            )

    def test_learn_bias_replays_smiles_tags_on_gold_targets(self):
        from boreft.learn_bias import _build_target_texts

        ckpt = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token="</s>"),
            from_chat_template=False,
            prompt="Here is a SMILES string:",
            saved_cfg={"task": "molopt", "smiles_tags": True},
        )
        _prompt, full = _build_target_texts(ckpt, "CCO")
        self.assertEqual(
            full, "Here is a SMILES string: <SMILES>CCO</SMILES></s>"
        )
        ckpt.saved_cfg["smiles_tags"] = False
        _prompt, full = _build_target_texts(ckpt, "CCO")
        self.assertEqual(full, "Here is a SMILES string: CCO</s>")

    def test_learn_bias_replays_mist_smiles_tags_on_gold_targets(self):
        from boreft.learn_bias import _build_target_texts

        ckpt = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token="</s>"),
            from_chat_template=False,
            prompt="Here is a SMILES string: [START_SMILES]",
            saved_cfg={"task": "molopt", "mist_smiles_tags": True},
        )
        _prompt, full = _build_target_texts(ckpt, "CCO")
        self.assertEqual(
            full, "Here is a SMILES string: [START_SMILES] CCO [END_SMILES]</s>"
        )

    def test_learn_bias_chat_uses_saved_instruction_and_system(self):
        from unittest.mock import patch

        from boreft.learn_bias import _build_target_texts

        ckpt = SimpleNamespace(
            tokenizer=object(),
            from_chat_template=True,
            prompt="<system>Be brief.<user>Custom instr.<assistant>",
            saved_cfg={
                "task": "molopt",
                "smiles_tags": False,
                "chat_instruction": "Custom instr.",
                "system_prompt": "Be brief.",
            },
            intervention_token_id=None,
        )
        with patch(
            "boreft.learn_bias.chat_assistant_target",
            return_value="CCO</s>",
        ) as mock_target:
            prompt, full = _build_target_texts(ckpt, "CCO")
        self.assertEqual(prompt, ckpt.prompt)
        self.assertEqual(full, ckpt.prompt + "CCO</s>")
        args, kwargs = mock_target.call_args
        self.assertEqual(args[1], "Custom instr.")
        self.assertEqual(kwargs["system_prompt"], "Be brief.")

    def test_omit_flag_requires_append_flag(self):
        with self.assertRaisesRegex(
            ValueError, "requires --append-rdkit-definitions"
        ):
            TrainConfig(
                task="molopt",
                molopt_csv="molecules.csv",
                use_definition_embeds=True,
                omit_molt5_definitions=True,
            )

    def test_flag_is_rejected_for_semantle(self):
        with self.assertRaisesRegex(ValueError, "only supported for task=molopt"):
            TrainConfig(
                task="semantle",
                semantle_csv=("words.csv",),
                use_definition_embeds=True,
                append_rdkit_definitions=True,
            )

    def test_training_cache_round_trip_replays_suffix(self):
        provenance, lookup, _ = training_cache_params(
            use_definition_embeds=True,
            task="molopt",
            words=["CCO"],
            definitions_path=self.definitions_path,
            append_rdkit_definitions=True,
            rdkit_definitions_path=self.rdkit_path,
            rdkit_definitions_map_path=self.map_path,
        )
        self.assertTrue(provenance["append_rdkit_definitions"])
        self.assertEqual(
            provenance["rdkit_definitions_path"],
            os.path.abspath(self.rdkit_path),
        )
        self.assertIn(
            "ethanol 2D properties: average molecular weight 46.069",
            lookup["CCO"],
        )

        cfg = dict(provenance)
        rebuilt = definition_lookup_for_cfg(cfg)
        self.assertEqual(rebuilt, lookup)

    def test_training_cache_can_use_only_rdkit_properties(self):
        provenance, lookup, _ = training_cache_params(
            use_definition_embeds=True,
            task="molopt",
            words=["CCO"],
            definitions_path=self.definitions_path,
            append_rdkit_definitions=True,
            omit_molt5_definitions=True,
            rdkit_definitions_path=self.rdkit_path,
            rdkit_definitions_map_path=self.map_path,
        )
        self.assertTrue(provenance["omit_molt5_definitions"])
        self.assertIn("2D properties: average molecular weight", lookup["CCO"])
        self.assertNotIn("ethanol", lookup["CCO"])

    def test_sidecar_content_change_invalidates_lookup_and_provenance(self):
        first = load_rdkit_definition_lookup(self.rdkit_path)
        first_provenance = embedding_provenance(
            task="molopt",
            use_definition_embeds=True,
            definitions_path=self.definitions_path,
            append_rdkit_definitions=True,
            rdkit_definitions_path=self.rdkit_path,
            rdkit_definitions_map_path=self.map_path,
        )
        changed_values = [99.0, *VALUES[1:]]
        with open(self.rdkit_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"target": "CCO", "definition": changed_values}
                )
                + "\n"
            )
        second = load_rdkit_definition_lookup(self.rdkit_path)
        second_provenance = embedding_provenance(
            task="molopt",
            use_definition_embeds=True,
            definitions_path=self.definitions_path,
            append_rdkit_definitions=True,
            rdkit_definitions_path=self.rdkit_path,
            rdkit_definitions_map_path=self.map_path,
        )
        self.assertEqual(first["CCO"][0], 46.069)
        self.assertEqual(second["CCO"][0], 99.0)
        self.assertNotEqual(
            first_provenance["rdkit_definitions_sha256"],
            second_provenance["rdkit_definitions_sha256"],
        )

    def test_eval_provenance_stays_definition_free(self):
        provenance = embedding_provenance(task="molopt")
        self.assertNotIn("append_rdkit_definitions", provenance)

    def test_batched_sdpo_teacher_replays_suffix(self):
        class RecordingTokenizer:
            eos_token = "<eos>"

            def __init__(self):
                self.texts = []

            def __call__(self, text, **_kwargs):
                self.texts.append(text)
                return {"input_ids": list(range(len(text)))}

        tokenizer = RecordingTokenizer()
        ckpt = SimpleNamespace(
            tokenizer=tokenizer,
            prompt="Generate:",
            from_chat_template=False,
            saved_cfg={
                "task": "molopt",
                "append_rdkit_definitions": True,
                "rdkit_definitions_path": self.rdkit_path,
            },
            reft_model=SimpleNamespace(model=object()),
        )
        _build_batch_sdpo_kwargs(
            ckpt,
            ["CCO"],
            {"CCO": "ethanol"},
            object(),
            lambda_sdpo=1.0,
        )
        self.assertTrue(
            any(
                "ethanol 2D properties: average molecular weight 46.069"
                in text
                for text in tokenizer.texts
            )
        )

    def test_custom_metric_map_is_persisted_without_suffix_flag(self):
        cfg = TrainConfig(
            task="molopt",
            molopt_csv="molecules.csv",
            rdkit_definitions_map_path=self.map_path,
        )
        saved = _build_training_config_dict(
            cfg,
            batch_size=1,
            dtype_name="float32",
            num_anchors=1,
            num_training_examples=1,
        )
        self.assertEqual(
            saved["rdkit_definitions_map_path"],
            os.path.abspath(self.map_path),
        )


if __name__ == "__main__":
    unittest.main()
