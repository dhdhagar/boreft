"""Integration test: instruction content span on Llama 3.2 Instruct."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "llama32_instruct_semantle_spans.json"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _llama_tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


@unittest.skipUnless(
    _llama_tokenizer_available(),
    f"{MODEL_NAME} tokenizer not cached locally",
)
class TestInstructionContentSpanLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.fixture = _load_fixture()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def _assert_span(self, *, use_chat_template: bool, key: str) -> None:
        from boreft.data_utils import chat_prompt, tokenize_model_text
        from boreft.intervention_marker import (
            POSITION_CONTENT_F1,
            POSITION_CONTENT_L1,
            content_position_indices,
            instruction_content_span,
        )
        from boreft.task_config import task_instruction

        expected = self.fixture[key]
        instruction = task_instruction("semantle", use_chat_template=use_chat_template)
        span = instruction_content_span(
            self.tokenizer,
            instruction,
            use_chat_template=use_chat_template,
            span_probe_token=self.fixture["probe_token"],
        )
        self.assertEqual(list(span), expected["content_span"])

        if use_chat_template:
            prompt = chat_prompt(self.tokenizer, instruction)
            ids = tokenize_model_text(
                self.tokenizer, prompt, from_chat_template=True
            )["input_ids"]
        else:
            ids = tokenize_model_text(
                self.tokenizer, instruction, from_chat_template=False
            )["input_ids"]

        f1_idx = content_position_indices(POSITION_CONTENT_F1, *span)[0]
        l1_idx = content_position_indices(POSITION_CONTENT_L1, *span)[0]
        self.assertEqual(
            self.tokenizer.decode([ids[f1_idx]]).strip(),
            expected["content_f1_token"],
        )
        self.assertEqual(
            self.tokenizer.decode([ids[l1_idx]]).strip(),
            expected["content_l1_token"],
        )

    def test_chat_semantle_default_instruction(self):
        self._assert_span(use_chat_template=True, key="chat")

    def test_plain_semantle_default_instruction(self):
        self._assert_span(use_chat_template=False, key="plain")


class TestValidateInterventionPosition(unittest.TestCase):
    def test_valid_special(self):
        from boreft.intervention_marker import validate_intervention_position

        self.assertEqual(validate_intervention_position("marker"), "marker")
        self.assertEqual(validate_intervention_position("content_f1"), "content_f1")
        self.assertEqual(validate_intervention_position("CONTENT_L1"), "content_l1")

    def test_valid_legacy(self):
        from boreft.intervention_marker import validate_intervention_position

        self.assertEqual(validate_intervention_position("l1"), "l1")
        self.assertEqual(validate_intervention_position("F2+l3"), "f2+l3")

    def test_invalid(self):
        from boreft.intervention_marker import validate_intervention_position

        for bad in ("content", "content_f1+f2", "marker+l1", "x1", "f", "l"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_intervention_position(bad)


if __name__ == "__main__":
    unittest.main()
