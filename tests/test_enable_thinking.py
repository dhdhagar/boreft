"""Guards and helpers for chat-template ``enable_thinking``."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from boreft.interactive import (
    build_checkpoint_prompt,
    end_relative_intervention_position,
    prepend_system_message,
    strip_thinking_text,
    target_intervention_flags,
)


class TestEndRelativePosition(unittest.TestCase):
    def test_l1_and_f1_l1_are_end_relative(self):
        self.assertTrue(end_relative_intervention_position("l1"))
        self.assertTrue(end_relative_intervention_position("f1+l1"))

    def test_f1_marker_content_are_not(self):
        self.assertFalse(end_relative_intervention_position("f1"))
        self.assertFalse(end_relative_intervention_position("marker"))
        self.assertFalse(end_relative_intervention_position("content_f1"))
        self.assertFalse(end_relative_intervention_position("content_l1"))


class TestStripThinkingText(unittest.TestCase):
    def test_strips_think_block(self):
        text = "<think>\nreason\n</think>\n\napple"
        self.assertEqual(strip_thinking_text(text), "apple")

    def test_noop_without_block(self):
        self.assertEqual(strip_thinking_text("apple"), "apple")


class TestTargetFlagsWithThinking(unittest.TestCase):
    def test_matches_answer_after_think_block(self):
        seen, is_target = target_intervention_flags(
            "<think>\nx\n</think>\n\napple",
            "apple",
            {"apple": 0},
        )
        self.assertTrue(seen)
        self.assertTrue(is_target)


class TestBuildCheckpointPromptThinkingGuard(unittest.TestCase):
    def _ckpt(self, position: str):
        return SimpleNamespace(
            from_chat_template=True,
            prompt="frozen",
            content_span=None,
            saved_cfg={"position": position},
            tokenizer=MagicMock(),
            intervention_token_id=None,
        )

    def test_rejects_thinking_with_l1_intervention(self):
        with self.assertRaisesRegex(ValueError, "enable_thinking is incompatible"):
            build_checkpoint_prompt(
                self._ckpt("l1"),
                user_text="hi",
                history=[],
                accumulate_history=False,
                use_checkpoint_prompt=False,
                generation_mode="intervention",
                model_name="Qwen/Qwen3-0.6B",
                enable_thinking=True,
            )

    def test_allows_thinking_in_base_mode(self):
        ckpt = self._ckpt("l1")
        ckpt.tokenizer.chat_template = "{{ messages }}"
        ckpt.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        prompt, from_chat, _ = build_checkpoint_prompt(
            ckpt,
            user_text="hi",
            history=[],
            accumulate_history=False,
            use_checkpoint_prompt=False,
            generation_mode="base",
            model_name="Qwen/Qwen3-0.6B",
            enable_thinking=True,
        )
        self.assertEqual(prompt, "prompt")
        self.assertTrue(from_chat)
        ckpt.tokenizer.apply_chat_template.assert_called_once()
        kwargs = ckpt.tokenizer.apply_chat_template.call_args.kwargs
        self.assertTrue(kwargs.get("enable_thinking"))

    def test_prepends_system_message_in_chat_mode(self):
        ckpt = self._ckpt("l1")
        ckpt.tokenizer.chat_template = "{{ messages }}"
        ckpt.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        build_checkpoint_prompt(
            ckpt,
            user_text="hi",
            history=[],
            accumulate_history=False,
            use_checkpoint_prompt=False,
            generation_mode="base",
            model_name="Qwen/Qwen3-0.6B",
            system_prompt="Be brief.",
        )
        conversation = ckpt.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            conversation,
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_checkpoint_prompt_flag_skips_system_message(self):
        ckpt = self._ckpt("l1")
        prompt, from_chat, span = build_checkpoint_prompt(
            ckpt,
            user_text="hi",
            history=[],
            accumulate_history=False,
            use_checkpoint_prompt=True,
            generation_mode="base",
            model_name="Qwen/Qwen3-0.6B",
            system_prompt="Be brief.",
        )
        self.assertEqual(prompt, "frozen")
        self.assertTrue(from_chat)
        self.assertIsNone(span)
        ckpt.tokenizer.apply_chat_template.assert_not_called()


class TestPrependSystemMessage(unittest.TestCase):
    def test_prepends_when_set(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            prepend_system_message(messages, "Be brief."),
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_noop_when_blank(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(prepend_system_message(messages, None), messages)
        self.assertIs(prepend_system_message(messages, "  "), messages)


if __name__ == "__main__":
    unittest.main()
