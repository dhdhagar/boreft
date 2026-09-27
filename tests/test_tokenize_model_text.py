import unittest
from unittest.mock import MagicMock

import torch

from boreft.data_utils import (
    FIXED_CHAT_DATE,
    chat_prompt,
    load_checkpoint_tokenizer,
    prompt_tokenization_from_cfg,
    system_prompt_from_cfg,
    tokenize_model_text,
)


class _FakeTokenizer:
    """Minimal tokenizer: prepends bos_token_id when add_special_tokens=True."""

    bos_token_id = 128000

    def __init__(self, body_ids: list[int]):
        self._body_ids = body_ids

    def __call__(self, text: str, *, add_special_tokens: bool = True, **kwargs):
        ids = list(self._body_ids)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        out = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if kwargs.get("return_tensors") != "pt":
            out["input_ids"] = out["input_ids"][0].tolist()
        return out


class TestPromptTokenizationFromCfg(unittest.TestCase):
    def test_false_when_missing(self):
        self.assertFalse(prompt_tokenization_from_cfg(None))
        self.assertFalse(prompt_tokenization_from_cfg({}))

    def test_true_when_set(self):
        self.assertTrue(prompt_tokenization_from_cfg({"use_chat_template": True}))


class TestTokenizeModelText(unittest.TestCase):
    def setUp(self):
        # Chat-template body already includes BOS at index 0 (Llama-style).
        self.tokenizer = _FakeTokenizer([128000, 100, 200, 300])
        self.chat_text = "<|begin_of_text|>user\nGenerate a word\nassistant\n"

    def test_plain_prompt_adds_bos(self):
        enc = tokenize_model_text(
            self.tokenizer,
            "Generate a single English word.",
            from_chat_template=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0].tolist()
        self.assertEqual(ids[0], 128000)
        self.assertEqual(len(ids), len(self.tokenizer._body_ids) + 1)

    def test_chat_template_skips_extra_bos(self):
        enc = tokenize_model_text(
            self.tokenizer,
            self.chat_text,
            from_chat_template=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0].tolist()
        self.assertEqual(ids.count(128000), 1)
        self.assertEqual(ids, self.tokenizer._body_ids)

    def test_duplicate_bos_regression(self):
        """Old path (always add_special_tokens) double-prepended BOS on chat strings."""
        old = self.tokenizer(self.chat_text, add_special_tokens=True, return_tensors="pt")
        fixed = tokenize_model_text(
            self.tokenizer,
            self.chat_text,
            from_chat_template=True,
            return_tensors="pt",
        )
        self.assertEqual(old["input_ids"][0].tolist().count(128000), 2)
        self.assertEqual(fixed["input_ids"][0].tolist().count(128000), 1)
        self.assertLess(
            len(fixed["input_ids"][0]),
            len(old["input_ids"][0]),
        )

    def test_forwards_kwargs(self):
        mock = MagicMock(return_value={"input_ids": torch.tensor([[1]])})
        tokenize_model_text(mock, "hello", from_chat_template=True, return_tensors="pt")
        mock.assert_called_once_with("hello", add_special_tokens=False, return_tensors="pt")


class TestFixedChatDate(unittest.TestCase):
    def test_chat_prompt_passes_fixed_date(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = "rendered"
        out = chat_prompt(tokenizer, "hello")
        self.assertEqual(out, "rendered")
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            tokenize=False,
            date_string=FIXED_CHAT_DATE,
            enable_thinking=False,
        )

    def test_chat_prompt_can_enable_thinking(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = "rendered"
        chat_prompt(tokenizer, "hello", enable_thinking=True)
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            tokenize=False,
            date_string=FIXED_CHAT_DATE,
            enable_thinking=True,
        )

    def test_chat_prompt_prepends_system_message(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = "rendered"
        chat_prompt(tokenizer, "hello", system_prompt="Be brief.")
        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ],
            add_generation_prompt=True,
            tokenize=False,
            date_string=FIXED_CHAT_DATE,
            enable_thinking=False,
        )

    def test_chat_prompt_omits_blank_system_message(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = "rendered"
        chat_prompt(tokenizer, "hello", system_prompt="  ")
        conversation = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(conversation, [{"role": "user", "content": "hello"}])

    def test_apply_chat_format_with_system_keeps_prefix(self):
        from boreft.data import ReftItem
        from boreft.data_utils import apply_chat_format

        class _RoleTok:
            chat_template = "yes"
            eos_token = "</s>"

            def apply_chat_template(
                self, conversation, tokenize=False, add_generation_prompt=False, **_kwargs
            ):
                parts = [f"<{m['role']}>{m['content']}" for m in conversation]
                if add_generation_prompt:
                    parts.append("<assistant>")
                return "".join(parts)

        items = [ReftItem(id=0, prompt="old", target="CCO")]
        suffix = apply_chat_format(
            items, _RoleTok(), "Generate SMILES.", system_prompt="Be brief."
        )
        self.assertEqual(
            items[0].prompt,
            "<system>Be brief.<user>Generate SMILES.<assistant>",
        )
        self.assertEqual(items[0].target, "CCO</s>")
        self.assertEqual(suffix, "")


class TestSystemPromptFromCfg(unittest.TestCase):
    def test_missing_or_blank_is_none(self):
        self.assertIsNone(system_prompt_from_cfg(None))
        self.assertIsNone(system_prompt_from_cfg({}))
        self.assertIsNone(system_prompt_from_cfg({"system_prompt": "  "}))

    def test_strips_recorded_text(self):
        self.assertEqual(
            system_prompt_from_cfg({"system_prompt": "  Be brief.  "}),
            "Be brief.",
        )


class TestLoadCheckpointTokenizer(unittest.TestCase):
    def test_prefers_fast_only_when_tokenizer_json_exists(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_from:
                mock_from.return_value = "tok"
                out = load_checkpoint_tokenizer(str(root), use_fast=True)
                self.assertEqual(out, "tok")
                mock_from.assert_called_once_with(str(root), use_fast=False)

            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_from:
                mock_from.return_value = "tok-fast"
                out = load_checkpoint_tokenizer(str(root), use_fast=False)
                self.assertEqual(out, "tok-fast")
                mock_from.assert_called_once_with(str(root), use_fast=True)


if __name__ == "__main__":
    unittest.main()
