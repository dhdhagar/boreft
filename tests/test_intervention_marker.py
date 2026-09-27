import unittest

import torch

from boreft.intervention_marker import (
    apply_injection,
    content_position_indices,
    intervention_locations_for_prompt,
    intervention_position_list,
    is_content_position,
    is_marker_position,
    marker_indices,
    prepare_instruction,
    resolve_instruction_for_prompt,
    validate_intervention_position,
    POSITION_CONTENT_F1,
    POSITION_CONTENT_L1,
    POSITION_MARKER,
)


class TestInjection(unittest.TestCase):
    def test_prefix(self):
        out = apply_injection("Hello world", "prefix", "<M>")
        self.assertEqual(out, "<M> Hello world")

    def test_suffix(self):
        out = apply_injection("Hello world", "suffix", "<M>")
        self.assertEqual(out, "Hello world <M>")

    def test_none(self):
        self.assertEqual(prepare_instruction("  hi  ", "none", None), "hi")


class TestResolveInstructionForPrompt(unittest.TestCase):
    def test_applies_prefix_to_raw(self):
        out = resolve_instruction_for_prompt(
            "Hello", "prefix", "<M>", tokenizer=None, intervention_token_id=None
        )
        self.assertEqual(out, "<M> Hello")

    def test_skips_when_marker_already_present(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                if text == "<M>":
                    return [99]
                if "Hello" in text:
                    return [99, 1, 2]
                return [1, 2]

        out = resolve_instruction_for_prompt(
            "<M> Hello",
            "prefix",
            "<M>",
            tokenizer=Tok(),
            intervention_token_id=99,
        )
        self.assertEqual(out, "<M> Hello")

    def test_raises_on_double_legacy_marker(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [99, 99, 1]

        with self.assertRaises(ValueError):
            resolve_instruction_for_prompt(
                "<M> <M> Hello",
                "prefix",
                "<M>",
                tokenizer=Tok(),
                intervention_token_id=99,
            )


class TestMarkerPosition(unittest.TestCase):
    def test_is_marker(self):
        self.assertTrue(is_marker_position("marker"))
        self.assertTrue(is_marker_position("Marker"))
        self.assertFalse(is_marker_position("l1"))

    def test_marker_indices(self):
        self.assertEqual(marker_indices([1, 99, 2, 3], 99), [1])
        with self.assertRaises(ValueError):
            marker_indices([1, 2, 3], 99)
        with self.assertRaises(ValueError):
            marker_indices([1, 99, 2, 99, 3], 99)

    def test_marker_locations(self):
        ids = [10, 42, 11, 12]
        locs = intervention_locations_for_prompt(
            POSITION_MARKER,
            4,
            ids,
            intervention_token_id=42,
        )
        self.assertEqual(locs, [[1]])

    def test_legacy_l1(self):
        ids = list(range(5))
        locs = intervention_position_list("l1", ids)
        self.assertEqual(locs, [4])


class TestContentPosition(unittest.TestCase):
    def test_is_content(self):
        self.assertTrue(is_content_position("content_f1"))
        self.assertTrue(is_content_position("content_l1"))
        self.assertFalse(is_content_position("l1"))

    def test_content_indices(self):
        self.assertEqual(content_position_indices(POSITION_CONTENT_F1, 10, 15), [10])
        self.assertEqual(content_position_indices(POSITION_CONTENT_L1, 10, 15), [14])

    def test_content_locations(self):
        ids = list(range(20))
        locs = intervention_position_list(
            POSITION_CONTENT_F1,
            ids,
            content_span=(10, 15),
        )
        self.assertEqual(locs, [10])
        locs = intervention_position_list(
            POSITION_CONTENT_L1,
            ids,
            content_span=(10, 15),
        )
        self.assertEqual(locs, [14])

    def test_content_requires_span(self):
        with self.assertRaises(ValueError):
            intervention_position_list(POSITION_CONTENT_F1, list(range(5)))

    def test_validate_position(self):
        self.assertEqual(validate_intervention_position("marker"), "marker")
        with self.assertRaises(ValueError):
            validate_intervention_position("not_a_position")

    def test_instruction_content_span_probe_lookup(self):
        from unittest.mock import patch

        from boreft.intervention_marker import instruction_content_span

        probe_id = 128002
        base = (
            [0] * 30
            + [32215]
            + list(range(100, 115))
            + [128009, 200, 201, 202, 203]
        )
        marked_prefix = base[:30] + [probe_id] + base[30:]
        marked_suffix = base[:47] + [probe_id, 999] + base[47:]

        class Tok:
            def encode(self, text, add_special_tokens=False):
                if text == "<probe>":
                    return [probe_id]
                return [1]

        tok = Tok()
        with patch(
            "boreft.intervention_marker._chat_prompt_ids",
            side_effect=[base, marked_prefix, marked_suffix],
        ):
            start, end = instruction_content_span(
                tok,
                "ignored",
                use_chat_template=True,
                span_probe_token="<probe>",
            )
        self.assertEqual(start, 30)
        self.assertEqual(end, 46)


class _FakeEmbed(torch.nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, dim))

    def forward(self, input_ids):
        return self.weight[input_ids]


class TestInitEmbedding(unittest.TestCase):
    def test_newline_init(self):
        from boreft.intervention_marker import init_intervention_token_embedding

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [5, 6] if text == "\n" else [1]

        embed = _FakeEmbed(10, 4)
        model = type("M", (), {"get_input_embeddings": lambda self: embed})()
        init_intervention_token_embedding(model, Tok(), 3, "newline")
        expected = embed.weight[5:7].mean(dim=0)
        self.assertTrue(torch.allclose(embed.weight[3], expected))


class TestEnsureInterventionToken(unittest.TestCase):
    def test_adds_special_token_within_embedding_table(self):
        from boreft.intervention_marker import ensure_intervention_token

        class Tok:
            def __init__(self):
                self._ids = {"a": 0, "b": 1}
                self._next = 2

            def add_tokens(self, tokens):
                from tokenizers import AddedToken

                added = 0
                for t in tokens:
                    s = t.content if isinstance(t, AddedToken) else str(t)
                    if s not in self._ids:
                        self._ids[s] = self._next
                        self._next += 1
                        added += 1
                return added

            def convert_tokens_to_ids(self, token):
                return self._ids[token]

            def encode(self, text, add_special_tokens=False):
                return [self._ids[text]]

        embed = _FakeEmbed(8, 4)  # spare rows beyond current vocab size 2
        model = type("M", (), {"get_input_embeddings": lambda self: embed})()
        tok = Tok()
        token_id = ensure_intervention_token(tok, model, "<|boreft_0|>")
        self.assertEqual(token_id, 2)
        self.assertEqual(ensure_intervention_token(tok, model, "<|boreft_0|>"), 2)

    def test_rejects_id_outside_embedding_table(self):
        from boreft.intervention_marker import ensure_intervention_token

        class Tok:
            def add_tokens(self, tokens):
                return 1

            def convert_tokens_to_ids(self, token):
                return 10

            def encode(self, text, add_special_tokens=False):
                return [10]

        embed = _FakeEmbed(4, 2)
        model = type("M", (), {"get_input_embeddings": lambda self: embed})()
        with self.assertRaisesRegex(ValueError, "outside the embedding table"):
            ensure_intervention_token(Tok(), model, "<|boreft_0|>")


class TestContentSpanFromCfg(unittest.TestCase):
    def test_matches_eval_prompt_instruction_including_inject(self):
        from unittest.mock import patch

        from boreft.intervention_marker import content_span_from_cfg

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1]

        saved = {
            "position": "content_f1",
            "task": "molopt",
            "use_chat_template": True,
            "chat_instruction": "Generate SMILES.",
            "intervention_inject": "prefix",
            "intervention_token": "<M>",
            "system_prompt": "  Be brief.  ",
            "span_probe_token": "<probe>",
        }
        with patch(
            "boreft.intervention_marker.instruction_content_span",
            return_value=(1, 2),
        ) as span:
            self.assertEqual(content_span_from_cfg(Tok(), saved, "model"), (1, 2))
        args, kwargs = span.call_args
        self.assertEqual(args[1], "<M> Generate SMILES.")
        self.assertEqual(kwargs["system_prompt"], "Be brief.")

    def test_appends_mist_open_tag_after_prefix_inject(self):
        from unittest.mock import patch

        from boreft.intervention_marker import content_span_from_cfg

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1]

        saved = {
            "position": "content_f1",
            "task": "molopt",
            "use_chat_template": False,
            "mist_smiles_tags": True,
            "intervention_inject": "prefix",
            "intervention_token": "<M>",
            "span_probe_token": "<probe>",
        }
        with patch(
            "boreft.intervention_marker.instruction_content_span",
            return_value=(1, 2),
        ) as span:
            content_span_from_cfg(Tok(), saved, "model")
        instruction = span.call_args[0][1]
        self.assertTrue(instruction.startswith("<M> "))
        self.assertTrue(instruction.endswith("[START_SMILES]"))


class TestBuildEvalPrompt(unittest.TestCase):
    def test_completion_prompt_appends_mist_open_tag(self):
        from boreft.intervention_marker import build_eval_prompt

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1]

        prompt = build_eval_prompt(
            Tok(),
            "molopt",
            use_chat_template=False,
            mist_smiles_tags=True,
        )
        self.assertTrue(prompt.endswith("[START_SMILES]"))
        self.assertNotIn("[END_SMILES]", prompt)

    def test_mist_open_tag_comes_after_suffix_marker(self):
        from boreft.intervention_marker import build_eval_prompt

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [99] if text == "<M>" else [1]

        prompt = build_eval_prompt(
            Tok(),
            "molopt",
            use_chat_template=False,
            mist_smiles_tags=True,
            intervention_inject="suffix",
            intervention_token="<M>",
        )
        self.assertTrue(prompt.endswith("[START_SMILES]"))
        self.assertLess(prompt.rfind("<M>"), prompt.rfind("[START_SMILES]"))


if __name__ == "__main__":
    unittest.main()
