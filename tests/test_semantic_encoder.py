"""Tests for LLM-encoder pooling and instruction masks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from boreft.pyreft.semantic_encoder import (
    POOLING_INSTRUCTION_MEAN,
    POOLING_LAST_INSTRUCTION,
    _pool_hidden_states,
    build_semantic_encoder,
    encode_definition_inputs,
    instruction_mask_from_offset_mapping,
    instruction_masks_from_batched_encoding,
    normalize_encoder_pooling,
)
from boreft.task_config import (
    definition_embedding_text,
    definition_instruction_char_span,
    definition_instruction_char_start,
)


class _Rotary(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, position_ids):
        return torch.full_like(hidden_states[..., :1], self.value)


class _SinglePositionLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        **kwargs,
    ):
        self.seen_position_embeddings = position_embeddings
        self.seen_attention_mask = attention_mask
        return (hidden_states,)


class _DualPositionLayer(nn.Module):
    attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        attention_mask=None,
        **kwargs,
    ):
        self.seen_global = position_embeddings_global
        self.seen_local = position_embeddings_local
        self.seen_attention_mask = attention_mask
        return (hidden_states,)


class _FakeInner(nn.Module):
    def __init__(self, layer, *, local_rotary=False):
        super().__init__()
        self.layers = nn.ModuleList([layer])
        self.norm = nn.Identity()
        self.rotary_emb = _Rotary(1)
        if local_rotary:
            self.rotary_emb_local = _Rotary(2)
        self.config = SimpleNamespace()


class _FakeCausalLM(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.model = inner


class TestEncoderPooling(unittest.TestCase):
    def test_normalize_last_token_alias(self):
        self.assertEqual(
            normalize_encoder_pooling("last_token"), POOLING_LAST_INSTRUCTION
        )

    def test_last_instruction_uses_final_definition_token(self):
        out = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [100.0, 0.0],
                    [200.0, 0.0],
                    [300.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
        instr = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]])
        pooled = _pool_hidden_states(
            out,
            attn,
            pooling=POOLING_LAST_INSTRUCTION,
            instruction_mask=instr,
        )
        self.assertTrue(torch.allclose(pooled, torch.tensor([[300.0, 0.0]])))

    def test_instruction_mean_averages_definition_tokens(self):
        out = torch.tensor([[[1.0], [2.0], [10.0], [20.0], [30.0], [0.0]]])
        attn = torch.tensor([[1, 1, 1, 1, 1, 0]])
        instr = torch.tensor([[0, 0, 1, 1, 1, 0]])
        pooled = _pool_hidden_states(
            out,
            attn,
            pooling=POOLING_INSTRUCTION_MEAN,
            instruction_mask=instr,
        )
        self.assertTrue(torch.allclose(pooled, torch.tensor([[20.0]])))

    def test_instruction_mean_requires_mask(self):
        out = torch.ones(1, 2, 4)
        attn = torch.ones(1, 2)
        with self.assertRaises(RuntimeError):
            _pool_hidden_states(
                out, attn, pooling=POOLING_INSTRUCTION_MEAN, instruction_mask=None
            )


class TestDecoderLayerCompatibility(unittest.TestCase):
    def test_single_position_embedding_uses_standard_causal_mask(self):
        encoder = build_semantic_encoder(_FakeCausalLM(_FakeInner(_SinglePositionLayer())))
        hidden = torch.ones(1, 3, 2)
        attention = torch.ones(1, 3, dtype=torch.long)
        mask = object()

        with patch(
            "boreft.pyreft.semantic_encoder.create_causal_mask", return_value=mask
        ) as create_mask:
            encoder(hidden, attention)

        create_mask.assert_called_once()
        self.assertIs(encoder.layer.seen_attention_mask, mask)
        self.assertTrue(
            torch.equal(
                encoder.layer.seen_position_embeddings,
                torch.ones(1, 3, 1),
            )
        )

    def test_dual_position_embeddings_use_sliding_mask(self):
        inner = _FakeInner(_DualPositionLayer(), local_rotary=True)
        encoder = build_semantic_encoder(_FakeCausalLM(inner))
        hidden = torch.ones(1, 3, 2)
        attention = torch.ones(1, 3, dtype=torch.long)
        mask = object()

        with patch(
            "boreft.pyreft.semantic_encoder.create_sliding_window_causal_mask",
            return_value=mask,
        ) as create_mask:
            encoder(hidden, attention)

        create_mask.assert_called_once()
        self.assertIs(encoder.layer.seen_attention_mask, mask)
        self.assertTrue(torch.equal(encoder.layer.seen_global, torch.ones(1, 3, 1)))
        self.assertTrue(
            torch.equal(encoder.layer.seen_local, torch.full((1, 3, 1), 2.0))
        )


class TestInstructionMasks(unittest.TestCase):
    def test_char_span_for_semantle_template(self):
        start, end = definition_instruction_char_span(
            "semantle", "laptop", "a portable machine."
        )
        full = definition_embedding_text("semantle", "laptop", "a portable machine.")
        self.assertEqual(full[start:end], "a portable machine.")

    def test_char_start_matches_span(self):
        start = definition_instruction_char_start("semantle", "laptop")
        span_start, _ = definition_instruction_char_span("semantle", "laptop", "x")
        self.assertEqual(start, span_start)

    def test_offset_mask_excludes_suffix(self):
        offsets = [(0, 1), (1, 4), (4, 7), (7, 12), (12, 16)]
        # chars: T|mpl|def| SUF|X
        mask = instruction_mask_from_offset_mapping(offsets, char_start=4, char_end=7)
        self.assertEqual(mask, [0, 0, 1, 0, 0])

    def test_batched_masks_respect_attention(self):
        offsets = [
            [(0, 1), (1, 2), (2, 3), (0, 0)],
            [(0, 1), (1, 2), (0, 0), (0, 0)],
        ]
        attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        mask = instruction_masks_from_batched_encoding(
            offsets,
            attn,
            [(1, 3), (1, 2)],
        )
        self.assertTrue(torch.equal(mask, torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]])))

    def test_encode_definition_inputs_batched_mask_alignment(self):
        class _Tok:
            pad_token_id = 0
            is_fast = True

            def __call__(self, texts, **kwargs):
                batch = []
                offsets = []
                for text in texts:
                    ids = [ord(c) for c in text]
                    offs = [(i, i + 1) for i in range(len(text))]
                    batch.append(ids)
                    offsets.append(offs)
                max_len = max(len(row) for row in batch)
                input_ids = []
                attn = []
                for ids, offs in zip(batch, offsets):
                    pad = max_len - len(ids)
                    input_ids.append(ids + [0] * pad)
                    attn.append([1] * len(ids) + [0] * pad)
                    offs = offs + [(0, 0)] * pad
                out = {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attn),
                }
                if kwargs.get("return_offsets_mapping"):
                    out["offset_mapping"] = offsets
                return out

        word, defn = "ab", "xyz"
        full = definition_embedding_text("semantle", word, defn)
        _ids, attn, instr = encode_definition_inputs(
            _Tok(),
            [full],
            device=torch.device("cpu"),
            max_length=64,
            task="semantle",
            word_definition_pairs=[(word, defn)],
            build_instruction_mask=True,
        )
        start, end = definition_instruction_char_span("semantle", word, defn)
        expected = [
            1 if start <= i < end else 0 for i in range(len(full))
        ] + [0] * (attn.shape[1] - len(full))
        self.assertEqual(instr[0].tolist(), expected)

    def test_encode_definition_inputs_requires_fast_for_instruction_mask(self):
        class _SlowTok:
            pad_token_id = 0
            is_fast = False
            padding_side = "right"

            def __call__(self, *args, **kwargs):
                raise AssertionError("tokenizer should not be called")

        with self.assertRaisesRegex(TypeError, "fast tokenizer"):
            encode_definition_inputs(
                _SlowTok(),
                ["x"],
                device=torch.device("cpu"),
                task="semantle",
                word_definition_pairs=[("a", "b")],
                build_instruction_mask=True,
            )


if __name__ == "__main__":
    unittest.main()
