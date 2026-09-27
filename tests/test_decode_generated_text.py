import unittest

from boreft.data_utils import decode_generated_text


class _FakeOutputIds:
    def __init__(self, rows: list[list[int]]):
        self._rows = rows

    def __getitem__(self, key):
        batch_idx, sl = key
        return self._rows[batch_idx][sl]


class _FakeTokenizer:
    """Maps token ids to string fragments for decode tests."""

    def __init__(self):
        self._pieces = {
            10: "hello",
            11: " ",
            12: "world",
            13: " again",
            14: "apple",
            15: "<SMILES>",
            16: "CCO",
            17: "</SMILES>",
            18: "Sure. ",
            99: "<|im_end|>",
            100: "\n",
        }
        self.eos_token_id = 2
        self.pad_token_id = 0
        self._special_ids = {self.eos_token_id, self.pad_token_id, 99}

    def decode(self, ids, skip_special_tokens: bool = False):
        out = []
        for tid in ids:
            if skip_special_tokens and tid in self._special_ids:
                continue
            out.append(self._pieces.get(tid, f"[{tid}]"))
        return "".join(out)


class TestDecodeGeneratedText(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        # prompt: [10, 11]  generation: [12, 13, 2]
        self.output_ids = _FakeOutputIds([[10, 11, 12, 13, 2]])

    def test_full_decode_until_eos_without_suffix(self):
        text = decode_generated_text(self.tokenizer, self.output_ids, prompt_len=2)
        self.assertEqual(text, "world again")

    def test_truncates_at_eos_even_when_suffix_has_trailing_newline(self):
        """Qwen/Gemma: assistant_suffix is '<|im_end|>\\n' but generate stops at EOS."""
        self.tokenizer.eos_token_id = 99
        output_ids = _FakeOutputIds([[10, 11, 14, 99]])
        text = decode_generated_text(
            self.tokenizer,
            output_ids,
            prompt_len=2,
            assistant_suffix="<|im_end|>\n",
        )
        self.assertEqual(text, "apple")

    def test_drops_tokens_after_eos(self):
        self.tokenizer.eos_token_id = 99
        # Trailing newline after EOS must not appear in the decode.
        output_ids = _FakeOutputIds([[10, 11, 14, 99, 100]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "apple")

    def test_respects_batch_idx(self):
        batch = _FakeOutputIds(
            [
                [10, 11, 12, 2],
                [10, 11, 13, 2],
            ]
        )
        self.assertEqual(
            decode_generated_text(self.tokenizer, batch, prompt_len=2, batch_idx=0),
            "world",
        )
        self.assertEqual(
            decode_generated_text(self.tokenizer, batch, prompt_len=2, batch_idx=1),
            "again",  # decode strips leading/trailing whitespace
        )

    def test_returns_truncated_generation_when_eos_not_emitted(self):
        output_ids = _FakeOutputIds([[10, 11, 12, 13]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "world again")

    def test_empty_generation(self):
        output_ids = _FakeOutputIds([[10, 11]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "")

    def test_unwraps_smiles_tags(self):
        output_ids = _FakeOutputIds([[10, 11, 15, 16, 17, 2]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "CCO")

    def test_unwraps_smiles_tags_from_prose(self):
        output_ids = _FakeOutputIds([[10, 11, 18, 15, 16, 17, 2]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "CCO")

    def test_unwraps_the_last_smiles_tag_pair(self):
        self.tokenizer._pieces[19] = "<SMILES>CC</SMILES> then "
        output_ids = _FakeOutputIds([[10, 11, 19, 15, 16, 17, 2]])
        text = decode_generated_text(self.tokenizer, output_ids, prompt_len=2)
        self.assertEqual(text, "CCO")


if __name__ == "__main__":
    unittest.main()
