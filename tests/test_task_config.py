import unittest
from unittest import mock

from boreft.task_config import (
    definition_embedding_text,
    definition_generation_field_label,
    definition_generation_instruction,
    definition_generation_query_label,
    definition_instruction_char_span,
    definition_instruction_char_start,
    sdpo_teacher_instruction,
    substitute_placeholders,
    task_config,
    task_embedding_model,
    task_supports_chat_template,
    task_supports_definition_generation,
    task_supports_fingerprints,
    task_supports_validity,
    task_system_prompt,
    task_target_kind,
)


class TestSdpoTeacherInstruction(unittest.TestCase):
    def test_plain_template(self):
        out = sdpo_teacher_instruction(
            "semantle", "  a small rodent  ", use_chat_template=False
        )
        self.assertIn("a small rodent", out)
        self.assertTrue(out.startswith("Here is the definition of an English word:"))

    def test_chat_template(self):
        out = sdpo_teacher_instruction(
            "semantle", "a small rodent", use_chat_template=True
        )
        self.assertIn("a small rodent", out)
        self.assertTrue(out.startswith("Here is the definition of an English word:"))

    def test_molopt_chat_template(self):
        """The prompt wording is a research knob; the contract is what is pinned."""
        definition = "a salicylate ester with anti-inflammatory activity"
        chat = sdpo_teacher_instruction("molopt", f"  {definition}  ", use_chat_template=True)
        plain = sdpo_teacher_instruction("molopt", definition, use_chat_template=False)
        for out in (chat, plain):
            self.assertIn(definition, out)
            self.assertNotIn("{definition}", out)
            self.assertEqual(out, out.strip())
        # Routing on the flag is the behavior worth asserting, not the wording.
        self.assertNotEqual(chat, plain)

    def test_hypogen_chat_template(self):
        definition = "To test whether temperature predicts diversification."
        chat = sdpo_teacher_instruction(
            "hypogen", f"  {definition}  ", use_chat_template=True
        )
        plain = sdpo_teacher_instruction("hypogen", definition, use_chat_template=False)
        for out in (chat, plain):
            self.assertIn(definition, out)
            self.assertNotIn("{definition}", out)
        self.assertNotEqual(chat, plain)

    def test_missing_chat_template_raises(self):
        """A task with only the plain teacher prompt must not silently fall back."""
        plain_only = {"sdpo_teacher_prompt": "definition: {definition}"}
        with mock.patch.dict(task_config, {"plain_only": plain_only}):
            self.assertEqual(
                sdpo_teacher_instruction("plain_only", "d"), "definition: d"
            )
            with self.assertRaises(KeyError) as ctx:
                sdpo_teacher_instruction("plain_only", "d", use_chat_template=True)
        self.assertIn("chat_sdpo_teacher_prompt", str(ctx.exception))


class TestTaskCapabilities(unittest.TestCase):
    def test_embedding_models(self):
        self.assertEqual(task_embedding_model("semantle"), "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(task_embedding_model("molopt"), "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(task_embedding_model("hypogen"), "Qwen/Qwen3-Embedding-0.6B")

    def test_target_kinds(self):
        self.assertEqual(task_target_kind("semantle"), "text")
        self.assertEqual(task_target_kind("molopt"), "smiles")
        self.assertEqual(task_target_kind("hypogen"), "text")

    def test_only_smiles_targets_have_validity(self):
        self.assertFalse(task_supports_validity("semantle"))
        self.assertTrue(task_supports_validity("molopt"))
        self.assertFalse(task_supports_validity("hypogen"))

    def test_only_smiles_targets_have_fingerprints(self):
        self.assertFalse(task_supports_fingerprints("semantle"))
        self.assertTrue(task_supports_fingerprints("molopt"))
        self.assertFalse(task_supports_fingerprints("hypogen"))

    def test_recon_eval_tasks_support_chat_template(self):
        self.assertTrue(task_supports_chat_template("semantle"))
        self.assertTrue(task_supports_chat_template("molopt"))
        self.assertTrue(task_supports_chat_template("hypogen"))

    def test_only_molopt_defines_a_system_prompt(self):
        self.assertIsNone(task_system_prompt("semantle"))
        self.assertIsNone(task_system_prompt("hypogen"))
        self.assertIsNone(task_system_prompt("arc"))
        molopt = task_system_prompt("molopt")
        self.assertIsNotNone(molopt)
        self.assertIn("SMILES", molopt)
        self.assertIn("[N+](=O)[O-]", molopt)

    def test_unknown_task_falls_back_to_text_defaults(self):
        self.assertEqual(task_target_kind("arc"), "text")
        self.assertEqual(task_embedding_model("arc"), "Qwen/Qwen3-Embedding-0.6B")
        self.assertFalse(task_supports_chat_template("arc"))
        self.assertFalse(task_supports_fingerprints("arc"))


class TestDefinitionGeneration(unittest.TestCase):
    def test_supported_tasks(self):
        self.assertTrue(task_supports_definition_generation("semantle"))
        self.assertTrue(task_supports_definition_generation("molopt"))
        self.assertTrue(task_supports_definition_generation("hypogen"))
        self.assertFalse(task_supports_definition_generation("arc"))

    def test_chat_flag_changes_semantle_prompt(self):
        plain = definition_generation_instruction("semantle", "cat")
        chat = definition_generation_instruction(
            "semantle", "cat", use_chat_template=True
        )
        self.assertNotEqual(plain, chat)
        self.assertIn("cat", chat)

    def test_field_labels(self):
        self.assertEqual(definition_generation_field_label("semantle"), "Definition")
        self.assertEqual(definition_generation_field_label("molopt"), "Description")
        self.assertEqual(
            definition_generation_field_label("hypogen"), "Research objective"
        )
        self.assertEqual(definition_generation_query_label("semantle"), "Word")
        self.assertEqual(definition_generation_query_label("molopt"), "Molecule")
        self.assertEqual(definition_generation_query_label("hypogen"), "Hypothesis")


class TestDefinitionEmbeddingText(unittest.TestCase):
    def test_semantle_template(self):
        out = definition_embedding_text("semantle", "Computer", "An Electronic Machine.")
        self.assertEqual(out, "The meaning of 'Computer' is: An Electronic Machine.")

    def test_molopt_template(self):
        """Both slots must be filled, whatever the current molopt wording is."""
        out = definition_embedding_text("molopt", "  CCO ", " a small alcohol. ")
        self.assertIn("CCO", out)
        self.assertIn("a small alcohol.", out)
        self.assertNotIn("{text}", out)
        self.assertNotIn("{definition}", out)
        self.assertEqual(out, out.strip())

    def test_hypogen_template(self):
        out = definition_embedding_text(
            "hypogen", "  H. ", " To test temperature. "
        )
        self.assertIn("H.", out)
        self.assertIn("To test temperature.", out)
        self.assertNotIn("{text}", out)
        self.assertNotIn("{definition}", out)

    def test_missing_template_raises(self):
        with self.assertRaises(KeyError) as ctx:
            definition_embedding_text("arc1d", "C", "def")
        self.assertIn("embedding_prompt_defn", str(ctx.exception))

    def test_instruction_char_start(self):
        start = definition_instruction_char_start("semantle", "laptop")
        full = definition_embedding_text("semantle", "laptop", "portable.")
        self.assertEqual(full[start:], "portable.")

    def test_instruction_char_span(self):
        start, end = definition_instruction_char_span(
            "semantle", "laptop", "portable."
        )
        full = definition_embedding_text("semantle", "laptop", "portable.")
        self.assertEqual(full[start:end], "portable.")
        self.assertNotIn("meaning", full[start:end])

    def test_braces_in_target_are_literal(self):
        """Hypogen targets can contain `{area}`; str.format would KeyError."""
        text = "Larger basins ({area}) have lower speciation."
        definition = "To test area vs speciation."
        out = definition_embedding_text("hypogen", text, definition)
        self.assertIn("{area}", out)
        self.assertIn(definition, out)
        start, end = definition_instruction_char_span("hypogen", text, definition)
        self.assertEqual(out[start:end], definition)
        self.assertEqual(
            substitute_placeholders("The meaning of '{text}'.", text=text),
            "The meaning of 'Larger basins ({area}) have lower speciation.'.",
        )
        self.assertEqual(
            substitute_placeholders(
                "x {text} y {definition}",
                text="has {definition} in it",
                definition="OBJ",
            ),
            "x has {definition} in it y OBJ",
        )


if __name__ == "__main__":
    unittest.main()
