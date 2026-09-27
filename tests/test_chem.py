"""Tests for SMILES canonicalization, validity, and Morgan/Tanimoto similarity."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from boreft.chem import (
    MORGAN_N_BITS,
    canonical_smiles,
    canonical_target_key,
    first_smiles_token,
    smiles_chemistry_problems,
    default_rdkit_map_path,
    is_valid_smiles,
    load_rdkit_descriptor_map,
    morgan_fingerprint,
    normalize_rdkit_values,
    rdkit_descriptor_values,
    rdkit_internal_diversity,
    rdkit_sim_per_text,
    rdkit_similarity,
    rdkit_similarity_from_values,
    repair_smiles,
    maybe_repair_invalid_smiles,
    robust_descriptor_stats,
    smiles_edit_dist_per_text,
    smiles_edit_distance,
    tanimoto_internal_diversity,
    tanimoto_sim_per_text,
    tanimoto_similarity,
    unwrap_smiles_tags,
    validity_rate,
    wrap_smiles_tags,
    wrap_mist_smiles_tags,
    append_mist_smiles_open_tag,
    generation_smiles,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEKULIZED = "CC(=O)OC1=CC=CC=C1C(=O)O"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ETHANOL = "CCO"


class SmilesTagTests(unittest.TestCase):
    def test_wrap_and_unwrap_round_trip(self):
        self.assertEqual(wrap_smiles_tags("CCO"), "<SMILES>CCO</SMILES>")
        self.assertEqual(unwrap_smiles_tags("<SMILES>CCO</SMILES>"), "CCO")

    def test_wrap_is_idempotent(self):
        self.assertEqual(
            wrap_smiles_tags("<SMILES>CCO</SMILES>"), "<SMILES>CCO</SMILES>"
        )

    def test_unwrap_extracts_from_prose(self):
        self.assertEqual(
            unwrap_smiles_tags("Sure. <SMILES>CCO</SMILES>"), "CCO"
        )

    def test_unwrap_prefers_the_last_complete_pair(self):
        self.assertEqual(
            unwrap_smiles_tags("<SMILES>CC</SMILES> then <SMILES>CCO</SMILES>"),
            "CCO",
        )

    def test_generation_smiles_respects_the_flag(self):
        self.assertEqual(generation_smiles("CCO", smiles_tags=False), "CCO")
        self.assertEqual(
            generation_smiles("CCO", smiles_tags=True), "<SMILES>CCO</SMILES>"
        )

    def test_unwrap_noop_without_tags(self):
        self.assertEqual(unwrap_smiles_tags("CCO"), "CCO")
        self.assertEqual(unwrap_smiles_tags("  CCO \n"), "CCO")

    def test_incomplete_tags_left_intact(self):
        self.assertEqual(unwrap_smiles_tags("<SMILES>CCO"), "<SMILES>CCO")
        self.assertEqual(unwrap_smiles_tags("<SMILES></SMILES>"), "<SMILES></SMILES>")

    def test_mist_wrap_is_the_close_tag_only(self):
        self.assertEqual(wrap_mist_smiles_tags("CCO"), "CCO [END_SMILES]")
        self.assertEqual(
            wrap_mist_smiles_tags("CCO [END_SMILES]"), "CCO [END_SMILES]"
        )
        self.assertEqual(
            wrap_mist_smiles_tags("CCO", include_open=True),
            "[START_SMILES] CCO [END_SMILES]",
        )

    def test_unwrap_mist_pair_and_close_only_suffix(self):
        self.assertEqual(
            unwrap_smiles_tags("[START_SMILES] CCO [END_SMILES]"), "CCO"
        )
        self.assertEqual(unwrap_smiles_tags("CCO [END_SMILES]"), "CCO")
        self.assertEqual(
            unwrap_smiles_tags("[BEGIN_SMILES] CCO [END_SMILES]"), "CCO"
        )

    def test_unwrap_mist_prefers_the_last_complete_pair(self):
        self.assertEqual(
            unwrap_smiles_tags(
                "[START_SMILES] CC [END_SMILES] then "
                "[START_SMILES] CCO [END_SMILES]"
            ),
            "CCO",
        )

    def test_unwrap_mist_from_prose_with_mol_name(self):
        self.assertEqual(
            unwrap_smiles_tags(
                "[START_MOL] aspirin [END_MOL][START_SMILES] "
                "CC(=O)Oc1ccccc1C(=O)O [END_SMILES] is used as"
            ),
            "CC(=O)Oc1ccccc1C(=O)O",
        )
        self.assertEqual(
            unwrap_smiles_tags(
                "[START_MOL] aspirin [END_MOL] CCO [END_SMILES]"
            ),
            "CCO",
        )

    def test_append_mist_open_tag_is_idempotent(self):
        prompt = "Here is a valid SMILES string:"
        once = append_mist_smiles_open_tag(prompt)
        self.assertTrue(once.endswith("[START_SMILES]"))
        self.assertEqual(append_mist_smiles_open_tag(once), once)

    def test_generation_smiles_mist_flag(self):
        self.assertEqual(
            generation_smiles("CCO", mist_smiles_tags=True),
            "CCO [END_SMILES]",
        )
        self.assertEqual(
            generation_smiles("CCO", mist_smiles_tags=True, mist_open_in_target=True),
            "[START_SMILES] CCO [END_SMILES]",
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            generation_smiles("CCO", smiles_tags=True, mist_smiles_tags=True)

    def test_incomplete_mist_tags_left_intact(self):
        self.assertEqual(
            unwrap_smiles_tags("[START_SMILES] CCO"), "[START_SMILES] CCO"
        )
        self.assertEqual(
            unwrap_smiles_tags("[START_SMILES][END_SMILES]"),
            "[START_SMILES][END_SMILES]",
        )

    def test_mist_open_tag_in_prompt_is_completion_only(self):
        from boreft.chem import mist_open_tag_in_prompt

        self.assertTrue(
            mist_open_tag_in_prompt(
                mist_smiles_tags=True, use_chat_template=False
            )
        )
        self.assertFalse(
            mist_open_tag_in_prompt(
                mist_smiles_tags=True, use_chat_template=True
            )
        )
        self.assertFalse(
            mist_open_tag_in_prompt(
                mist_smiles_tags=False, use_chat_template=False
            )
        )


class FirstSmilesTokenTests(unittest.TestCase):
    def test_bare_smiles(self):
        self.assertEqual(first_smiles_token("CCO"), "CCO")

    def test_stops_at_space_or_newline(self):
        self.assertEqual(first_smiles_token("CCO extra words"), "CCO")
        self.assertEqual(first_smiles_token("CCO\nCCC"), "CCO")

    def test_unwraps_tags_then_takes_first_token(self):
        self.assertEqual(first_smiles_token("<SMILES>CCO CCC</SMILES>"), "CCO")
        self.assertEqual(
            first_smiles_token("Sure. <SMILES>CCO</SMILES>"), "CCO"
        )
        self.assertEqual(
            first_smiles_token("[START_SMILES] CCO extra [END_SMILES]"), "CCO"
        )
        self.assertEqual(first_smiles_token("CCO [END_SMILES]"), "CCO")

    def test_empty(self):
        self.assertEqual(first_smiles_token(""), "")
        self.assertEqual(first_smiles_token("   \n"), "")


class CanonicalSmilesTests(unittest.TestCase):
    def test_round_trip_is_stable(self):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        once = canonical_smiles(aspirin)
        self.assertIsNotNone(once)
        self.assertEqual(canonical_smiles(once), once)

    def test_equivalent_spellings_collapse(self):
        """Ethanol written from either end is the same molecule."""
        self.assertEqual(canonical_smiles("CCO"), canonical_smiles("OCC"))

    def test_isomeric_flag_keeps_or_drops_stereo(self):
        trans = "F/C=C/F"
        cis = r"F/C=C\F"
        self.assertNotEqual(
            canonical_smiles(trans, isomeric=True),
            canonical_smiles(cis, isomeric=True),
        )
        self.assertEqual(
            canonical_smiles(trans, isomeric=False),
            canonical_smiles(cis, isomeric=False),
        )

    def test_kekulized_and_aromatic_benzene_agree(self):
        self.assertEqual(canonical_smiles("c1ccccc1"), canonical_smiles("C1=CC=CC=C1"))

    def test_surrounding_whitespace_ignored(self):
        self.assertEqual(canonical_smiles("  CCO \n"), canonical_smiles("CCO"))

    def test_smiles_tags_are_stripped_before_parse(self):
        self.assertEqual(canonical_smiles("<SMILES>CCO</SMILES>"), canonical_smiles("CCO"))
        self.assertTrue(is_valid_smiles("<SMILES>CCO</SMILES>"))
        self.assertEqual(
            canonical_smiles("[START_SMILES] CCO [END_SMILES]"),
            canonical_smiles("CCO"),
        )
        self.assertTrue(is_valid_smiles("CCO [END_SMILES]"))
        self.assertEqual(
            canonical_target_key("<SMILES>OCC</SMILES>"),
            canonical_target_key("CCO"),
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(canonical_smiles("not a molecule"))
        self.assertIsNone(canonical_smiles("C(C(C"))
        self.assertIsNone(canonical_smiles(""))
        self.assertIsNone(canonical_smiles("   "))

    def test_moltosmiles_invariant_returns_none(self):
        """RDKit Canon.cpp can raise after MolFromSmiles succeeds.

        Full eval used to abort RECON / RECON_TEST on that RuntimeError.
        """
        from boreft.chem import _canonical_stripped, _chem

        Chem = _chem()
        _canonical_stripped.cache_clear()

        def boom(mol, *args, **kwargs):
            raise RuntimeError(
                "Invariant Violation\n\tcould not find atom1\n"
                "\tViolation occurred on line 227 in file Code/GraphMol/Canon.cpp"
            )

        try:
            with patch.object(Chem, "MolToSmiles", side_effect=boom):
                self.assertIsNone(canonical_smiles(ETHANOL))
                self.assertFalse(is_valid_smiles(ETHANOL))
                self.assertEqual(canonical_target_key(ETHANOL), ETHANOL)
                self.assertEqual(smiles_edit_distance(ETHANOL, ETHANOL), 0)
                self.assertGreater(smiles_edit_distance(ETHANOL, "OCC"), 0)
        finally:
            _canonical_stripped.cache_clear()


class IsValidSmilesTests(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertTrue(is_valid_smiles("CC(=O)Oc1ccccc1C(=O)O"))
        self.assertTrue(is_valid_smiles("CCO"))
        self.assertFalse(is_valid_smiles("banana"))
        self.assertFalse(is_valid_smiles(""))


class SmilesChemistryProblemsTests(unittest.TestCase):
    def test_valid_and_syntax_errors_have_no_problems(self):
        self.assertEqual(smiles_chemistry_problems("CCO"), ())
        self.assertEqual(smiles_chemistry_problems("banana"), ())
        self.assertEqual(smiles_chemistry_problems(""), ())

    def test_valence_error_is_reported(self):
        problems = smiles_chemistry_problems("C(C)(C)(C)(C)C")
        self.assertTrue(problems)
        self.assertTrue(any(p.startswith("AtomValenceException:") for p in problems))
        self.assertTrue(any("valence" in p.lower() for p in problems))

    def test_kekulize_error_is_reported(self):
        problems = smiles_chemistry_problems("c1cccc1")
        self.assertTrue(problems)
        self.assertTrue(any("KekulizeException:" in p for p in problems))

    def test_smiles_tags_are_unwrapped(self):
        problems = smiles_chemistry_problems("<SMILES>C(C)(C)(C)(C)C</SMILES>")
        self.assertTrue(any(p.startswith("AtomValenceException:") for p in problems))


def _smiself_available() -> bool:
    try:
        import smiself  # noqa: F401
    except ImportError:
        return False
    return True


class RepairSmilesFallbackTests(unittest.TestCase):
    def test_missing_package_returns_none_for_invalid(self):
        import sys

        with patch.dict(sys.modules, {"smiself": None}):
            self.assertIsNone(repair_smiles("C(C(C"))


class MaybeRepairInvalidSmilesTests(unittest.TestCase):
    def test_valid_smiles_are_left_unchanged(self):
        smiles, repaired = maybe_repair_invalid_smiles("OCC")
        self.assertEqual(smiles, "OCC")
        self.assertFalse(repaired)

    def test_tags_are_unwrapped_before_validity_check(self):
        smiles, repaired = maybe_repair_invalid_smiles("CCO [END_SMILES]")
        self.assertEqual(smiles, "CCO")
        self.assertFalse(repaired)

    def test_invalid_is_replaced_when_repair_parses(self):
        with patch("boreft.chem.repair_smiles", return_value="CCO"):
            smiles, repaired = maybe_repair_invalid_smiles(
                "not a molecule [END_SMILES]"
            )
        self.assertEqual(smiles, "CCO")
        self.assertTrue(repaired)

    def test_invalid_is_kept_when_repair_fails(self):
        with patch("boreft.chem.repair_smiles", return_value=None):
            smiles, repaired = maybe_repair_invalid_smiles("not a molecule")
        self.assertEqual(smiles, "not a molecule")
        self.assertFalse(repaired)


@unittest.skipUnless(_smiself_available(), "smiself is not installed")
class RepairSmilesTests(unittest.TestCase):
    def test_valid_smiles_are_canonical_without_rewriting(self):
        self.assertEqual(repair_smiles("OCC"), canonical_smiles("CCO"))

    def test_readme_extra_paren_becomes_aspirin(self):
        """SmiSelf README example: trailing ')' on kekulized aspirin."""
        self.assertFalse(is_valid_smiles("CC(=O)OC1=CC=CC=C1=C(=O)O)"))
        self.assertEqual(
            repair_smiles("CC(=O)OC1=CC=CC=C1=C(=O)O)"),
            canonical_smiles(ASPIRIN),
        )

    def test_smiles_tags_are_unwrapped(self):
        tagged = "<SMILES>CC(=O)OC1=CC=CC=C1=C(=O)O)</SMILES>"
        self.assertEqual(repair_smiles(tagged), canonical_smiles(ASPIRIN))

    def test_empty_is_none(self):
        self.assertEqual(repair_smiles(""), None)
        self.assertEqual(repair_smiles("   "), None)

    def test_encoder_index_error_returns_none(self):
        import smiself

        with patch.object(smiself, "encoder", side_effect=IndexError):
            self.assertIsNone(repair_smiles("C(C(C"))


class CanonicalTargetKeyTests(unittest.TestCase):
    def test_equivalent_molecules_share_a_key(self):
        self.assertEqual(canonical_target_key("CCO"), canonical_target_key("OCC"))

    def test_case_distinguishes_distinct_molecules(self):
        """SMILES are case-sensitive: cyclohexane is not benzene."""
        self.assertNotEqual(
            canonical_target_key("C1CCCCC1"), canonical_target_key("c1ccccc1")
        )

    def test_lowercasing_a_target_would_change_its_meaning(self):
        """The Semantle text normalizer would have merged these two."""
        chloro = "CCCl"
        self.assertNotEqual(
            canonical_target_key(chloro), canonical_target_key(chloro.lower())
        )

    def test_distinct_invalid_decodes_stay_distinct(self):
        self.assertNotEqual(
            canonical_target_key("gibberish one"),
            canonical_target_key("gibberish two"),
        )

    def test_invalid_key_collapses_whitespace_but_keeps_case(self):
        self.assertEqual(canonical_target_key("  Xy   zzy  "), "Xy zzy")

    def test_invalid_decode_never_matches_a_valid_target(self):
        self.assertNotEqual(canonical_target_key("CCO"), canonical_target_key("CC0"))


class SmilesEditDistanceTests(unittest.TestCase):
    def test_identity_and_equivalent_spelling_are_zero(self):
        self.assertEqual(smiles_edit_distance(ETHANOL, ETHANOL), 0)
        self.assertEqual(smiles_edit_distance("CCO", "OCC"), 0)

    def test_single_atom_substitution_is_one(self):
        self.assertEqual(smiles_edit_distance("CCO", "CCN"), 1)

    def test_invalid_decode_uses_raw_string_distance(self):
        dist = smiles_edit_distance(ETHANOL, "not a molecule")
        self.assertGreater(dist, 0)

    def test_smiles_tags_are_ignored(self):
        self.assertEqual(
            smiles_edit_distance("<SMILES>CCO</SMILES>", "OCC"), 0
        )

    def test_per_text_is_parallel_to_inputs(self):
        dists = smiles_edit_dist_per_text(
            [ETHANOL, ETHANOL], [ETHANOL, "CCN"]
        )
        np.testing.assert_allclose(dists, [0.0, 1.0])


class RdkitDescriptorTests(unittest.TestCase):
    def test_vector_has_fixed_scalar_shape(self):
        values = rdkit_descriptor_values(ETHANOL)
        self.assertIsNotNone(values)
        self.assertEqual(len(values), 10)
        self.assertTrue(all(isinstance(v, (int, float)) for v in values))

    def test_invalid_smiles_has_no_vector(self):
        self.assertIsNone(rdkit_descriptor_values("not a molecule"))

    def test_equivalent_spellings_have_identical_vectors(self):
        self.assertEqual(
            rdkit_descriptor_values("CCO"), rdkit_descriptor_values("OCC")
        )

    def test_non_round_trippable_canonical_does_not_raise(self):
        """RDKit can canonicalize a valid mol to a SMILES it will not re-parse.

        Generation eval used to crash in ``rdkit_similarity`` on that case.
        Scoring must return a vector or None, never raise.
        """
        smiles = "CCCCOCc1cc2oc1C=2"
        values = rdkit_descriptor_values(smiles)
        if values is None:
            self.assertEqual(rdkit_similarity(ETHANOL, smiles), 0.0)
        else:
            self.assertEqual(len(values), 10)
            sim = rdkit_similarity(ETHANOL, smiles)
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)

    def test_canonical_reparse_failure_scores_the_parsed_mol(self):
        """A failed canonical re-parse must still yield descriptors from the mol."""
        from boreft.chem import _chem, _descriptor_values_stripped

        original = "OCC"
        expected = rdkit_descriptor_values(original)
        self.assertIsNotNone(expected)
        Chem = _chem()
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(original))
        self.assertNotEqual(original, canonical)
        real_from_smiles = Chem.MolFromSmiles

        def reject_canonical(text, *args, **kwargs):
            if text == canonical:
                return None
            return real_from_smiles(text, *args, **kwargs)

        _descriptor_values_stripped.cache_clear()
        with patch.object(Chem, "MolFromSmiles", side_effect=reject_canonical):
            values = rdkit_descriptor_values(original)
        self.assertEqual(values, expected)

    def test_moltosmiles_invariant_still_scores_descriptors(self):
        """Canon.cpp abort must not kill scoring; fall back to the parsed mol."""
        from boreft.chem import _canonical_stripped, _chem, _descriptor_values_stripped

        expected = rdkit_descriptor_values(ETHANOL)
        self.assertIsNotNone(expected)
        Chem = _chem()
        _canonical_stripped.cache_clear()
        _descriptor_values_stripped.cache_clear()

        def boom(mol, *args, **kwargs):
            raise RuntimeError(
                "Invariant Violation\n\tcould not find atom1\n"
                "\tViolation occurred on line 227 in file Code/GraphMol/Canon.cpp"
            )

        try:
            with patch.object(Chem, "MolToSmiles", side_effect=boom):
                values = rdkit_descriptor_values(ETHANOL)
                self.assertEqual(values, expected)
                self.assertGreater(rdkit_similarity(ETHANOL, "OCC"), 0.0)
                sims = rdkit_sim_per_text([ETHANOL, "banana"], [ETHANOL, "CC"])
                self.assertEqual(len(sims), 2)
                self.assertGreater(float(sims[0]), 0.0)
                self.assertEqual(float(sims[1]), 0.0)
        finally:
            _canonical_stripped.cache_clear()
            _descriptor_values_stripped.cache_clear()

    def test_robust_stats_fall_back_for_constant_dimensions(self):
        rows = [[float(i)] + [0.0] * 9 for i in range(3)]
        stats = robust_descriptor_stats(rows)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(len(stats["median"]), 10)
        self.assertTrue(all(scale > 0 for scale in stats["scale"]))

    def test_map_rejects_descriptor_name_mismatch(self):
        with open(default_rdkit_map_path(), encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["positions"][0]["name"] = "wrong_descriptor"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as f:
            json.dump(metadata, f)
            f.flush()
            with self.assertRaisesRegex(ValueError, "exactly match schema"):
                load_rdkit_descriptor_map(f.name)

    def test_map_cache_refreshes_when_content_changes(self):
        with open(default_rdkit_map_path(), encoding="utf-8") as f:
            metadata = json.load(f)
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", encoding="utf-8"
        ) as f:
            json.dump(metadata, f)
            f.flush()
            first = load_rdkit_descriptor_map(f.name)
            metadata["normalization"]["median"][0] += 1.0
            f.seek(0)
            f.truncate()
            json.dump(metadata, f)
            f.flush()
            second = load_rdkit_descriptor_map(f.name)
        self.assertNotEqual(
            first["normalization"]["median"][0],
            second["normalization"]["median"][0],
        )


class RdkitSimilarityTests(unittest.TestCase):
    def test_identity_and_equivalent_spelling_score_one(self):
        self.assertEqual(rdkit_similarity(ETHANOL, ETHANOL), 1.0)
        self.assertEqual(rdkit_similarity("CCO", "OCC"), 1.0)

    def test_invalid_side_scores_zero(self):
        self.assertEqual(rdkit_similarity(ETHANOL, "not a molecule"), 0.0)

    def test_similarity_is_symmetric_and_bounded(self):
        ab = rdkit_similarity(ASPIRIN, SALICYLIC_ACID)
        ba = rdkit_similarity(SALICYLIC_ACID, ASPIRIN)
        self.assertAlmostEqual(ab, ba)
        self.assertGreaterEqual(ab, 0.0)
        self.assertLessEqual(ab, 1.0)

    def test_per_text_is_parallel_to_inputs(self):
        sims = rdkit_sim_per_text(
            [ETHANOL, ETHANOL], [ETHANOL, "not a molecule"]
        )
        np.testing.assert_allclose(sims, [1.0, 0.0])

    def test_internal_diversity_drops_invalid_and_detects_collapse(self):
        self.assertEqual(
            rdkit_internal_diversity([ETHANOL, ETHANOL, "junk"]), 0.0
        )
        self.assertIsNone(rdkit_internal_diversity([ETHANOL, "junk"]))
        diverse = rdkit_internal_diversity([ETHANOL, ASPIRIN])
        self.assertIsNotNone(diverse)
        self.assertGreater(diverse, 0.0)

    def test_from_values_matches_smiles_similarity(self):
        a = rdkit_descriptor_values(ETHANOL)
        b = rdkit_descriptor_values(ASPIRIN)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(
            rdkit_similarity_from_values(a, b),
            rdkit_similarity(ETHANOL, ASPIRIN),
        )

    def test_from_values_identity_is_one(self):
        a = rdkit_descriptor_values(ETHANOL)
        self.assertEqual(rdkit_similarity_from_values(a, a), 1.0)

    def test_from_values_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, r"shape \(10,\)"):
            rdkit_similarity_from_values([1.0], [1.0])
        with self.assertRaisesRegex(ValueError, r"shape \(10,\)"):
            normalize_rdkit_values([1.0] * 9)


class ValidityRateTests(unittest.TestCase):
    def test_fraction_of_parseable_decodes(self):
        self.assertAlmostEqual(
            validity_rate(["CCO", "nonsense", "c1ccccc1"]), 2.0 / 3.0
        )

    def test_all_valid_and_all_invalid(self):
        self.assertEqual(validity_rate(["CCO", "CCC"]), 1.0)
        self.assertEqual(validity_rate(["???", ""]), 0.0)

    def test_empty_list_is_zero(self):
        self.assertEqual(validity_rate([]), 0.0)


class MorganFingerprintTests(unittest.TestCase):
    def test_fingerprint_width_matches_the_module_default(self):
        self.assertEqual(morgan_fingerprint(ASPIRIN).GetNumBits(), MORGAN_N_BITS)

    def test_unparseable_and_blank_have_no_fingerprint(self):
        self.assertIsNone(morgan_fingerprint("banana"))
        self.assertIsNone(morgan_fingerprint(""))
        self.assertIsNone(morgan_fingerprint("   "))

    def test_smiles_tags_are_stripped_before_fingerprint(self):
        self.assertEqual(
            tanimoto_similarity("<SMILES>CCO</SMILES>", "CCO"),
            1.0,
        )


class TanimotoSimilarityTests(unittest.TestCase):
    def test_a_molecule_is_identical_to_itself(self):
        self.assertEqual(tanimoto_similarity(ASPIRIN, ASPIRIN), 1.0)

    def test_fingerprints_ignore_smiles_spelling(self):
        """Unlike embed_sim, TFS needs no canonicalization to see one molecule."""
        self.assertEqual(tanimoto_similarity(ASPIRIN, ASPIRIN_KEKULIZED), 1.0)
        self.assertEqual(tanimoto_similarity(ETHANOL, "OCC"), 1.0)

    def test_related_scaffolds_score_above_unrelated_ones(self):
        related = tanimoto_similarity(ASPIRIN, SALICYLIC_ACID)
        unrelated = tanimoto_similarity(ASPIRIN, ETHANOL)
        self.assertGreater(related, unrelated)
        self.assertTrue(0.0 < unrelated < related < 1.0)

    def test_similarity_is_symmetric(self):
        self.assertEqual(
            tanimoto_similarity(ASPIRIN, ETHANOL),
            tanimoto_similarity(ETHANOL, ASPIRIN),
        )

    def test_invalid_side_scores_zero(self):
        """An unparseable decode is a total miss, not an error."""
        self.assertEqual(tanimoto_similarity(ASPIRIN, "banana"), 0.0)
        self.assertEqual(tanimoto_similarity("banana", ASPIRIN), 0.0)
        self.assertEqual(tanimoto_similarity("", ASPIRIN), 0.0)


class TanimotoPerTextTests(unittest.TestCase):
    def test_values_are_parallel_to_the_inputs(self):
        sims = tanimoto_sim_per_text(
            [ASPIRIN, ASPIRIN, ETHANOL], [ASPIRIN_KEKULIZED, "banana", ETHANOL]
        )
        self.assertEqual(sims.shape, (3,))
        np.testing.assert_allclose(sims[[0, 2]], [1.0, 1.0])
        self.assertEqual(sims[1], 0.0)

    def test_empty_input_returns_an_empty_array(self):
        self.assertEqual(tanimoto_sim_per_text([], []).shape, (0,))


class TanimotoInternalDiversityTests(unittest.TestCase):
    def test_one_repeated_molecule_has_no_diversity(self):
        self.assertEqual(
            tanimoto_internal_diversity([ASPIRIN, ASPIRIN_KEKULIZED]), 0.0
        )

    def test_unrelated_molecules_are_more_diverse_than_related_ones(self):
        related = tanimoto_internal_diversity([ASPIRIN, SALICYLIC_ACID])
        unrelated = tanimoto_internal_diversity([ASPIRIN, ETHANOL])
        self.assertGreater(unrelated, related)

    def test_too_few_parseable_molecules_is_undefined(self):
        self.assertIsNone(tanimoto_internal_diversity([]))
        self.assertIsNone(tanimoto_internal_diversity([ASPIRIN]))
        self.assertIsNone(tanimoto_internal_diversity(["banana", "gibberish"]))

    def test_invalid_entries_are_dropped_rather_than_scored(self):
        """Junk decodes must not inflate diversity; validity_rate reports them."""
        self.assertEqual(
            tanimoto_internal_diversity([ASPIRIN, ASPIRIN_KEKULIZED, "banana"]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
