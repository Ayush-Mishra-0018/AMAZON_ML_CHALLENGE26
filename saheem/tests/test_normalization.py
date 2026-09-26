from __future__ import annotations

import unittest

from entity_resolution.normalization import address_views, name_views, normalize_text


class NormalizationTests(unittest.TestCase):
    def test_case_spacing_punctuation_and_legal_suffix(self) -> None:
        views = name_views("  Líreque & Co., LTD. ")
        self.assertEqual(views.normalized, "líreque and co ltd")
        self.assertEqual(views.latin_folded, "lireque and co ltd")
        self.assertEqual(views.core, "líreque")

    def test_indic_script_is_preserved(self) -> None:
        value = "సాయి ఫుడ్స్ ఎల్‌ఎల్‌పీ"
        views = name_views(value)
        self.assertIn("సాయి", views.normalized)
        self.assertIn("TELUGU", views.scripts)
        self.assertEqual(views.latin_folded, views.normalized)

    def test_nfkc_unifies_compatibility_characters(self) -> None:
        self.assertEqual(normalize_text("ＡＰＥＸ  INC"), "apex inc")

    def test_address_views_keep_numbers_and_canonicalize_abbreviations(self) -> None:
        views = address_views("1001 17th St., Apt 4B")
        self.assertEqual(views.canonical_tokens, "1001 17th street apartment 4b")
        self.assertEqual(views.numeric_tokens, ("1001", "17th", "4b"))

    def test_blank_address_has_missing_flag(self) -> None:
        self.assertTrue(address_views("   ").missing)


if __name__ == "__main__":
    unittest.main()
