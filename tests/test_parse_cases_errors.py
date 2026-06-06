"""Error-path tests for benchmarks.cases.parse_cases."""

import unittest

from benchmarks.cases import parse_cases


class ParseCasesErrorTests(unittest.TestCase):
    def test_rejects_missing_x_separator(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_cases("1024")
        self.assertIn("n_samples>x<variability_samples", str(ctx.exception))

    def test_rejects_non_integer_tokens(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_cases("abcxdef")
        self.assertIn("must be integers", str(ctx.exception))

    def test_rejects_zero_dimensions(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_cases("0x64")
        self.assertIn("must be positive", str(ctx.exception))

    def test_rejects_negative_dimensions(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_cases("-10x64")
        self.assertIn("must be positive", str(ctx.exception))

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            parse_cases("")

    def test_rejects_only_commas(self) -> None:
        with self.assertRaises(ValueError):
            parse_cases(", ,")

    def test_accepts_valid_multiple_cases(self) -> None:
        cases = parse_cases("1024x64, 2048x128")
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].n_samples, 1024)
        self.assertEqual(cases[0].variability_samples, 64)
        self.assertEqual(cases[1].n_samples, 2048)
        self.assertEqual(cases[1].variability_samples, 128)

    def test_accepts_uppercase_x(self) -> None:
        cases = parse_cases("100X50")
        self.assertEqual(cases[0].case_id, "100x50")


if __name__ == "__main__":
    unittest.main()
