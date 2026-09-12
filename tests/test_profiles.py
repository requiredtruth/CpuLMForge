import unittest
from cpulmforge.profiles import Sample, aggregate, select_profile

class ProfileTests(unittest.TestCase):
    def sample(self, threads: int, tokens: int, seconds: float, rss: int, run: str) -> Sample:
        return Sample("model with space.gguf", threads, 2048, tokens, seconds, rss, 256, run)

    def test_aggregates_median_and_worst_case(self) -> None:
        profile = aggregate([self.sample(4,100,10,1000,"a"), self.sample(4,120,10,1200,"b")])[0]
        self.assertEqual(profile.median_tokens_per_second, 11)
        self.assertEqual(profile.minimum_tokens_per_second, 10)
        self.assertEqual(profile.peak_rss_bytes, 1200)

    def test_selects_fastest_eligible_profile(self) -> None:
        samples = [self.sample(2,80,10,1000,"a"), self.sample(4,150,10,1500,"b")]
        result = select_profile(samples, memory_limit_bytes=2000, minimum_tps=5)
        self.assertEqual(result.selected.threads if result.selected else None, 4)
        self.assertIn("'model with space.gguf'", result.command or "")

    def test_rejects_memory_and_minimum_speed(self) -> None:
        result = select_profile([self.sample(4,40,10,3000,"a")], memory_limit_bytes=2000, minimum_tps=5)
        self.assertIsNone(result.selected)
        self.assertEqual(len(result.rejected[0]["reasons"]), 2)

    def test_rejects_non_finite_measurements(self) -> None:
        for seconds in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    self.sample(4, 40, seconds, 3000, "a")
        sample = self.sample(4, 10**1000, 5e-324, 3000, "a")
        with self.assertRaisesRegex(ValueError, "tokens_per_second must be finite"):
            aggregate([sample])

    def test_rejects_non_text_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_path must be non-empty text"):
            Sample(7, 4, 2048, 40, 10, 3000)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "run_id must be text"):
            Sample("model.gguf", 4, 2048, 40, 10, 3000, run_id=7)  # type: ignore[arg-type]

    def test_rejects_non_finite_selection_constraints(self) -> None:
        sample = self.sample(4, 40, 10, 3000, "a")
        with self.assertRaisesRegex(ValueError, "minimum_tps must be a finite"):
            select_profile([sample], memory_limit_bytes=4000, minimum_tps=float("nan"))
        with self.assertRaisesRegex(ValueError, "memory_limit_bytes must be a positive integer"):
            select_profile([sample], memory_limit_bytes=True)  # type: ignore[arg-type]

if __name__ == "__main__":
    unittest.main()
