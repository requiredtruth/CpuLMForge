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

if __name__ == "__main__":
    unittest.main()
