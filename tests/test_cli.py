import contextlib
import io
import unittest

from cpulmforge.cli import main


class CliTests(unittest.TestCase):
    def test_infinite_memory_limit_is_clean_input_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["examples/samples.jsonl", "--memory-gib", "inf"])
        self.assertEqual(code, 2)
        self.assertIn("memory_gib must be a positive finite number", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
