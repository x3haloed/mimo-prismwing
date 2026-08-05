import json
from pathlib import Path
import unittest

from tools.generate_tiny_moe_fixture import generate


class TinyMoeFixtureTests(unittest.TestCase):
    def test_committed_fixture_is_reproducible(self):
        path = Path("evals/fixtures/tiny/moe-noaux-tc-swiglu.json")
        self.assertEqual(json.loads(path.read_text()), generate())


if __name__ == "__main__":
    unittest.main()
