import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bounded_routed_row_oracle", ROOT / "tools/generate_bounded_routed_row_oracle.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BoundedRoutedRowOracleTests(unittest.TestCase):
    def test_authorities_are_pinned(self):
        self.assertEqual(len(MODULE.INPUT_SHA256), 64)
        self.assertEqual(len(MODULE.MANIFEST_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
