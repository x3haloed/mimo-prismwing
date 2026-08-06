import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "generate_staged_metal_expert_oracle",
    TOOLS / "generate_staged_metal_expert_oracle.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StagedMetalExpertOracleTests(unittest.TestCase):
    def test_authority_constants_are_pinned(self):
        self.assertEqual(MODULE.PREFIX, "model.layers.43.mlp.experts.32")
        self.assertEqual(len(MODULE.VERIFICATION_SHA256), 64)
        self.assertEqual(len(MODULE.REVISION), 40)


if __name__ == "__main__":
    unittest.main()
