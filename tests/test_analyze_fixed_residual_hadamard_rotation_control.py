import tempfile
import unittest
from pathlib import Path

from tools.analyze_fixed_residual_hadamard_rotation_control import analyze


class FixedResidualRotationAnalysisTests(unittest.TestCase):
    def test_rejects_unbound_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
