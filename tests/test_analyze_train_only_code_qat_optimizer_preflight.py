from pathlib import Path
import tempfile
import unittest

from tools.analyze_train_only_code_qat_optimizer_preflight import analyze


class CodeQatOptimizerPreflightAnalysisTests(unittest.TestCase):
    def test_rejects_unbound_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze(source)


if __name__ == "__main__":
    unittest.main()

