from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.openrouter_reference import canonical_json
from tools.run_train_only_threshold_crossing_code_qat import (
    LEARNING_RATE,
    PW0145_SHA256,
    run_threshold,
)


class ThresholdCrossingCodeQatTests(unittest.TestCase):
    def test_wrapper_binds_prior_and_exact_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            prior = Path(temporary) / "pw0145.json"
            prior.write_bytes(
                canonical_json(
                    {
                        "decision": "reject_tested_fixed_grid_code_qat_optimizer_family",
                        "validation_loaded": False,
                        "holdout_unsealed": False,
                    }
                )
            )
            with patch(
                "tools.run_train_only_threshold_crossing_code_qat.sha256_file",
                return_value=PW0145_SHA256,
            ), patch(
                "tools.run_train_only_threshold_crossing_code_qat.run",
                return_value={"decision": "ok"},
            ) as delegated:
                result = run_threshold(
                    Path("checkpoint"), Path("verification"), Path("corpus"),
                    Path("pw0139"), Path("pw0144"), prior, Path("output"), "a" * 40,
                )
            self.assertEqual(result["decision"], "ok")
            self.assertEqual(delegated.call_args.kwargs["learning_rates"], (LEARNING_RATE,))
            self.assertEqual(
                delegated.call_args.kwargs["evidence_class"],
                "pw0146_train_only_threshold_crossing_code_qat",
            )


if __name__ == "__main__":
    unittest.main()

