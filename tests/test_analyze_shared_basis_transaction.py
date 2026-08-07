import unittest
from pathlib import Path

from tools.analyze_shared_basis_transaction import analyze


class SharedBasisTransactionTests(unittest.TestCase):
    def test_transaction_identities_and_nonlinear_compute_rejection(self):
        result = analyze(
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0115/analysis-001/manifest.json"),
            Path("/Users/chad/Models/mimo-prismwing/evidence/PW-0116/analysis-001/manifest.json"),
            Path("evals/fixtures/tiny/pw0117-shared-basis-algebra.json"),
        )
        self.assertLess(result["fixture_result"]["gate_up_maximum_absolute_error"], 1e-12)
        self.assertLess(result["fixture_result"]["down_mixture_maximum_absolute_error"], 1e-12)
        self.assertTrue(result["published_nonlinear_form_rejected_for_transaction_compute"])
        self.assertTrue(result["identity_basis_forms_remain_physically_eligible"])
        self.assertTrue(all(row["transaction_linear_ratio"] <= 0.5 for row in result["configurations"]))
        self.assertTrue(all(row["published_nonlinear_lower_bound_ratio"] > 0.5 for row in result["configurations"]))
        self.assertIsNone(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
