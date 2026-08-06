import unittest

from tools.analyze_routed_layer_artifact import median, stage_totals


class RoutedLayerArtifactAnalysisTests(unittest.TestCase):
    def test_median_requires_three_trials(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        with self.assertRaises(ValueError):
            median([1.0, 2.0])

    def test_stage_totals_preserve_wait_as_gpu_superset(self):
        trial = {
            "trusted_tensor_bind_ms": 1.0,
            "final_release_ms": 2.0,
            "expert_tomography": [
                {
                    "tensor_lookup_validation_ms": 3.0,
                    "matrix_transient_release_ms": 4.0,
                    "dynamic_input_ms": 5.0,
                    "dynamic_hidden_ms": 6.0,
                    "gate_up_sparse_repair_ms": 7.0,
                    "down_sparse_repair_ms": 8.0,
                    "swiglu_ms": 9.0,
                    "projections": [
                        {
                            "source_buffer_install_ms": 10.0,
                            "synchronous_wait_ms": 11.0,
                            "gpu_interval_ms": 2.5,
                        }
                    ],
                }
            ],
        }
        totals = stage_totals(trial)
        self.assertEqual(totals["dynamic_fp8_ms"], 11.0)
        self.assertEqual(totals["sparse_repair_ms"], 15.0)
        self.assertEqual(totals["synchronous_wait_ms"], 11.0)
        self.assertEqual(totals["gpu_interval_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()
