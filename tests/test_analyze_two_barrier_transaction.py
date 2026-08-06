import unittest

from tools.analyze_two_barrier_transaction import (
    candidate_stage_totals,
    median,
    serial_stage_totals,
)


class TwoBarrierTransactionAnalysisTests(unittest.TestCase):
    def test_median_requires_three_trials(self):
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        with self.assertRaises(ValueError):
            median([1.0, 2.0])

    def test_candidate_stage_totals_preserve_dynamic_hidden_as_subset(self):
        trial = {
            "weighted_scatter_ms": 10.0,
            "transaction": {
                "wall_ms": 1.0,
                "dynamic_input_ms": 2.0,
                "gate_up_cpu_stage_ms": 3.0,
                "dynamic_hidden_ms": 4.0,
                "down_cpu_stage_ms": 5.0,
                "phases": [
                    {
                        "source_buffer_bind_ms": 6.0,
                        "small_buffer_install_ms": 7.0,
                        "synchronous_wait_ms": 8.0,
                        "gpu_interval_ms": 2.5,
                    }
                ],
            },
        }
        totals = candidate_stage_totals(trial)
        self.assertEqual(totals["gate_up_cpu_stage_ms"], 3.0)
        self.assertEqual(totals["dynamic_hidden_ms_subset"], 4.0)
        self.assertEqual(totals["synchronous_wait_ms"], 8.0)

    def test_serial_stage_totals_sum_all_projection_waits(self):
        trial = {
            "serial_expert_tomography": [
                {
                    "projections": [
                        {
                            "synchronous_wait_ms": 11.0,
                            "gpu_interval_ms": 2.0,
                            "source_buffer_install_ms": 1.0,
                        },
                        {
                            "synchronous_wait_ms": 12.0,
                            "gpu_interval_ms": 3.0,
                            "source_buffer_install_ms": 2.0,
                        },
                    ]
                }
            ]
        }
        totals = serial_stage_totals(trial)
        self.assertEqual(totals["synchronous_wait_ms"], 23.0)
        self.assertEqual(totals["gpu_interval_ms"], 5.0)
        self.assertEqual(totals["source_buffer_bind_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
