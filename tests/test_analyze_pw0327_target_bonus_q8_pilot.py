import unittest

from tools.analyze_pw0327_target_bonus_q8_pilot import route_metrics, safety_gate


def traces():
    result = [
        {
            "layer": 0,
            "selected_experts_by_position": [],
            "route_weights_by_position": [],
            "U": 0.0,
        }
    ]
    for layer in range(1, 48):
        selected = [list(range(8)) for _ in range(8)]
        result.append(
            {
                "layer": layer,
                "selected_experts_by_position": selected,
                "route_weights_by_position": [[0.125] * 8 for _ in range(8)],
                "U": 1.0,
            }
        )
    return result


class Pw0327TargetBonusQ8PilotTests(unittest.TestCase):
    def test_route_metrics_rederive_layer_qualified_identity_union(self):
        result = route_metrics(traces())
        self.assertEqual(result["U"], 1.0)
        self.assertEqual(result["unique_identities"], 47 * 8)
        self.assertEqual(len(result["identities"]), 47 * 8)

    def test_route_metrics_fail_closed_on_weight_or_reported_u_change(self):
        changed = traces()
        changed[1]["route_weights_by_position"][0][0] = 0.5
        with self.assertRaisesRegex(ValueError, "route weight"):
            route_metrics(changed)
        changed = traces()
        changed[20]["U"] = 2.0
        with self.assertRaisesRegex(ValueError, "layer U"):
            route_metrics(changed)

    def test_safety_allows_pid_replacement_but_not_service_loss(self):
        baseline = {
            "protected_service_pids": {"WindowServer": [1], "nxnode": [2]},
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "release_boundary": False,
            "system_memory_free_percent": 80,
            "process_peak_resident_bytes": 10,
        }
        replaced = {
            **baseline,
            "protected_service_pids": {"WindowServer": [1], "nxnode": [3]},
            "release_boundary": True,
            "system_memory_free_percent": 79,
            "process_peak_resident_bytes": 20,
        }
        self.assertEqual(safety_gate([baseline, replaced])["minimum_system_memory_free_percent"], 79)
        replaced["protected_service_pids"]["nxnode"] = []
        with self.assertRaisesRegex(ValueError, "Gate 8"):
            safety_gate([baseline, replaced])


if __name__ == "__main__":
    unittest.main()
