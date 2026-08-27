import unittest

from tools.analyze_pw0327_target_bonus_q8_pilot import (
    route_metrics,
    safety_gate,
    validate_byte_ledgers,
    validate_proposal_traces,
)


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


def proposal_traces():
    result = []
    for _ in range(7):
        step = traces()
        for row in step[1:]:
            row["selected_experts_by_position"] = [row["selected_experts_by_position"][0]]
            row["route_weights_by_position"] = [row["route_weights_by_position"][0]]
            row["U"] = 8.0
        result.append(step)
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
            "process_physical_footprint_bytes": 10,
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

    def test_safety_fail_closed_on_every_normative_memory_limit(self):
        baseline = {
            "protected_service_pids": {"WindowServer": [1]},
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "release_boundary": False,
            "system_memory_free_percent": 80,
            "process_physical_footprint_bytes": 10,
            "process_peak_resident_bytes": 10,
        }
        released = {**baseline, "release_boundary": True}
        for field, value in (
            ("system_memory_free_percent", 9),
            ("process_physical_footprint_bytes", 8 * 1024**3 + 1),
            ("process_peak_resident_bytes", 8 * 1024**3 + 1),
        ):
            changed = {**baseline, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Gate 8"):
                safety_gate([baseline, changed, released])
        too_large_after_release = {
            **released,
            "process_physical_footprint_bytes": 4 * 1024**3 + 1,
        }
        with self.assertRaisesRegex(ValueError, "Gate 8"):
            safety_gate([baseline, too_large_after_release])

    def test_proposal_trace_authenticates_seven_single_token_steps(self):
        validate_proposal_traces(proposal_traces())
        changed = proposal_traces()
        changed.pop()
        with self.assertRaisesRegex(ValueError, "seven steps"):
            validate_proposal_traces(changed)
        changed = proposal_traces()
        changed[2][1]["selected_experts_by_position"].append(list(range(8)))
        with self.assertRaisesRegex(ValueError, "route row count"):
            validate_proposal_traces(changed)

    def test_byte_ledgers_require_positive_ordered_progress_bound_values(self):
        transaction = {"logical_source_bytes": 10, "process_disk_bytes_read": 12}
        report = {"logical_source_bytes": 20, "process_disk_bytes_read": 24}
        progress = {"process_disk_bytes_read": 12}
        self.assertEqual(
            validate_byte_ledgers(transaction, report, progress, category="code")[
                "transaction_process_disk_bytes_read"
            ],
            12,
        )
        for changed_transaction, changed_report, changed_progress, message in (
            ({**transaction, "logical_source_bytes": -1}, report, progress, "logical byte"),
            ({**transaction, "logical_source_bytes": 21}, report, progress, "byte-ledger order"),
            (transaction, report, {"process_disk_bytes_read": 13}, "progress physical"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_byte_ledgers(
                    changed_transaction,
                    changed_report,
                    changed_progress,
                    category="code",
                )


if __name__ == "__main__":
    unittest.main()
