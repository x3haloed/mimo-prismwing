import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.analyze_pw0331_k4_rank1 import (
    ANALYTICAL_ALPHA_MIN_ABSOLUTE_TOLERANCE,
    AUTHENTICATED_ANALYTICAL_ALPHA_MIN,
    FROZEN_ATTENUATION_FLOOR,
    MAXIMUM_RELATIVE_L2,
    MAXIMUM_ROW_RELATIVE_L2,
    analysis_slices,
    authenticate_error_direction_diagnostic,
    authenticate_historical_dense_zero_control,
    error_direction_diagnostic,
    require_repeated_fits,
    sliced_gate,
    stage_a_gate,
    verify_fit_array_bindings,
)
from tools.openrouter_reference import canonical_json
from tools.reproduce_pw0311_k4_expert import sha256_file
from tools.run_pw0331_k4_rank1_fit import (
    CONTRACT_COMMIT,
    CONTRACT_GIT_BLOB,
    CONTRACT_SHA256,
    CORPUS_SHA256,
    CHECKPOINT_EXPERT96_SHARD_SHA256,
    CHECKPOINT_INDEX_SHA256,
    CHECKPOINT_RECEIPT_SHA256,
    CHECKPOINT_REVISION,
    EXPECTED_COUNTS,
    EXPERIMENT_ID,
    IMPLEMENTATION_HASHES,
    PANEL_CONTRACT_SHA256,
    PANEL_EXPORT_SHA256,
    PANEL_IMPLEMENTATION_SHA256,
    PW0315_SUMMARY_SHA256,
    PW0316_REJECTION_SHA256,
    PW0318_BUNDLE_SHA256,
    PW0318_MANIFEST_SHA256,
    RED_LINES_SHA256,
    SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256,
    SERIALIZED_DENSE_CONTROL_GIT_BLOB,
    SERIALIZED_DENSE_CONTROL_SHA256,
    SERIALIZED_DENSE_CONTROL_STAGES_SHA256,
    SEMANTIC,
    STAGE_A_BASE_SEMANTIC,
    TARGET_SHA256,
    array_sha256,
    bf16,
    deterministic_tree,
    schema2_layout_ledger,
    stage_a_numerics_authority,
)

CONTROL_FIXTURE = json.loads(
    (
        Path(__file__).parents[1]
        / "evals/fixtures/tiny/pw0331-serialized-dense-control.json"
    ).read_text()
)


def passing_slices(relative: float = 0.001, maximum: float = 0.002):
    return {
        name: {
            "relative_l2": relative,
            "maximum_row_relative_l2": maximum,
        }
        for name in ("overall", "fit", "validation", "pilot")
    }


class Pw0331AnalysisTests(unittest.TestCase):
    def _rewrite_authority(self, root: Path, mutate) -> None:
        authority_path = root / "fit-authority.json"
        authority = json.loads(authority_path.read_text())
        mutate(authority)
        authority_path.write_bytes(canonical_json(authority))
        report_path = root / "construction.json"
        report = json.loads(report_path.read_text())
        report["fit_authority_sha256"] = sha256_file(authority_path)
        report["deterministic_tree"] = deterministic_tree(root)
        report_path.write_bytes(canonical_json(report))

    def _fit_run(
        self,
        root: Path,
        *,
        commit: str,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        process_receipt: dict | None = None,
    ) -> Path:
        root.mkdir()
        left = np.ones((4096, 1), dtype="<f2") if left is None else left.astype("<f2")
        right = np.ones((1, 2048), dtype="<f2") if right is None else right.astype("<f2")
        left_path = root / "correction-left.f16le"
        right_path = root / "correction-right.f16le"
        left_path.write_bytes(left.tobytes())
        right_path.write_bytes(right.tobytes())
        factors = {
            "correction_left": {
                "file": left_path.name,
                "dtype": "<f2",
                "shape": [4096, 1],
                "bytes": left_path.stat().st_size,
                "sha256": sha256_file(left_path),
            },
            "correction_right": {
                "file": right_path.name,
                "dtype": "<f2",
                "shape": [1, 2048],
                "bytes": right_path.stat().st_size,
                "sha256": sha256_file(right_path),
            },
        }
        authority = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "semantic": SEMANTIC,
            "exactness_class": "L3_modified_expert_weights",
            "commit": commit,
            "construction_surface": "fit_rows_only_no_primary_validation_or_pilot_payload_access",
            "execution_authority": {
                "contract_commit": CONTRACT_COMMIT,
                "contract_git_blob": CONTRACT_GIT_BLOB,
                "contract_sha256": CONTRACT_SHA256,
                "serialized_dense_control_git_blob": SERIALIZED_DENSE_CONTROL_GIT_BLOB,
                "serialized_dense_control_sha256": SERIALIZED_DENSE_CONTROL_SHA256,
                "target_sha256": TARGET_SHA256,
                "red_lines_sha256": RED_LINES_SHA256,
                "unchanged_implementation_sha256": IMPLEMENTATION_HASHES,
            },
            "metadata_authority": {
                "pw0315_summary_sha256": PW0315_SUMMARY_SHA256,
                "pw0316_rejection_sha256": PW0316_REJECTION_SHA256,
                "published_position1_route_relative_l2": 0.010988841869031155,
                "published_position1_final_relative_l2": 0.0027743952049186665,
                "held_out_payloads_opened": False,
            },
            "panel_authority": {
                "contract_sha256": PANEL_CONTRACT_SHA256,
                "export_sha256": PANEL_EXPORT_SHA256,
                "implementation_sha256": PANEL_IMPLEMENTATION_SHA256,
            },
            "corpus_authority": {
                "corpus_manifest_sha256": CORPUS_SHA256,
                "whole_capture_payloads_rescanned": False,
                "split_counts": EXPECTED_COUNTS,
                "input_read": {
                    "whole_payload_rescanned": False,
                    "selected_rows": [0, *range(2, 109)],
                },
                "source_read": {
                    "whole_payload_rescanned": False,
                    "selected_rows": list(range(108)),
                },
            },
            "fit_positions": [0, *range(2, 109)],
            "fit_source_offsets": list(range(108)),
            "fit_slots": [0] * 108,
            "array_sha256": {
                name: "0" * 64
                for name in (
                    "fit_input_f32",
                    "fit_source_bf16_f32",
                    "fit_route_weights_f32",
                    "candidate_dynamic_hidden_f32",
                    "candidate_down_base_raw_f32",
                    "candidate_down_base_bf16_f32",
                )
            },
            "tlut_authority": {
                "manifest_sha256": PW0318_MANIFEST_SHA256,
                "bundle_sha256": PW0318_BUNDLE_SHA256,
                "tlut_sha256": "1" * 64,
                "tlut_offset": 0,
                "tlut_bytes": 4096,
            },
            "k4_authority": {
                name: {
                    "manifest_sha256": "2" * 64,
                    "candidate_array_sha256": "3" * 64,
                    "rank": 1,
                    "row_scale_identity": True,
                    "correction_zero": True,
                }
                for name in ("gate", "up", "down")
            }
            | {
                "checkpoint": {
                    "revision": CHECKPOINT_REVISION,
                    "receipt_sha256": CHECKPOINT_RECEIPT_SHA256,
                    "index_sha256": CHECKPOINT_INDEX_SHA256,
                    "expert96_shard_sha256_from_receipt": (
                        CHECKPOINT_EXPERT96_SHARD_SHA256
                    ),
                    "source_shard_rescanned": False,
                }
            },
            "fit": {
                "fit_rows": 108,
                "input_columns": 2048,
                "output_rows": 4096,
                "rcond": 1e-12,
            },
            "serialized_dense_control": {
                "semantic": CONTROL_FIXTURE["semantic"],
                "fixture_sha256": SERIALIZED_DENSE_CONTROL_SHA256,
                "diagnostic_sha256": SERIALIZED_DENSE_CONTROL_DIAGNOSTIC_SHA256,
                "stages_sha256": SERIALIZED_DENSE_CONTROL_STAGES_SHA256,
                "independent_process_replays": 2,
                "fit_rows": 108,
                "held_out_payloads_opened": False,
                "stages": CONTROL_FIXTURE["diagnostic"]["stages"],
                "pass": True,
            },
            "numerics": stage_a_numerics_authority(),
            "factors": factors,
            "layout": schema2_layout_ledger(4, 4),
            "accepted_tokens": 0,
            "A": 0,
            "U": 0,
            "performance_claim": None,
        }
        authority_path = root / "fit-authority.json"
        authority_path.write_bytes(canonical_json(authority))
        report = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "fit_factors_frozen_without_heldout_access",
            "commit": commit,
            "accepted_tokens": 0,
            "A": 0,
            "U": 0,
            "performance_claim": None,
            "fit_authority_sha256": sha256_file(authority_path),
            "deterministic_tree": deterministic_tree(root),
            "process_receipt": process_receipt
            or {
                "pid": 1001 if root.name == "first" else 1002,
                "started_ns": 1,
                "nonce": hashlib.sha256(str(root).encode()).hexdigest(),
            },
        }
        (root / "construction.json").write_bytes(canonical_json(report))
        return root

    @staticmethod
    def _real_process_receipt() -> dict:
        script = (
            "import json,os,secrets,time;"
            "print(json.dumps({'pid':os.getpid(),'started_ns':time.time_ns(),"
            "'nonce':secrets.token_hex(32)}))"
        )
        return json.loads(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )

    def test_analysis_slices_exclude_primary_from_fit(self):
        slices = analysis_slices()
        self.assertEqual(len(slices["fit"]), 111)
        self.assertNotIn(1, slices["fit"])
        self.assertEqual(len(slices["validation"]), 56)
        self.assertEqual(len(slices["pilot"]), 56)

    def test_historical_dense_zero_control_is_distinct_and_bit_exact(self):
        stored = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        historical = {"candidate_output_bf16_f32": stored.copy()}
        result = authenticate_historical_dense_zero_control(historical, stored)
        self.assertTrue(result["bit_identical"])
        self.assertEqual(
            result["candidate_output_raw_sha256"],
            result["stored_output_raw_sha256"],
        )
        historical["candidate_output_bf16_f32"][0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(ValueError, "historical dense"):
            authenticate_historical_dense_zero_control(historical, stored)
        with self.assertRaisesRegex(ValueError, "historical dense"):
            authenticate_historical_dense_zero_control(
                {"candidate_output_bf16_f32": stored.astype(np.float64)}, stored
            )

    def test_error_direction_uses_smaller_nonnegative_root_and_is_diagnostic(self):
        report = error_direction_diagnostic(
            np.asarray([100.0, 0.0], dtype=np.float32),
            np.asarray([102.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([4.0, 0.0], dtype=np.float32),
            np.asarray([2.0, 0.0], dtype=np.float32),
            1.0,
        )
        self.assertEqual(report["alpha_min"], 0.25)
        self.assertEqual(report["applied_minimum_attenuation"], 0.25)
        self.assertEqual(report["observed_attenuation"], 0.5)
        self.assertTrue(report["attenuation_requirement_pass"])
        self.assertEqual(
            report["gate_role"],
            "subordinate_sanity_condition_after_sliced_fidelity_gates",
        )
        with self.assertRaisesRegex(ValueError, "degenerate"):
            error_direction_diagnostic(
                np.asarray([1.0], dtype=np.float32),
                np.asarray([1.1], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                1.0,
            )

    def test_error_direction_never_weakens_frozen_attenuation_floor(self):
        report = error_direction_diagnostic(
            np.asarray([200.0, 0.0], dtype=np.float32),
            np.asarray([202.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([4.0, 0.0], dtype=np.float32),
            np.asarray([3.5, 0.0], dtype=np.float32),
            1.0,
        )
        self.assertEqual(report["alpha_min"], 0.0)
        self.assertEqual(
            report["applied_minimum_attenuation"], FROZEN_ATTENUATION_FLOOR
        )
        self.assertEqual(report["observed_attenuation"], 0.125)
        self.assertFalse(report["attenuation_requirement_pass"])

    def test_real_error_direction_authority_fails_closed_without_weakening(self):
        report = {
            "semantic": "position1_f64_root_with_conservative_frozen_floor_v2",
            "alpha_min": AUTHENTICATED_ANALYTICAL_ALPHA_MIN,
            "authenticated_analytical_alpha_min": (
                AUTHENTICATED_ANALYTICAL_ALPHA_MIN
            ),
            "analytical_alpha_min_absolute_tolerance": (
                ANALYTICAL_ALPHA_MIN_ABSOLUTE_TOLERANCE
            ),
            "frozen_attenuation_floor": FROZEN_ATTENUATION_FLOOR,
            "frozen_floor_is_conservative": True,
            "applied_minimum_attenuation": FROZEN_ATTENUATION_FLOOR,
            "observed_attenuation": 0.5,
            "attenuation_requirement_pass": True,
            "gate_role": "subordinate_sanity_condition_after_sliced_fidelity_gates",
        }
        authenticate_error_direction_diagnostic(report)
        report["alpha_min"] += 1e-6
        with self.assertRaisesRegex(ValueError, "authority mismatch"):
            authenticate_error_direction_diagnostic(report)
        report["alpha_min"] = AUTHENTICATED_ANALYTICAL_ALPHA_MIN
        report["frozen_floor_is_conservative"] = False
        with self.assertRaisesRegex(ValueError, "authority mismatch"):
            authenticate_error_direction_diagnostic(report)

    def test_gate_is_exclusive_and_checks_both_identity_and_cumulative(self):
        passing = passing_slices()
        result = stage_a_gate(
            passing,
            passing,
            passing,
            passing,
            {"relative_l2": 0.009},
            {"relative_l2": 0.009},
            True,
        )
        self.assertTrue(result["pass"])
        boundary = passing_slices()
        boundary["validation"] = {
            "relative_l2": MAXIMUM_RELATIVE_L2,
            "maximum_row_relative_l2": 0.001,
        }
        self.assertFalse(sliced_gate(boundary))
        row_boundary = passing_slices()
        row_boundary["pilot"] = {
            "relative_l2": 0.001,
            "maximum_row_relative_l2": MAXIMUM_ROW_RELATIVE_L2,
        }
        self.assertFalse(sliced_gate(row_boundary))
        result = stage_a_gate(
            passing,
            passing,
            boundary,
            passing,
            {"relative_l2": 0.009},
            {"relative_l2": 0.009},
            True,
        )
        self.assertFalse(result["cumulative_route"])
        self.assertFalse(result["pass"])

    def test_primary_boundary_and_nonfinite_values_fail(self):
        passing = passing_slices()
        result = stage_a_gate(
            passing,
            passing,
            passing,
            passing,
            {"relative_l2": 0.01},
            {"relative_l2": 0.001},
            True,
        )
        self.assertFalse(result["pass"])
        result = stage_a_gate(
            passing,
            passing,
            passing,
            passing,
            {"relative_l2": 0.001},
            {"relative_l2": 0.001},
            False,
        )
        self.assertTrue(result["sliced_and_primary_gates_pass"])
        self.assertFalse(result["attenuation_sanity"])
        self.assertFalse(result["pass"])
        passing["overall"]["relative_l2"] = float("nan")
        self.assertFalse(sliced_gate(passing))

    def test_all_six_fit_arrays_are_rebound_to_analysis_authorities(self):
        selected_input = np.arange(12, dtype=np.float32).reshape(3, 4)
        expert_down = np.arange(24, dtype=np.float32).reshape(6, 4)
        route_weights = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
        source_offsets = np.asarray([4, 2, 5], dtype=np.int64)
        fit_local = np.asarray([0, 2], dtype=np.int64)
        dynamic_hidden = np.arange(18, dtype=np.float32).reshape(3, 6)
        base_raw = (np.arange(12, dtype=np.float32).reshape(3, 4) + 0.125)
        values = {
            "fit_input_f32": selected_input[fit_local],
            "fit_source_bf16_f32": expert_down[source_offsets[fit_local]],
            "fit_route_weights_f32": route_weights[fit_local],
            "candidate_dynamic_hidden_f32": dynamic_hidden[fit_local],
            "candidate_down_base_raw_f32": base_raw[fit_local],
            "candidate_down_base_bf16_f32": bf16(base_raw[fit_local]),
        }
        hashes = {name: array_sha256(value) for name, value in values.items()}
        arguments = {
            "fit_hashes": hashes,
            "selected_input": selected_input,
            "expert_down": expert_down,
            "route_weights": route_weights,
            "source_offsets": source_offsets,
            "fit_local": fit_local,
            "dynamic_hidden": dynamic_hidden,
            "base_raw": base_raw,
        }
        verify_fit_array_bindings(**arguments)
        mutations = {
            "fit_input_f32": ("selected_input", (0, 0)),
            "fit_source_bf16_f32": ("expert_down", (4, 0)),
            "fit_route_weights_f32": ("route_weights", (0,)),
            "candidate_dynamic_hidden_f32": ("dynamic_hidden", (0, 0)),
            "candidate_down_base_raw_f32": ("base_raw", (0, 0)),
        }
        for expected_name, (argument_name, index) in mutations.items():
            with self.subTest(array=expected_name):
                changed = {
                    name: value.copy() if isinstance(value, np.ndarray) else value
                    for name, value in arguments.items()
                }
                changed[argument_name][index] += np.float32(1.0)
                with self.assertRaisesRegex(ValueError, expected_name):
                    verify_fit_array_bindings(**changed)
        wrong_bf16_hash = dict(hashes)
        wrong_bf16_hash["candidate_down_base_bf16_f32"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate_down_base_bf16_f32"):
            verify_fit_array_bindings(**(arguments | {"fit_hashes": wrong_bf16_hash}))

    def test_two_fresh_fit_roots_are_required_and_tamper_fails_closed(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._fit_run(
                root / "first",
                commit=commit,
                process_receipt=self._real_process_receipt(),
            )
            second = self._fit_run(
                root / "second",
                commit=commit,
                process_receipt=self._real_process_receipt(),
            )
            authority, left, right, repeat = require_repeated_fits(
                [first, second], commit
            )
            self.assertEqual(authority["semantic"], SEMANTIC)
            self.assertEqual(left.shape, (4096, 1))
            self.assertEqual(right.shape, (1, 2048))
            self.assertTrue(repeat["fresh_process_repeat"])
            with self.assertRaisesRegex(ValueError, "two distinct"):
                require_repeated_fits([first], commit)
            first_report = json.loads((first / "construction.json").read_text())
            second_report_path = second / "construction.json"
            second_report = json.loads(second_report_path.read_text())
            original_receipt = second_report["process_receipt"]
            second_report["process_receipt"] = first_report["process_receipt"]
            second_report_path.write_bytes(canonical_json(second_report))
            with self.assertRaisesRegex(ValueError, "process receipts"):
                require_repeated_fits([first, second], commit)
            second_report["process_receipt"] = original_receipt
            second_report_path.write_bytes(canonical_json(second_report))
            (second / "correction-right.f16le").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "factor payload"):
                require_repeated_fits([first, second], commit)

    def test_fresh_process_factor_drift_fails_before_analysis(self):
        commit = "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._fit_run(root / "first", commit=commit)
            changed = np.ones((4096, 1), dtype=np.float16)
            changed[0, 0] = np.float16(2.0)
            second = self._fit_run(root / "second", commit=commit, left=changed)
            with self.assertRaisesRegex(ValueError, "fresh-process fit mismatch"):
                require_repeated_fits([first, second], commit)

    def test_wrong_semantic_rank_dtype_and_zero_factor_fail_closed(self):
        commit = "c" * 40
        mutations = (
            (
                lambda authority: authority.__setitem__("semantic", "wrong"),
                "fit authority",
            ),
            (
                lambda authority: authority["factors"]["correction_left"].__setitem__(
                    "shape", [4096, 2]
                ),
                "factor record",
            ),
            (
                lambda authority: authority["factors"]["correction_right"].__setitem__(
                    "dtype", ">f2"
                ),
                "factor record",
            ),
            (
                lambda authority: authority["factors"]["correction_left"].__setitem__(
                    "file", "../escaped.f16le"
                ),
                "escapes fit root",
            ),
            (
                lambda authority: authority["numerics"].pop("fit_algebra"),
                "fit authority",
            ),
            (
                lambda authority: authority["serialized_dense_control"][
                    "stages"
                ]["candidate_output_bf16_f32"].__setitem__(
                    "bit_mismatches", 225
                ),
                "fit authority",
            ),
            (
                lambda authority: authority.__setitem__("A", 1),
                "fit authority",
            ),
            (
                lambda authority: authority.__setitem__("U", 1),
                "fit authority",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first = self._fit_run(root / "first", commit=commit)
                second = self._fit_run(root / "second", commit=commit)
                self._rewrite_authority(first, mutate)
                self._rewrite_authority(second, mutate)
                with self.assertRaisesRegex(ValueError, message):
                    require_repeated_fits([first, second], commit)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero = np.zeros((4096, 1), dtype=np.float16)
            first = self._fit_run(root / "first", commit=commit, left=zero)
            second = self._fit_run(root / "second", commit=commit, left=zero)
            with self.assertRaisesRegex(ValueError, "zero or nonfinite"):
                require_repeated_fits([first, second], commit)


if __name__ == "__main__":
    unittest.main()
