import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.host_safety import HostSafetyViolation
from tools.run_pw0330_cyclic_mtp_prefix import (
    BANDWIDTH_EXACT,
    BANDWIDTH_FAVORABLE,
    CONTRACT_COMMIT,
    HIDDEN,
    PW0327_PREFIX_COUNTS,
    S_FIXED,
    S_LM_HEAD,
    S_MTP_LAYER,
    S_MTP_ONLY,
    authenticate_execution_contract,
    disposition,
    execute_prefix,
    input_ids_identity,
    load_prefill_hidden,
    main,
    mtp_inventory_authority,
    prefix_route_metrics,
    prepare_output,
    storage_ceiling,
    strict_hash,
    target_token_rank,
    tensor_identity,
    validate_target_spine,
)


def full_match_transaction():
    return {
        "index": 0,
        "proposal_token_ids": list(range(8)),
        "posterior_token_ids": list(range(1, 9)),
        "verifier_authorized_token_ids": list(range(1, 9)),
        "emitted_token_ids": list(range(1, 9)),
        "verifier_retained_proposal_rows": 8,
        "retained_proposal_rows": 8,
        "proposal_converged": True,
    }


def route_traces():
    traces = [{
        "layer": 0,
        "selected_experts_by_position": [],
        "route_weights_by_position": [],
        "U": 0.0,
    }]
    for layer in range(1, 48):
        selected = [[position * 8 + offset for offset in range(8)] for position in range(8)]
        weights = [[0.125] * 8 for _position in range(8)]
        traces.append({
            "layer": layer,
            "selected_experts_by_position": selected,
            "route_weights_by_position": weights,
            "U": 8.0,
        })
    return traces


def fake_runner(proposals, *, mutate_hidden=False, mutate_input=False, error=None):
    calls = []

    def run(head, layer, hidden, input_ids, _target):
        calls.append({"head": head, "layer": layer, "input_ids": list(input_ids)})
        if error is not None:
            raise error
        if mutate_hidden:
            hidden[0, 0] = hidden[0, 0] + 1
        if mutate_input:
            input_ids.append(99)
        return {
            "layer": layer,
            "input_token_ids": calls[-1]["input_ids"],
            "input_token_ids_sha256": input_ids_identity(calls[-1]["input_ids"]),
            "proposal_token_id": proposals[head],
            "logits_sha256": f"logits-{head}",
        }

    return run, calls


class Pw0330SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.zeros((3, 4), dtype=torch.bfloat16)
        self.input_ids = [10, 11, 12]

    def execute(self, proposals, targets, **kwargs):
        runner, calls = fake_runner(proposals, **kwargs)
        result = execute_prefix(
            target_hidden=self.hidden.clone(),
            initial_input_ids=self.input_ids,
            target_tokens=targets,
            run_head=runner,
        )
        return result, calls

    def test_exact_layer_order_and_rotations(self):
        targets = [20, 21, 22, 23, 24, 25, 26, 27]
        result, calls = self.execute(targets, targets)
        self.assertEqual([row["layer"] for row in calls], [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(calls[0]["input_ids"], [10, 11, 12])
        self.assertEqual(calls[1]["input_ids"], [11, 12, 20])
        self.assertEqual(calls[3]["input_ids"], [20, 21, 22])
        self.assertEqual(result["final_input_token_ids"], [25, 26, 27])
        self.assertTrue(result["head_results"][3]["first_new_scheduler_behavior"])
        self.assertTrue(result["prefix_authority_exhausted"])
        self.assertIsNone(result["A"])

    def test_mismatch_at_heads_zero_three_and_seven(self):
        targets = list(range(100, 108))
        for mismatch in (0, 3, 7):
            with self.subTest(mismatch=mismatch):
                proposals = list(targets)
                proposals[mismatch] = 999
                result, calls = self.execute(proposals, targets)
                self.assertEqual(result["first_mismatch_index"], mismatch)
                self.assertEqual(result["A"], mismatch + 1)
                self.assertEqual(result["authenticated_prefix_matches"], mismatch)
                self.assertEqual(len(calls), mismatch + 1)

    def test_eight_matches_exhaust_authority_without_inferred_A(self):
        tokens = list(range(40, 48))
        result, calls = self.execute(tokens, tokens)
        self.assertEqual(len(calls), 8)
        self.assertIsNone(result["first_mismatch_index"])
        self.assertIsNone(result["A"])
        self.assertEqual(result["authenticated_prefix_matches"], 8)
        self.assertEqual(disposition(None, None), "prefix_authority_exhausted")

    def test_arbitrary_suffix_cannot_change_earlier_causal_inputs(self):
        proposals = [20, 21, 22, 23, 24, 25, 26, 27]
        first_targets = [20, 21, 99, 1, 2, 3, 4, 5]
        second_targets = [20, 21, 99, 150, 151, 152, 153, 154]
        first, first_calls = self.execute(proposals, first_targets)
        second, second_calls = self.execute(proposals, second_targets)
        self.assertEqual(first["A"], 3)
        self.assertEqual(second["A"], 3)
        self.assertEqual(first_calls, second_calls)

    def test_hidden_and_input_are_immutable(self):
        targets = list(range(20, 28))
        with self.assertRaisesRegex(ValueError, "immutable target hidden"):
            self.execute(targets, targets, mutate_hidden=True)
        with self.assertRaisesRegex(ValueError, "mutated its input"):
            self.execute(targets, targets, mutate_input=True)

    def test_trained_q4_reproduction_is_a_hard_gate(self):
        targets = list(range(20, 28))
        runner, _calls = fake_runner(targets)
        expected = [
            {"proposal_token_id": token, "logits_sha256": f"logits-{index}"}
            for index, token in enumerate(targets[:3])
        ]
        execute_prefix(
            target_hidden=self.hidden,
            initial_input_ids=self.input_ids,
            target_tokens=targets,
            run_head=runner,
            q4_expected=expected,
        )
        expected[2]["logits_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "trained q4 reproduction"):
            execute_prefix(
                target_hidden=self.hidden,
                initial_input_ids=self.input_ids,
                target_tokens=targets,
                run_head=fake_runner(targets)[0],
                q4_expected=expected,
            )

    def test_safety_failure_propagates_without_a_disposition(self):
        runner, _calls = fake_runner(
            list(range(20, 28)), error=HostSafetyViolation("fixture safety stop")
        )
        with self.assertRaisesRegex(HostSafetyViolation, "fixture safety stop"):
            execute_prefix(
                target_hidden=self.hidden,
                initial_input_ids=self.input_ids,
                target_tokens=list(range(20, 28)),
                run_head=runner,
            )


class Pw0330AuthorityTests(unittest.TestCase):
    def test_execution_contract_hashes_and_ancestry_are_fail_closed(self):
        with patch(
            "tools.run_pw0330_cyclic_mtp_prefix.strict_hash",
            side_effect=lambda _path, expected, _label: expected,
        ) as strict, patch(
            "tools.run_pw0330_cyclic_mtp_prefix.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as command:
            result = authenticate_execution_contract(Path("/repo"), "a" * 40)
        self.assertEqual(strict.call_count, 3)
        self.assertEqual(result["contract_commit"], CONTRACT_COMMIT)
        self.assertIn("--is-ancestor", command.call_args.args[0])

        with patch(
            "tools.run_pw0330_cyclic_mtp_prefix.strict_hash",
            side_effect=lambda _path, expected, _label: expected,
        ), patch(
            "tools.run_pw0330_cyclic_mtp_prefix.subprocess.run",
            return_value=SimpleNamespace(returncode=1),
        ):
            with self.assertRaisesRegex(ValueError, "does not descend"):
                authenticate_execution_contract(Path("/repo"), "b" * 40)

    def test_full_unclipped_target_spine_is_required(self):
        result = validate_target_spine(full_match_transaction())
        self.assertEqual(result["A"], 8)
        self.assertEqual(result["verifier_retained_proposal_rows"], 8)

        mismatch = full_match_transaction()
        mismatch.update({
            "posterior_token_ids": [99, 2, 3, 4, 5, 6, 7, 8],
            "verifier_authorized_token_ids": [99],
            "emitted_token_ids": [99],
            "verifier_retained_proposal_rows": 1,
            "retained_proposal_rows": 1,
            "proposal_converged": False,
        })
        with self.assertRaisesRegex(ValueError, "did not converge"):
            validate_target_spine(mismatch)

        clipped = full_match_transaction()
        clipped["emitted_token_ids"] = [1]
        clipped["retained_proposal_rows"] = 1
        with self.assertRaisesRegex(ValueError, "nonterminal|output-clipped"):
            validate_target_spine(clipped)

        retained = full_match_transaction()
        retained["verifier_retained_proposal_rows"] = 7
        with self.assertRaisesRegex(ValueError, "retention|retained"):
            validate_target_spine(retained)

    def test_route_prefix_counts_layer_qualified_identities(self):
        traces = route_traces()
        first = prefix_route_metrics(traces, 1)
        third = prefix_route_metrics(traces, 3)
        self.assertEqual(first["N_A"], 47 * 8)
        self.assertEqual(third["N_A"], 47 * 24)
        self.assertNotEqual(first["identity_sha256"], third["identity_sha256"])
        self.assertEqual(first["per_layer"][0]["experts"], list(range(8)))

    def test_route_rows_fail_closed(self):
        duplicate = route_traces()
        duplicate[1]["selected_experts_by_position"][0][-1] = 0
        with self.assertRaisesRegex(ValueError, "expert route row"):
            prefix_route_metrics(duplicate, 1)

        unnormalized = route_traces()
        unnormalized[1]["route_weights_by_position"][0] = [0.1] * 8
        with self.assertRaisesRegex(ValueError, "route weight"):
            prefix_route_metrics(unnormalized, 1)

        wrong_order = route_traces()
        wrong_order[2]["layer"] = 3
        with self.assertRaisesRegex(ValueError, "route layer order"):
            prefix_route_metrics(wrong_order, 1)

    def test_prefill_loader_rejects_hash_size_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden.f32"
            values = np.zeros((2, HIDDEN), dtype="<f4")
            path.write_bytes(values.tobytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hidden, identity = load_prefill_hidden(path, 2, expected_sha256=digest)
            self.assertEqual(tuple(hidden.shape), (2, HIDDEN))
            self.assertEqual(identity, tensor_identity(hidden))

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_prefill_hidden(path, 2, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "byte size"):
                load_prefill_hidden(path, 3, expected_sha256=digest)

            values[0, 0] = np.inf
            path.write_bytes(values.tobytes())
            nonfinite_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_prefill_hidden(path, 2, expected_sha256=nonfinite_digest)

    def test_strict_hash_rejects_wrong_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text("{}\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(strict_hash(path, digest, "fixture"), digest)
            with self.assertRaisesRegex(ValueError, "fixture SHA-256"):
                strict_hash(path, "0" * 64, "fixture")

    def test_clean_commit_precedes_output_creation_and_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-001"
            with patch(
                "tools.run_pw0330_cyclic_mtp_prefix.verify_clean_commit"
            ) as verify:
                prepare_output(root, "a" * 40, output)
                verify.assert_called_once_with(root, "a" * 40)
            self.assertTrue(output.is_dir())
            with patch("tools.run_pw0330_cyclic_mtp_prefix.verify_clean_commit"):
                with self.assertRaises(FileExistsError):
                    prepare_output(root, "a" * 40, output)

    def test_existing_output_is_not_modified_by_failure_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-001"
            output.mkdir()
            argv = [
                "run_pw0330_cyclic_mtp_prefix.py",
                "--repo", directory,
                "--commit", "a" * 40,
                "--checkpoint", directory,
                "--verification", str(Path(directory) / "receipt.json"),
                "--source-root", directory,
                "--output", str(output),
            ]
            with patch("sys.argv", argv), patch(
                "tools.run_pw0330_cyclic_mtp_prefix.run",
                side_effect=FileExistsError(output),
            ):
                self.assertEqual(main(), 1)
            self.assertEqual(list(output.iterdir()), [])


class Pw0330AccountingTests(unittest.TestCase):
    def test_mtp_inventory_excludes_lm_head_and_closes_bytes(self):
        result = mtp_inventory_authority()
        self.assertEqual(result["per_layer_logical_tensor_bytes"], [S_MTP_LAYER] * 3)
        self.assertEqual(result["mtp_only_logical_tensor_bytes"], S_MTP_ONLY)
        self.assertFalse(result["lm_head_included"])
        self.assertGreater(S_LM_HEAD, 0)

    def test_planning_byte_table_and_ceilings(self):
        expected_misses = [
            5_512_395_520,
            11_956_419_328,
            16_990_812_928,
            22_100_722_432,
            27_160_288_000,
            31_061_943_040,
            34_737_050_368,
            40_274_883_328,
        ]
        expected_tps = [
            0.629572,
            0.580516,
            0.612763,
            0.628115,
            0.638883,
            0.670360,
            0.699344,
            0.689352,
        ]
        rows = [storage_ceiling(index, count) for index, count in enumerate(PW0327_PREFIX_COUNTS, 1)]
        self.assertEqual([row["miss_bytes"] for row in rows], expected_misses)
        self.assertEqual(
            [round(row["candidate_favorable_tps_ceiling"], 6) for row in rows],
            expected_tps,
        )
        self.assertEqual(
            rows[0]["logical_bytes_before_joint_residency"],
            S_FIXED + S_MTP_ONLY + PW0327_PREFIX_COUNTS[0] * 25_171_968,
        )

    def test_zero_miss_uses_json_null_not_infinity(self):
        result = storage_ceiling(1, 0)
        self.assertEqual(result["miss_bytes"], 0)
        self.assertTrue(result["unbounded_storage_only_ceiling"])
        self.assertIsNone(result["candidate_favorable_tps_ceiling"])
        json.dumps(result, allow_nan=False)

    def test_exact_and_rounded_bandwidths_remain_distinct(self):
        self.assertEqual(BANDWIDTH_FAVORABLE, 3_470_448_309.677419)
        self.assertEqual(BANDWIDTH_EXACT, 3_470_425_919.832775)
        self.assertGreater(BANDWIDTH_FAVORABLE, BANDWIDTH_EXACT)

    def test_disposition_precedence_and_one_tps_boundary(self):
        self.assertEqual(
            disposition(3, {"candidate_favorable_at_or_below_one": True}),
            "conditional_hard_storage_rejection",
        )
        self.assertEqual(
            disposition(3, {"candidate_favorable_at_or_below_one": False}),
            "analytical_only_direct_q32_trace_required",
        )
        self.assertEqual(disposition(None, None), "prefix_authority_exhausted")

    def test_target_rank_has_deterministic_token_id_ties(self):
        logits = torch.tensor([5.0, 5.0, 4.0], dtype=torch.float32)
        self.assertEqual(target_token_rank(logits, 0), 1)
        self.assertEqual(target_token_rank(logits, 1), 2)
        self.assertEqual(target_token_rank(logits, 2), 3)


if __name__ == "__main__":
    unittest.main()
    authenticate_execution_contract,
