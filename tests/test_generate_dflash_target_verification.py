import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.generate_dflash_target_verification import (
    PW0206_DFLASH_TARGET_SHA256,
    PW0206_JACOBI_SECOND_SHA256,
    PW0206_PROPOSED_BLOCK,
    apply_rope_positions,
    jacobi_successor_block,
    kv_ledger,
    selected_proposed_block,
)


class TargetVerificationTests(unittest.TestCase):
    def test_corrected_dflash_proposal_is_selected_without_legacy_jacobi(self):
        arguments = argparse.Namespace(
            corrected_decode="corrected.json",
            jacobi_second_iteration=False,
            jacobi_third_iteration=False,
            prior_target_manifest=None,
        )
        proposed, evidence_class, identities = selected_proposed_block(arguments)
        self.assertEqual(proposed, PW0206_PROPOSED_BLOCK)
        self.assertEqual(
            evidence_class, "pw0206_corrected_qkv_dflash_block_verification"
        )
        self.assertEqual(identities, {})

        arguments.jacobi_third_iteration = True
        with self.assertRaisesRegex(ValueError, "second-iteration hash mismatch"):
            selected_proposed_block(arguments)

    def test_corrected_jacobi_second_iteration_shifts_authenticated_posterior(self):
        posterior = [0, 2585, 2585, 2585, 2585, 2585, 2585, 2585]
        prior = {
            "status": "passed",
            "evidence_class": "pw0206_corrected_qkv_dflash_block_verification",
            "proposed_block_token_ids": PW0206_PROPOSED_BLOCK,
            "target_posterior_token_ids": posterior,
            "greedy_verification": {"accepted_length_a": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text(json.dumps(prior))
            arguments = argparse.Namespace(
                corrected_decode="corrected.json",
                jacobi_second_iteration=True,
                jacobi_third_iteration=False,
                prior_target_manifest=path,
            )
            with patch(
                "tools.generate_dflash_target_verification.sha256_file",
                return_value=PW0206_DFLASH_TARGET_SHA256,
            ):
                proposed, evidence_class, identities = selected_proposed_block(arguments)
        self.assertEqual(proposed, [9707, 0, 2585, 2585, 2585, 2585, 2585, 2585])
        self.assertEqual(
            evidence_class, "pw0206_corrected_qkv_jacobi_second_iteration"
        )
        self.assertEqual(
            identities["pw0206_dflash_target_manifest_sha256"],
            PW0206_DFLASH_TARGET_SHA256,
        )

    def test_corrected_jacobi_third_iteration_shifts_second_posterior(self):
        proposal = [9707, 0, 2585, 2585, 2585, 2585, 2585, 2585]
        posterior = [0, 2585, 646, 646, 2585, 2585, 2585, 2585]
        prior = {
            "status": "passed",
            "evidence_class": "pw0206_corrected_qkv_jacobi_second_iteration",
            "proposed_block_token_ids": proposal,
            "target_posterior_token_ids": posterior,
            "greedy_verification": {"accepted_length_a": 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text(json.dumps(prior))
            arguments = argparse.Namespace(
                corrected_decode="corrected.json",
                jacobi_second_iteration=False,
                jacobi_third_iteration=True,
                prior_target_manifest=path,
            )
            with patch(
                "tools.generate_dflash_target_verification.sha256_file",
                return_value=PW0206_JACOBI_SECOND_SHA256,
            ):
                proposed, evidence_class, identities = selected_proposed_block(arguments)
        self.assertEqual(proposed, [9707, 0, 2585, 646, 646, 2585, 2585, 2585])
        self.assertEqual(
            evidence_class, "pw0206_corrected_qkv_jacobi_third_iteration"
        )
        self.assertEqual(
            identities["pw0206_jacobi_second_manifest_sha256"],
            PW0206_JACOBI_SECOND_SHA256,
        )

    def test_jacobi_successor_preserves_anchor_and_shifts_posterior(self):
        proposed = [264, 10, 11, 12, 13, 14, 15, 16]
        posterior = [20, 21, 22, 23, 24, 25, 26, 27]
        self.assertEqual(
            jacobi_successor_block(proposed, posterior),
            [264, 20, 21, 22, 23, 24, 25, 26],
        )
        proposed[0] = 9707
        self.assertEqual(
            jacobi_successor_block(proposed, posterior),
            [9707, 20, 21, 22, 23, 24, 25, 26],
        )

    def test_block_rope_preserves_unrotated_tail_and_uses_absolute_positions(self):
        values = torch.arange(2 * 3 * 192, dtype=torch.float32).reshape(2, 3, 192).to(torch.bfloat16)
        actual = apply_rope_positions(values, 10000.0, 0)
        self.assertTrue(torch.equal(actual[0], values[0]))
        self.assertTrue(torch.equal(actual[:, :, 64:], values[:, :, 64:]))
        self.assertFalse(torch.equal(actual[1, :, :64], values[1, :, :64]))

    def test_kv_ledger_increases_for_prefix_visibility(self):
        no_prefix = kv_ledger(8)
        with_prefix = kv_ledger(8, 27)
        self.assertEqual(no_prefix["cache_write_bytes"], with_prefix["cache_write_bytes"])
        self.assertGreater(
            with_prefix["attention_cache_read_bytes"], no_prefix["attention_cache_read_bytes"]
        )


if __name__ == "__main__":
    unittest.main()
