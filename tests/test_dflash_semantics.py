import unittest

import torch

from tools.dflash_semantics import (
    extract_context_feature,
    first_block_position_ids,
    initial_block_ids,
    install_greedy_draft_suffix,
    validate_first_block_cache_lengths,
    verify_greedy_block,
)


class DFlashSemanticsTests(unittest.TestCase):
    def test_hidden_selection_uses_output_state_offset_and_order(self):
        hidden = [torch.full((1, 3, 2), float(i)) for i in range(49)]
        actual = extract_context_feature(hidden, [0, 2, 47])
        self.assertEqual(tuple(actual.shape), (1, 3, 6))
        self.assertEqual(actual[0, 0].tolist(), [1.0, 1.0, 3.0, 3.0, 48.0, 48.0])
        hidden[3][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            extract_context_feature(hidden, [0, 2, 47])

    def test_first_block_mask_and_positions(self):
        block = initial_block_ids(264)
        self.assertEqual(block.tolist(), [[264] + [151675] * 7])
        positions = first_block_position_ids(27)
        self.assertEqual(tuple(positions.shape), (1, 35))
        self.assertEqual(positions[0, -8:].tolist(), list(range(27, 35)))

    def test_draft_argmax_installs_only_suffix(self):
        block = initial_block_ids(264, block_size=4, mask_token_id=99)
        logits = torch.tensor(
            [[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0], [0.0, 1.0, 4.0]]]
        )
        self.assertEqual(
            install_greedy_draft_suffix(block, logits).tolist(), [[264, 1, 0, 2]]
        )
        logits[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            install_greedy_draft_suffix(block, logits)

    def test_greedy_verification_matches_published_start_advance(self):
        proposed = torch.tensor([[264, 10, 11, 12, 13, 14, 15, 16]])
        posterior = torch.tensor([[10, 11, 99, 13, 14, 15, 16, 17]])
        result = verify_greedy_block(proposed, posterior)
        self.assertEqual(result.matching_draft_tokens, 2)
        self.assertEqual(result.accepted_length_a, 3)
        self.assertEqual(result.accepted_block_token_ids, (264, 10, 11))
        self.assertEqual(result.correction_token_id, 99)
        self.assertEqual(result.rejected_draft_token_ids, (12, 13, 14, 15, 16))

    def test_full_acceptance_is_bounded_by_eight(self):
        proposed = torch.tensor([[264, 10, 11, 12, 13, 14, 15, 16]])
        posterior = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        result = verify_greedy_block(proposed, posterior)
        self.assertEqual(result.accepted_length_a, 8)
        self.assertEqual(result.correction_token_id, 17)
        self.assertEqual(result.rejected_draft_token_ids, ())

    def test_cache_growth_fails_closed(self):
        validate_first_block_cache_lengths([35] * 5, 27)
        with self.assertRaisesRegex(ValueError, "five copies of 35"):
            validate_first_block_cache_lengths([35, 35, 34, 35, 35], 27)


if __name__ == "__main__":
    unittest.main()
