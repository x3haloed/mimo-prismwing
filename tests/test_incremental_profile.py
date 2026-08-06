import unittest

from tools.profile_incremental_decode import run_profile, semantic_projection


class IncrementalProfileTests(unittest.TestCase):
    def test_semantic_projection_excludes_only_timing(self):
        report = {
            "schema_version": 2,
            "semantic": "endpoint",
            "revision": "revision",
            "fixture_sha256": "fixture",
            "checkpoint_verification_sha256": "checkpoint",
            "prompt_token_ids": [1],
            "generated_token_ids": [2],
            "generated_text": "x",
            "batch_size": 1,
            "concurrency": 1,
            "accepted_tokens": 1,
            "A": 1,
            "exactness": "exact",
            "implementation": "test",
            "steps": [{
                "input_token_id": 1,
                "input_token_ids": [1],
                "output_token_id": 2,
                "output_token_text": "x",
                "top_logits": [[2, 1.0]],
                "full_logits": [1.0],
                "layer_traces": [{"layer": 0, "cache_length": 1, "U": 0.0, "wall_ms": 1.0}],
            }],
        }
        changed = {**report, "steps": [{**report["steps"][0], "layer_traces": [
            {**report["steps"][0]["layer_traces"][0], "wall_ms": 99.0}
        ]}]}
        self.assertEqual(semantic_projection(report), semantic_projection(changed))
        changed["steps"][0]["layer_traces"][0]["cache_length"] = 2
        self.assertNotEqual(semantic_projection(report), semantic_projection(changed))

    def test_profile_partitions_routed_work(self):
        traces = [{"layer": 0, "attention": "full", "wall_ms": 2.0}]
        traces.extend({"layer": layer,
                       "attention": "full" if layer % 6 == 5 else "sliding_window_128",
                       "wall_ms": 3.0} for layer in range(1, 48))
        report = {"complete_wall_ms": 200.0,
                  "steps": [{"wall_ms": 40.0}, {"wall_ms": 150.0, "layer_traces": traces}],
                  "ledger": {"actual_process_disk_bytes_read": 123}}
        result = run_profile(report, 17_207_905_152, 25_171_968)
        self.assertEqual(result["incremental_expert_source_bytes"], 9_464_659_968)
        self.assertEqual(result["incremental_expert_fp8_matrices_expanded"], 1128)
        self.assertAlmostEqual(result["routed_layer_wall_ms"], 141.0)


if __name__ == "__main__":
    unittest.main()
