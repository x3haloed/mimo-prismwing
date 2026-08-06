import json
from pathlib import Path
import unittest


class TextEndpointFixtureTests(unittest.TestCase):
    def test_pw0050_walking_fixture_is_exact_and_bounded(self):
        fixture = json.loads(
            Path("evals/fixtures/real/pw0050-text-endpoint.json").read_text()
        )
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(
            fixture["semantic"],
            "mimo_v2_5_target_faithful_raw_text_incremental_decode",
        )
        self.assertEqual(
            fixture["revision"], "63651580ca774f8504f676040460aed3e1244ac1"
        )
        self.assertEqual(fixture["prompt_utf8"], "Hello")
        self.assertEqual(fixture["expected_prompt_token_ids"], [9707])
        self.assertEqual(
            fixture["full_attention_qkv_scale_layout"],
            {
                "weight_shape": [13568, 4096],
                "scale_shape": [108, 32],
                "query_rows": 12288,
                "query_scale_rows": 96,
                "key_heads": 4,
                "key_rows_per_head": 192,
                "key_scale_rows_per_head": 2,
                "key_scale_row_start": 96,
                "value_heads": 4,
                "value_rows_per_head": 128,
                "value_scale_row_start": 104,
            },
        )
        self.assertEqual(fixture["decode"]["new_tokens"], 2)
        self.assertTrue(fixture["decode"]["use_kv_cache"])
        self.assertEqual(fixture["decode"]["batch_size"], 1)
        self.assertEqual(fixture["decode"]["concurrency"], 1)
        self.assertEqual(fixture["safety"]["minimum_system_memory_free_percent"], 20)
        self.assertEqual(
            fixture["safety"]["maximum_process_physical_footprint_bytes"],
            8 * 1024**3,
        )
        self.assertEqual(
            fixture["safety"]["maximum_post_phase_physical_footprint_bytes"],
            4 * 1024**3,
        )
        self.assertEqual(fixture["safety"]["maximum_swap_growth_bytes"], 512 * 1024**2)
        self.assertEqual(fixture["safety"]["maximum_new_throttled_pages"], 0)
        self.assertTrue(fixture["safety"]["require_malloc_pressure_relief"])
        self.assertEqual(
            fixture["safety"]["protect_resident_services"],
            ["ChatGPT", "WindowServer", "nxnode", "syncthing"],
        )


if __name__ == "__main__":
    unittest.main()
