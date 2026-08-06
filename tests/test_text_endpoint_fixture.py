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
        self.assertEqual(fixture["decode"]["new_tokens"], 2)
        self.assertTrue(fixture["decode"]["use_kv_cache"])
        self.assertEqual(fixture["decode"]["batch_size"], 1)
        self.assertEqual(fixture["decode"]["concurrency"], 1)


if __name__ == "__main__":
    unittest.main()
