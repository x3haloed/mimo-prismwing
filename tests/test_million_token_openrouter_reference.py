import json
from pathlib import Path
import tempfile
import unittest

from tools.million_token_openrouter_reference import (
    MODEL,
    PROVIDER,
    build_probe,
    needle_code,
    validate_endpoint_metadata,
    validate_response,
)


CHECKPOINT = Path("/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580")


class MillionTokenOpenRouterReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CHECKPOINT / "tokenizer.json").is_file():
            raise unittest.SkipTest("pinned tokenizer is not installed")

    def test_probe_has_exact_requested_count_and_edge_offsets(self):
        request, generation = build_probe(CHECKPOINT, target_prompt_tokens=1_000)
        self.assertEqual(generation["prompt_tokens"], 1_000)
        self.assertLess(generation["needle_token_offset"], 256)
        self.assertGreaterEqual(generation["question_token_offset"], 744)
        self.assertTrue(generation["decode_reencode_exact"])
        self.assertEqual(request["provider"]["order"], [PROVIDER])
        self.assertFalse(request["provider"]["allow_fallbacks"])
        self.assertEqual(request["top_logprobs"], 20)

    def test_endpoint_metadata_requires_parasail_logprobs_and_context(self):
        models = {
            "data": [{
                "id": MODEL,
                "context_length": 1_050_000,
                "supported_parameters": ["logprobs", "top_logprobs"],
            }]
        }
        endpoints = {
            "data": {
                "id": MODEL,
                "endpoints": [{
                    "provider_name": PROVIDER,
                    "context_length": 1_048_576,
                    "quantization": "fp8",
                    "tag": "parasail/fp8",
                    "pricing": {"prompt": "0.00000014"},
                    "status": 0,
                    "supported_parameters": ["logprobs", "top_logprobs"],
                }],
            }
        }
        summary = validate_endpoint_metadata(models, endpoints)
        self.assertEqual(summary["provider_context_length"], 1_048_576)
        endpoints["data"]["endpoints"][0]["context_length"] = 999_999
        with self.assertRaisesRegex(ValueError, "below one million"):
            validate_endpoint_metadata(models, endpoints)

    def test_response_gate_checks_exact_usage_answer_and_top20(self):
        code = needle_code()
        positions = [
            {
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": f"candidate-{index}", "logprob": -float(index + 1)}
                    for index in range(20)
                ],
            }
            for token in code
        ]
        response = {
            "model": MODEL,
            "provider": PROVIDER,
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": code},
                "logprobs": {"content": positions},
            }],
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": len(positions),
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
                "cost": 0.140004,
            },
        }
        summary = validate_response(response, {"needle_code": code})
        self.assertEqual(summary["minimum_top_logprobs"], 20)
        self.assertEqual(summary["prompt_tokens"], 1_000_000)
        response["usage"]["prompt_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "prompt-token count"):
            validate_response(response, {"needle_code": code})


if __name__ == "__main__":
    unittest.main()
