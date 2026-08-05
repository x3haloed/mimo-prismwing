import json
from pathlib import Path
import tempfile
import unittest

from tools.openrouter_reference import (
    atomic_write_new,
    canonical_json,
    sha256_bytes,
    validate_capture_request,
    verify,
)


def valid_request():
    return {
        "model": "xiaomi/mimo-v2.5",
        "messages": [{"role": "user", "content": "Return only: prismwing"}],
        "temperature": 0,
        "max_tokens": 8,
        "reasoning": {"enabled": False},
        "logprobs": True,
        "top_logprobs": 20,
        "stream": False,
        "provider": {
            "order": ["Parasail"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }


class ReferenceCaptureTests(unittest.TestCase):
    def test_policy_fails_closed(self):
        request = valid_request()
        request["provider"]["allow_fallbacks"] = True
        with self.assertRaisesRegex(ValueError, "allow_fallbacks"):
            validate_capture_request(request)

    def test_atomic_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value"
            atomic_write_new(path, b"first")
            with self.assertRaises(FileExistsError):
                atomic_write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_offline_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            request = valid_request()
            response = {
                "model": "xiaomi/mimo-v2.5",
                "provider": "Parasail",
                "choices": [{"logprobs": {"content": [{"top_logprobs": [{}] * 20}]}}],
            }
            request_bytes = canonical_json(request)
            response_bytes = canonical_json(response)
            (output / "request.json").write_bytes(request_bytes)
            (output / "response.json").write_bytes(response_bytes)
            manifest = {
                "schema_version": 1,
                "request_file": "request.json",
                "request_sha256": sha256_bytes(request_bytes),
                "response_file": "response.json",
                "response_sha256": sha256_bytes(response_bytes),
            }
            (output / "manifest.json").write_text(json.dumps(manifest))
            verify(output)
            (output / "response.json").write_text('{"choices":[],"model":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "response hash mismatch"):
                verify(output)


if __name__ == "__main__":
    unittest.main()
