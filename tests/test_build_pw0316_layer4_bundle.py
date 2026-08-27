from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.build_pw0316_layer4_bundle import (
    ALIGNMENT,
    ROUTE,
    align,
    append_file,
    digest,
    replay_source_expert,
)


class Pw0316Layer4BundleTests(unittest.TestCase):
    def test_declared_route_is_four_k4_then_four_source(self):
        self.assertEqual(ROUTE, (96, 64, 232, 31, 88, 245, 223, 151))
        self.assertEqual(len(set(ROUTE)), 8)

    def test_payload_append_is_aligned_and_authority_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            payload = b"prismwing"
            path.write_bytes(payload)
            stream = BytesIO(b"prefix")
            stream.seek(0, 2)
            record = append_file(
                stream,
                path,
                {"bytes": len(payload), "sha256": digest(payload)},
            )
            self.assertEqual(record["offset"], ALIGNMENT)
            self.assertEqual(stream.getvalue()[ALIGNMENT:], payload)
            self.assertEqual(record["alignment"], ALIGNMENT)

    def test_align_does_not_add_a_page_at_an_exact_boundary(self):
        stream = BytesIO(bytes(ALIGNMENT))
        stream.seek(0, 2)
        self.assertEqual(align(stream), ALIGNMENT)
        self.assertEqual(len(stream.getvalue()), ALIGNMENT)

    def test_source_replay_preserves_expert_major_batch_before_row_selection(self):
        class Panel:
            def __init__(self):
                self.observed = None

            def complete_outputs(self, values, _weights):
                self.observed = values.copy()
                return {"candidate_output_bf16_f32": values + np.float32(1.0)}

        panel = Panel()
        values = np.arange(20, dtype=np.float32).reshape(5, 4)
        positions = np.asarray([3, 1, 4], dtype=np.int64)
        expected = values[positions] + np.float32(1.0)
        actual = replay_source_expert(panel, values, positions, {}, expected)
        np.testing.assert_array_equal(panel.observed, values[positions])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
