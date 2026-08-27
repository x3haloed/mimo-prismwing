from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from tools.build_pw0316_layer4_bundle import ALIGNMENT, ROUTE, align, append_file, digest


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


if __name__ == "__main__":
    unittest.main()
