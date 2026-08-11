import json
from pathlib import Path
import unittest

import mlx.core as mx
import numpy as np

from tools.run_microscaling_fp4_real_expert import quantize_projection, quantized_linear


class MicroscalingFp4RealExpertTest(unittest.TestCase):
    def test_tiny_quantized_matmul_matches_explicit_dequantization(self):
        fixture = json.loads((Path(__file__).parents[1] / "evals/fixtures/tiny/pw0182-microscaling-fp4.json").read_text())
        weight = np.asarray(fixture["weight"], dtype=np.float16)
        values = np.asarray(fixture["input"], dtype=np.float16)
        for mode, group_size in (("mxfp4", 32), ("nvfp4", 16)):
            config = {"mode": mode, "group_size": group_size, "bits": 4}
            first = quantize_projection(weight, config); second = quantize_projection(weight, config)
            for left, right in zip(first[:2], second[:2]): np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
            actual = quantized_linear(mx.array(values), first, config)
            dense = mx.dequantize(first[0], first[1], biases=first[2], group_size=group_size, bits=4, mode=mode, dtype=mx.float16)
            expected = mx.array(values) @ dense.T
            mx.eval(actual, expected)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3)


if __name__ == "__main__": unittest.main()
