import unittest

try:
    import mlx.core as mx
    import numpy as np

    from tools.mlx_full_expert_benchmark import full_expert, quantize
    from tools.mlx_moe_block_benchmark import route_schedule
except ModuleNotFoundError:
    mx = None


@unittest.skipIf(mx is None, "MLX is unavailable")
class FullExpertTests(unittest.TestCase):
    def test_route_schedule_preserves_token_and_slot_authority(self):
        schedule = route_schedule(np.array([[7, 3], [3, 9]], dtype=np.int32))
        self.assertEqual(schedule[3], ([0, 1], [1, 0]))
        self.assertEqual(schedule[7], ([0], [0]))
        self.assertEqual(schedule[9], ([1], [1]))

    def test_quantized_full_expert_matches_dequantized_composition(self):
        width = 128
        hidden = 128
        input_values = np.array(
            [[np.sin(index / 11.0) * 0.05 for index in range(width)]],
            dtype=np.float16,
        )
        weights = []
        for offset in (3, 17, 29):
            weights.append(
                np.array(
                    [
                        [np.cos((row * width + column + offset) / 31.0) * 0.03 for column in range(width)]
                        for row in range(hidden)
                    ],
                    dtype=np.float16,
                )
            )
        arrays = [quantize(__import__("torch").from_numpy(weight)) for weight in weights]
        output = full_expert(mx.array(input_values), *arrays)
        mx.eval(output)

        dequantized = [
            mx.dequantize(
                packed,
                scales,
                biases,
                group_size=128,
                bits=4,
                mode="affine",
            )
            for packed, scales, biases in arrays
        ]
        gate = mx.matmul(mx.array(input_values), dequantized[0].T)
        up = mx.matmul(mx.array(input_values), dequantized[1].T)
        expected = mx.matmul(mx.sigmoid(gate) * gate * up, dequantized[2].T)
        mx.eval(expected)
        self.assertLess(float(mx.max(mx.abs(output - expected)).item()), 1e-5)


if __name__ == "__main__":
    unittest.main()
