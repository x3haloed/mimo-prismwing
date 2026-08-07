import unittest

from tools.run_r720_cpu_arithmetic_ceiling import ceiling, matrix_macs, peak_flops


class R720CpuArithmeticCeilingTests(unittest.TestCase):
    def test_shape_mac_and_impossible_peak_algebra(self):
        self.assertEqual(matrix_macs((2048, 4096)), 8_388_608)
        peak = peak_flops(2, 10, 3.60, 16)
        self.assertEqual(peak, 1_152_000_000_000.0)
        report = ceiling(peak, 14_820_573_184)
        self.assertAlmostEqual(report["impossible_maximum_tps"], 38.86489360761285)
        self.assertFalse(report["targets"]["50.0"]["arithmetically_possible_at_impossible_peak"])
        self.assertTrue(report["targets"]["34.3"]["arithmetically_possible_at_impossible_peak"])
        self.assertGreater(report["targets"]["34.3"]["required_fraction_of_impossible_peak"], 0.88)

    def test_invalid_shapes_and_candidate_specifications_fail_closed(self):
        for shape in ((1,), (0, 2), (1, 2, 3)):
            with self.assertRaises(ValueError):
                matrix_macs(shape)
        for arguments in ((0, 10, 3.6, 16), (2, -1, 3.6, 16), (2, 10, 0, 16)):
            with self.assertRaises(ValueError):
                peak_flops(*arguments)
        with self.assertRaises(ValueError):
            ceiling(0, 1)


if __name__ == "__main__":
    unittest.main()
