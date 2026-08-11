import unittest

from tools.analyze_layer_major_prefill_roofline import (
    ARENA_LIMIT_BYTES,
    analyze,
    kv_cache_bytes,
    model_length,
    transient_arena_bytes,
)


class LayerMajorPrefillRooflineTests(unittest.TestCase):
    def test_exact_f32_kv_cache_accounting(self):
        expected_per_token = (9 * 4 + 39 * 8) * (192 + 128) * 4
        self.assertEqual(kv_cache_bytes(8_000), 8_000 * expected_per_token)

    def test_arenas_scale_linearly_and_fit(self):
        small = sum(transient_arena_bytes(128).values())
        large = sum(transient_arena_bytes(8_000).values())
        self.assertEqual(large, small * 62.5)
        self.assertLess(kv_cache_bytes(8_000) + large, ARENA_LIMIT_BYTES)

    def test_record_count_increases_source_floor(self):
        low = model_length(1_024, 1_000, "test")
        high = model_length(1_024, 2_000, "test")
        self.assertGreater(
            high["uncached_acquisition_floor_ms"], low["uncached_acquisition_floor_ms"]
        )

    def test_frozen_gates_separate_speedup_from_ttft(self):
        report = analyze()
        self.assertTrue(report["gates"]["arena_gate_passed"])
        self.assertTrue(report["gates"]["four_x_roofline_gate_passed"])
        self.assertFalse(report["gates"]["fifteen_second_source_acquisition_gate_passed"])
        self.assertTrue(report["gates"]["authorize_width128_layer_slice"])


if __name__ == "__main__":
    unittest.main()
