import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.run_pw0331_k4_rank1_fit import (
    EXPECTED_COUNTS,
    _libc_fmaf,
    apply_serialized_rank_one,
    array_sha256,
    bf16,
    fit_rank_one,
    partition_fit_positions,
    pread_f32_rows,
    schema2_layout_ledger,
    serialized_k4_projection_base,
    stage_a_candidate_stages,
    unpack_serialized_k4_transformed,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "evals/fixtures/tiny/pw0331-rank1-repair.json"
)


class Pw0331RankOneFitTests(unittest.TestCase):
    @staticmethod
    def _manual_fwht(values):
        result = np.asarray(values, dtype=np.float32).copy()
        stride = 1
        while stride < result.size:
            for block in range(0, result.size, 2 * stride):
                for offset in range(stride):
                    left = result[block + offset]
                    right = result[block + stride + offset]
                    result[block + offset] = np.float32(left + right)
                    result[block + stride + offset] = np.float32(left - right)
            stride *= 2
        normalization = np.float32(
            np.float32(1.0) / np.sqrt(np.float32(result.size))
        )
        return np.asarray(result * normalization, dtype=np.float32)

    def test_stage_a_base_emulates_serialized_k4_kernel_order(self):
        fixture = json.loads(FIXTURE.read_text())["serialized_base"]
        rows, columns = fixture["rows"], fixture["columns"]
        packed = np.zeros((rows * columns // 256, 64), dtype="<u2")
        tlut = np.zeros((512, 2), dtype=np.float32)
        tlut[0] = fixture["tlut_index_zero_f32"]
        inputs = np.empty(columns, dtype=np.float32)
        inputs[:3] = fixture["input_prefix_f32"]
        for index in range(3, columns):
            inputs[index] = np.float32(((index * 7) % 19) - 9) * np.float32(
                0.03125
            )
        right_sign = np.asarray(
            [-1 if index % 3 == 0 else 1 for index in range(columns)],
            dtype=np.int8,
        )
        left_sign = np.asarray(
            [-1 if index % 2 else 1 for index in range(rows)], dtype=np.int8
        )
        projection = {
            "rows": rows,
            "columns": columns,
            "packed": packed,
            "left_sign": left_sign,
            "right_sign": right_sign,
            "global_scale": np.asarray(
                [fixture["global_scale_f32"]], dtype=np.float32
            ),
            "row_scale": np.full(
                rows, fixture["row_scale_f16"], dtype=np.float16
            ),
        }
        actual = serialized_k4_projection_base(inputs[None, :], projection, tlut)[0]
        transformed = unpack_serialized_k4_transformed(
            packed, tlut, rows, columns
        )
        transformed_input = self._manual_fwht(
            inputs * right_sign.astype(np.float32)
        )
        fused = _libc_fmaf()
        raw = np.empty(rows, dtype=np.float32)
        for output in range(rows):
            partials = np.zeros(64, dtype=np.float32)
            for lane in range(64):
                accumulator = 0.0
                for column in range(lane, columns, 64):
                    accumulator = fused(
                        transformed[output, column],
                        transformed_input[column],
                        accumulator,
                    )
                partials[lane] = accumulator
            for offset in (32, 16, 8, 4, 2, 1):
                for lane in range(offset):
                    partials[lane] = np.float32(
                        partials[lane] + partials[lane + offset]
                    )
            raw[output] = partials[0]
        transformed_output = self._manual_fwht(raw)
        scaled = np.asarray(
            transformed_output
            * left_sign.astype(np.float32)
            * np.float32(fixture["global_scale_f32"]),
            dtype=np.float32,
        )
        expected = np.asarray(
            [
                fused(value, fixture["row_scale_f16"], 0.0)
                for value in scaled
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(
            serialized_k4_projection_base(inputs[None, :], projection, tlut).view(
                np.uint32
            )[0],
            actual.view(np.uint32),
        )

    def test_serialized_stage_fit_rows_are_batch_shape_invariant(self):
        tlut = np.zeros((512, 2), dtype=np.float32)
        tlut[0] = [0.75, -0.5]

        def projection():
            return {
                "rows": 16,
                "columns": 16,
                "packed": np.zeros((1, 64), dtype="<u2"),
                "left_sign": np.ones(16, dtype=np.int8),
                "right_sign": np.ones(16, dtype=np.int8),
                "global_scale": np.asarray([0.5], dtype=np.float32),
                "row_scale": np.ones(16, dtype=np.float16),
            }

        values = np.asarray(
            [
                np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                np.linspace(0.25, -0.75, 16, dtype=np.float32),
                np.arange(16, dtype=np.float32) * np.float32(0.03125),
            ]
        )
        projections = {name: projection() for name in ("gate", "up", "down")}
        together = stage_a_candidate_stages(
            values, projections, tlut, lambda array: array.copy()
        )
        separately = [
            stage_a_candidate_stages(
                values[index : index + 1],
                projections,
                tlut,
                lambda array: array.copy(),
            )
            for index in range(len(values))
        ]
        for name, expected in together.items():
            actual = np.concatenate([row[name] for row in separately], axis=0)
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )

    def test_serialized_tile_unpack_matches_scalar_kernel_indexing(self):
        rows, columns = 16, 32
        rng = np.random.default_rng(331)
        packed = rng.integers(
            0, 1 << 16, size=(rows * columns // 256, 64), dtype=np.uint16
        ).astype("<u2")
        tlut = rng.standard_normal((512, 2)).astype(np.float32)
        actual = unpack_serialized_k4_transformed(packed, tlut, rows, columns)
        expected = np.empty((rows, columns), dtype=np.float32)

        def permuted_index(original_index):
            value = original_index
            e = value & 1
            value >>= 1
            d = value & 3
            value >>= 2
            c = value & 1
            value >>= 1
            b = value & 7
            value >>= 3
            a = value
            return ((((b * 4 + d) * 2 + c) * 2 + a) * 2 + e)

        for row in range(rows):
            for column in range(columns):
                tile = (row // 16) * (columns // 16) + column // 16
                original = (row & 15) * 16 + (column & 15)
                permuted = permuted_index(original)
                state_index = permuted >> 1
                bit_offset = state_index * 8
                word_index = bit_offset >> 4
                word_shift = bit_offset & 15
                joined = (
                    int(packed[tile, word_index]) << 16
                ) | int(packed[tile, (word_index + 1) & 63])
                state = (joined >> (16 - word_shift)) & 0xFFFF
                mixed = (state * (state + 1)) & 0xFFFFFFFF
                lookup = (mixed >> 6) & 511
                component = permuted & 1
                value = tlut[lookup, component]
                if component == 0 and ((mixed >> 15) & 1):
                    value = np.float32(-value)
                expected[row, column] = value
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_tiny_fit_has_runtime_orientation_and_recovers_rank_one_signal(self):
        fixture = json.loads(FIXTURE.read_text())["fit"]
        inputs = np.asarray(fixture["inputs_f32"], dtype=np.float32)
        base = np.asarray(fixture["base_raw_f32"], dtype=np.float32)
        source = np.asarray(fixture["source_bf16_f32"], dtype=np.float32)
        weights = np.asarray(fixture["route_weights_f32"], dtype=np.float32)
        left, right, report = fit_rank_one(inputs, base, source, weights)
        self.assertEqual(left.shape, (3, 1))
        self.assertEqual(right.shape, (1, 2))
        self.assertEqual(left.dtype, np.float16)
        self.assertEqual(right.dtype, np.float16)
        self.assertGreaterEqual(left[report["sign_pivot"], 0], 0.0)
        corrected = apply_serialized_rank_one(inputs, base, left, right)
        relative = np.linalg.norm(corrected - source) / np.linalg.norm(source)
        self.assertLess(relative, fixture["maximum_corrected_relative_l2"])

    def test_scalar_serialized_application_matches_tiny_fixture(self):
        fixture = json.loads(FIXTURE.read_text())["application"]
        actual = apply_serialized_rank_one(
            np.asarray(fixture["inputs_f32"], dtype=np.float32),
            np.asarray(fixture["base_raw_f32"], dtype=np.float32),
            np.asarray(fixture["correction_left_f16"], dtype=np.float16),
            np.asarray(fixture["correction_right_f16"], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            actual, np.asarray(fixture["expected_bf16_f32"], dtype=np.float32)
        )

    def test_scalar_application_rounds_correction_before_finish_add(self):
        actual = apply_serialized_rank_one(
            np.asarray([[-2.1997482776641846]], dtype=np.float32),
            np.asarray([[-18.752052307128906]], dtype=np.float32),
            np.asarray([[-7.36328125]], dtype=np.float16),
            np.asarray([[1.0]], dtype=np.float16),
        )
        self.assertEqual(int(actual[0, 0].view(np.uint32)), 0xC0240000)

    def test_zero_factors_reproduce_only_the_bf16_consumer_boundary(self):
        inputs = np.asarray([[1.0, -2.0]], dtype=np.float32)
        base = np.asarray([[1.001, -0.999]], dtype=np.float32)
        actual = apply_serialized_rank_one(
            inputs,
            base,
            np.zeros((2, 1), dtype=np.float16),
            np.zeros((1, 2), dtype=np.float16),
        )
        np.testing.assert_array_equal(actual, bf16(base))

    def test_exact_split_is_108_plus_primary_plus_validation_plus_pilot(self):
        positions = np.asarray(
            [position for position in range(177) if position not in {2, 4, 6}],
            dtype=np.int64,
        )
        split = partition_fit_positions(positions)
        self.assertEqual(
            {name: len(indices) for name, indices in split.items()}, EXPECTED_COUNTS
        )
        self.assertNotIn(1, positions[split["fit"]])
        np.testing.assert_array_equal(positions[split["primary"]], [1])

    def test_selected_pread_never_touches_or_depends_on_heldout_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.f32"
            values = np.arange(30, dtype="<f4").reshape(5, 6)
            path.write_bytes(values.tobytes())
            first, audit = pread_f32_rows(
                path, [0, 2], total_rows=5, columns=6
            )
            changed = values.copy()
            changed[[1, 3, 4]] *= np.float32(-1000.0)
            path.write_bytes(changed.astype("<f4").tobytes())
            second, second_audit = pread_f32_rows(
                path, [0, 2], total_rows=5, columns=6
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(audit["selected_bytes_sha256"], second_audit["selected_bytes_sha256"])
            self.assertFalse(audit["whole_payload_rescanned"])
            self.assertEqual(
                [(row["offset"], row["bytes"]) for row in audit["pread_ranges"]],
                [(0, 24), (48, 24)],
            )

    def test_heldout_target_mutation_cannot_change_frozen_fit_hashes(self):
        fixture = json.loads(FIXTURE.read_text())["fit"]
        fit_base = np.asarray(fixture["base_raw_f32"], dtype=np.float32)
        weights = np.asarray(fixture["route_weights_f32"], dtype=np.float32)
        selected = [0, 2, 4]
        full_inputs = np.full((6, 2), np.float32(19.0), dtype="<f4")
        full_source = np.full((6, 3), np.float32(-23.0), dtype="<f4")
        full_inputs[selected] = np.asarray(fixture["inputs_f32"], dtype=np.float32)
        full_source[selected] = np.asarray(
            fixture["source_bf16_f32"], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.f32"
            source_path = Path(directory) / "source.f32"
            input_path.write_bytes(full_inputs.tobytes())
            source_path.write_bytes(full_source.tobytes())
            first_input, _ = pread_f32_rows(
                input_path, selected, total_rows=6, columns=2
            )
            first_source, _ = pread_f32_rows(
                source_path, selected, total_rows=6, columns=3
            )
            first = fit_rank_one(first_input, fit_base, first_source, weights)[:2]
            full_inputs[[1, 3, 5]] *= np.float32(-999.0)
            full_source[[1, 3, 5]] *= np.float32(777.0)
            input_path.write_bytes(full_inputs.tobytes())
            source_path.write_bytes(full_source.tobytes())
            second_input, _ = pread_f32_rows(
                input_path, selected, total_rows=6, columns=2
            )
            second_source, _ = pread_f32_rows(
                source_path, selected, total_rows=6, columns=3
            )
            second = fit_rank_one(second_input, fit_base, second_source, weights)[:2]
            np.testing.assert_array_equal(first_input, second_input)
            np.testing.assert_array_equal(first_source, second_source)
            self.assertEqual(array_sha256(first[0]), array_sha256(second[0]))
            self.assertEqual(array_sha256(first[1]), array_sha256(second[1]))

    def test_invalid_fit_and_row_requests_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "projection authority"):
            serialized_k4_projection_base(
                np.ones((1, 16), dtype=np.float64),
                {
                    "rows": 16,
                    "columns": 16,
                    "packed": np.zeros((1, 64), dtype="<u2"),
                    "left_sign": np.ones(16, dtype=np.int8),
                    "right_sign": np.ones(16, dtype=np.int8),
                    "global_scale": np.ones(1, dtype=np.float32),
                    "row_scale": np.ones(16, dtype=np.float16),
                },
                np.zeros((512, 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "no rank-one signal"):
            fit_rank_one(
                np.eye(2, dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "fit matrices"):
            fit_rank_one(
                np.asarray([[np.nan]], dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
                np.ones(1, dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "serialized rank-one"):
            apply_serialized_rank_one(
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
                np.ones((1, 2), dtype=np.float16),
            )
        with self.assertRaisesRegex(ValueError, "serialized rank-one"):
            apply_serialized_rank_one(
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
                np.asarray([[np.inf]], dtype=np.float16),
                np.ones((1, 2), dtype=np.float16),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.f32"
            path.write_bytes(np.zeros((2, 2), dtype="<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "row request"):
                pread_f32_rows(path, [0, 0], total_rows=2, columns=2)

    def test_current_layout_is_byte_neutral_at_density_four(self):
        ledger = schema2_layout_ledger(4, 4)
        self.assertEqual(ledger["k4_logical_bytes"], 12_654_604)
        self.assertEqual(ledger["k4_stride_bytes"], 12_877_824)
        self.assertEqual(ledger["down_factor_bytes"], 12_288)
        self.assertEqual(ledger["bundle_bytes"], 152_387_584)
        self.assertEqual(
            np.zeros((4096, 1), dtype="<f2").nbytes
            + np.zeros((1, 2048), dtype="<f2").nbytes,
            ledger["down_factor_bytes"],
        )
        self.assertEqual(
            np.ones((4096, 1), dtype="<f2").nbytes
            + np.ones((1, 2048), dtype="<f2").nbytes,
            ledger["down_factor_bytes"],
        )
        self.assertEqual(ledger, schema2_layout_ledger(4, 4))
        with self.assertRaisesRegex(ValueError, "eight experts"):
            schema2_layout_ledger(4, 3)


if __name__ == "__main__":
    unittest.main()
