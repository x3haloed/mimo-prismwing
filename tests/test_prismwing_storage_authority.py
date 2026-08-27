import copy
from pathlib import Path
import unittest

from tools.prismwing_storage_authority import (
    BANDWIDTH_EXACT_BYTES_PER_SECOND,
    BANDWIDTH_FAVORABLE_BYTES_PER_SECOND,
    CHECKPOINT_VERIFICATION_PATH,
    FIXED_ALLOCATED_BYTES,
    FIXED_FP8_CODE_BYTES,
    FIXED_LOGICAL_BYTES,
    FIXED_MAX_OBJECT_BYTES,
    FIXED_NON_FP8_BYTES,
    FIXED_OBJECT_COUNT,
    PW0136_ANALYSIS_PATH,
    PW0136_RAW_PATH,
    PW0207_OFFLINE_PATH,
    authenticate_prismwing_storage,
    derive_pw0136_bandwidth,
    fixed_census_from_metadata,
    fixed_tensor_names,
    resident_allocation_bytes,
)


CHECKPOINT_ROOT = Path(
    "/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580"
)


def fixed_weight_map_fixture():
    result = {"model.embed_tokens.weight": "embed.safetensors"}
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        for suffix in (
            "input_layernorm.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.qkv_proj.weight_scale_inv",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
        ):
            result[f"{prefix}.{suffix}"] = "fixed.safetensors"
        if layer == 0:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                result[f"{prefix}.mlp.{projection}.weight"] = "fixed.safetensors"
                result[
                    f"{prefix}.mlp.{projection}.weight_scale_inv"
                ] = "fixed.safetensors"
        else:
            result[f"{prefix}.mlp.gate.weight"] = "fixed.safetensors"
            result[
                f"{prefix}.mlp.gate.e_score_correction_bias"
            ] = "fixed.safetensors"
    result["model.norm.weight"] = "fixed.safetensors"
    result["lm_head.weight"] = "fixed.safetensors"
    return result


def pw0136_raw_fixture():
    medians = {
        1: [60.0, 60.1, 60.2],
        2: [58.0, 58.125375, 58.2],
        4: [59.0, 59.1, 59.2],
        8: [61.0, 61.1, 61.2],
    }
    trials = []
    for state in ("cold", "warm"):
        for workers in (1, 2, 4, 8):
            for repetition, cold_wall in enumerate(medians[workers]):
                trials.append(
                    {
                        "cache_state": state,
                        "workers": workers,
                        "repetition": repetition,
                        "requested_bytes": 201_719_808,
                        "returned_bytes": 201_719_808,
                        "pread_calls": 8,
                        "slot_stream_sha256": (
                            "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
                        ),
                        "expert_reads": [
                            {
                                "expert": expert,
                                "slot": expert,
                                "source_offset": expert * 25_214_976,
                                "requested_bytes": 25_214_976,
                                "returned_bytes": 25_214_976,
                                "pread_calls": 1,
                                "wall_ms": 8.0,
                            }
                            for expert in range(8)
                        ],
                        "activity": {
                            "disk_bytes_read": 201_719_808 if state == "cold" else 0
                        },
                        "transfer_wall_ms": (
                            cold_wall if state == "cold" else cold_wall / 4.0
                        ),
                    }
                )
    services = {"WindowServer": [101], "launchservicesd": [102]}
    safety = [
        {
            "phase": "start",
            "system_memory_free_percent": 80,
            "process_peak_resident_bytes": 128 * 1024**2,
            "process_physical_footprint_bytes": 96 * 1024**2,
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "protected_service_pids": services,
        },
        {
            "phase": "buffer_release",
            "system_memory_free_percent": 79,
            "process_peak_resident_bytes": 256 * 1024**2,
            "process_physical_footprint_bytes": 80 * 1024**2,
            "swap_growth_bytes": 0,
            "new_throttled_pages": 0,
            "protected_service_pids": services,
        },
    ]
    return {
        "schema_version": 1,
        "commit": "cebc5150b0bd92f6f4098b1d7d1f39c53364e05b",
        "metal_device": "Apple M1",
        "semantic": "mimo_v2_5_layer4_page_aligned_pread_expert_slot_acquisition",
        "artifact_manifest_sha256": (
            "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        ),
        "artifact_sha256": (
            "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        ),
        "artifact_bytes": 201_719_808,
        "expert_stride_bytes": 25_214_976,
        "expert_count": 8,
        "selected_experts": list(range(8)),
        "worker_counts": [1, 2, 4, 8],
        "slot_capacity_bytes": 201_719_808,
        "slot_alignment_bytes": 2 * 1024 * 1024,
        "slot_buffer_pointer_identity": [True] * 8,
        "slot_buffer_lengths": [25_214_976] * 8,
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "U": 8,
        "performance_claim": None,
        "trials": trials,
        "safety_snapshots": safety,
    }


class PrismwingStorageAuthorityTests(unittest.TestCase):
    def test_resident_allocation_uses_exact_16k_ceiling(self):
        self.assertEqual(resident_allocation_bytes(1), 16 * 1024)
        self.assertEqual(resident_allocation_bytes(16 * 1024), 16 * 1024)
        self.assertEqual(resident_allocation_bytes(16 * 1024 + 1), 32 * 1024)
        for invalid in (0, True, 1.5):
            with self.assertRaises(ValueError):
                resident_allocation_bytes(invalid)

    def test_fixed_tensor_names_are_structural_and_exclude_embedding(self):
        weight_map = fixed_weight_map_fixture()
        names = fixed_tensor_names(weight_map)
        self.assertEqual(len(names), 342)
        self.assertNotIn("model.embed_tokens.weight", names)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", names)
        self.assertIn("model.layers.47.mlp.gate.weight", names)
        missing = copy.deepcopy(weight_map)
        del missing["model.layers.47.mlp.gate.weight"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixed_tensor_names(missing)

    def test_fixed_census_rederives_logical_fp8_and_allocated_bytes(self):
        metadata = {
            "f8.weight": {
                "dtype": "F8_E4M3",
                "shape": [2, 3],
                "bytes": 6,
                "backing_file": "one.safetensors",
                "backing_file_sha256": "a" * 64,
            },
            "norm.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "bytes": 8,
                "backing_file": "two.safetensors",
                "backing_file_sha256": "b" * 64,
            },
        }
        result = fixed_census_from_metadata(metadata)
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["logical_source_bytes"], 14)
        self.assertEqual(result["page_aligned_allocation_bytes"], 32 * 1024)
        self.assertEqual(result["fp8_code_bytes"], 6)
        self.assertEqual(result["non_fp8_bytes"], 8)
        self.assertEqual(result["largest_object"], "norm.weight")

        corrupt = copy.deepcopy(metadata)
        corrupt["f8.weight"]["bytes"] = 5
        with self.assertRaisesRegex(ValueError, "metadata bytes"):
            fixed_census_from_metadata(corrupt)

    def test_pw0136_bandwidth_is_derived_from_complete_interleaved_trials(self):
        result = derive_pw0136_bandwidth(pw0136_raw_fixture())
        self.assertEqual(result["selected_workers"], 2)
        self.assertEqual(result["raw_exact_median_ms"], 58.125375)
        self.assertAlmostEqual(
            result["raw_exact_bytes_per_second"],
            BANDWIDTH_EXACT_BYTES_PER_SECOND,
            places=6,
        )
        self.assertAlmostEqual(
            result["candidate_favorable_bytes_per_second"],
            BANDWIDTH_FAVORABLE_BYTES_PER_SECOND,
            places=6,
        )
        self.assertEqual(result["cold_trial_walls_ms"], [58.0, 58.125375, 58.2])

    def test_pw0136_rejects_cache_state_and_bandwidth_drift(self):
        wrong_cache = pw0136_raw_fixture()
        warm = next(row for row in wrong_cache["trials"] if row["cache_state"] == "warm")
        warm["activity"]["disk_bytes_read"] = 1
        with self.assertRaisesRegex(ValueError, "cache-state physical reads"):
            derive_pw0136_bandwidth(wrong_cache)

        wrong_median = pw0136_raw_fixture()
        for row in wrong_median["trials"]:
            if row["cache_state"] == "cold" and row["workers"] == 2:
                row["transfer_wall_ms"] = 70.0 + row["repetition"]
        with self.assertRaisesRegex(ValueError, "bandwidth constants"):
            derive_pw0136_bandwidth(wrong_median)

    @unittest.skipUnless(
        all(
            path.is_file()
            for path in (
                CHECKPOINT_ROOT / "model.safetensors.index.json",
                CHECKPOINT_VERIFICATION_PATH,
                PW0207_OFFLINE_PATH,
                PW0136_RAW_PATH,
                PW0136_ANALYSIS_PATH,
            )
        ),
        "canonical storage authorities unavailable",
    )
    def test_canonical_storage_authorities_rederive_fixed_constants(self):
        result = authenticate_prismwing_storage(CHECKPOINT_ROOT)
        fixed = result["fixed"]
        self.assertEqual(fixed["object_count"], FIXED_OBJECT_COUNT)
        self.assertEqual(fixed["logical_source_bytes"], FIXED_LOGICAL_BYTES)
        self.assertEqual(fixed["page_aligned_allocation_bytes"], FIXED_ALLOCATED_BYTES)
        self.assertEqual(fixed["fp8_code_bytes"], FIXED_FP8_CODE_BYTES)
        self.assertEqual(fixed["non_fp8_bytes"], FIXED_NON_FP8_BYTES)
        self.assertEqual(fixed["largest_object_bytes"], FIXED_MAX_OBJECT_BYTES)
        self.assertAlmostEqual(
            result["bandwidth"]["raw_exact_bytes_per_second"],
            BANDWIDTH_EXACT_BYTES_PER_SECOND,
            places=6,
        )
        self.assertAlmostEqual(
            result["bandwidth"]["candidate_favorable_bytes_per_second"],
            BANDWIDTH_FAVORABLE_BYTES_PER_SECOND,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
