import unittest

from tools.audit_dflash_draft import expected_inventory, validate_config, validate_inventory


class DFlashDraftAuditTests(unittest.TestCase):
    def test_expected_inventory_is_complete_and_shape_locked(self):
        inventory = expected_inventory()
        self.assertEqual(len(inventory), 63)
        validate_inventory(inventory)
        changed = dict(inventory)
        changed["layers.4.self_attn.q_proj.weight"] = ("BF16", (4096, 8192))
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            validate_inventory(changed)

    def test_config_fails_closed_on_semantic_change(self):
        config = {
            "model_type": "qwen3",
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 5,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "partial_rotary_factor": 0.5,
            "block_size": 8,
            "num_target_layers": 48,
            "vocab_size": 152576,
            "max_position_embeddings": 262144,
            "rope_theta": 10000,
            "sliding_window": 1024,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16",
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
            "dflash_config": {
                "target_layer_ids": [0, 11, 23, 35, 47],
                "mask_token_id": 151675,
                "block_size": 8,
                "use_swa": True,
                "swa_window_size": 1024,
                "backbone_rotary_base": 5000000,
                "attention_value_scale": 0.612,
                "attention_sink_bias": True,
            },
        }
        validate_config(config)
        config["dflash_config"]["target_layer_ids"][-1] = 46
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
