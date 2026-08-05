import unittest

from tools.remote_checkpoint_census import classify_tensor


class RemoteCheckpointCensusTests(unittest.TestCase):
    def test_standalone_and_main_categories(self):
        self.assertEqual(classify_tensor("model.foo", "audio_tokenizer/model.safetensors"), "audio_tokenizer")
        self.assertEqual(classify_tensor("model.mtp.layers.0.weight", "model_mtp.safetensors"), "mtp")
        self.assertEqual(
            classify_tensor("model.layers.4.mlp.experts.7.up_proj.weight", "main.safetensors"),
            "routed_experts",
        )


if __name__ == "__main__":
    unittest.main()
