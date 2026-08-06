import unittest

from tools.analyze_expert_block_canonicalization_result import codec


class ExpertBlockCanonicalizationResultTests(unittest.TestCase):
    def test_codec_level_is_unique_and_required(self):
        stream = {"codec": [{"level": 1, "value": "fast"}, {"level": 19, "value": "high"}]}
        self.assertEqual(codec(stream, 1)["value"], "fast")
        with self.assertRaises(ValueError):
            codec(stream, 3)
        stream["codec"].append({"level": 1})
        with self.assertRaises(ValueError):
            codec(stream, 1)


if __name__ == "__main__":
    unittest.main()
