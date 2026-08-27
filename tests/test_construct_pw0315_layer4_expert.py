import unittest

import tools.construct_pw0314_layer4_k4 as base
from tools.construct_pw0315_layer4_expert import (
    EXPERT64_CONTROL,
    EXPERT_AUTHORITIES,
    EXPECTED_PLACEMENTS,
    configure,
)


class Pw0315Layer4ExpertTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "experiment": base.EXPERIMENT_ID,
            "expert": base.EXPERT,
            "shard": base.SOURCE_SHARD,
            "digest": base.SOURCE_SHARD_SHA256,
            "control": base.EXPECTED_PROJECTION_HASHES,
            "placements": base.EXPECTED_PLACEMENT_COUNT,
            "verify": base.verify_full_checkpoint_install,
        }

    def tearDown(self):
        base.EXPERIMENT_ID = self.original["experiment"]
        base.EXPERT = self.original["expert"]
        base.SOURCE_SHARD = self.original["shard"]
        base.SOURCE_SHARD_SHA256 = self.original["digest"]
        base.EXPECTED_PROJECTION_HASHES = self.original["control"]
        base.EXPECTED_PLACEMENT_COUNT = self.original["placements"]
        base.verify_full_checkpoint_install = self.original["verify"]

    def test_configure_pins_identity_authorities(self):
        configure(96)
        self.assertEqual(base.EXPERIMENT_ID, "PW-0315")
        self.assertEqual(base.EXPERT, 96)
        self.assertEqual(base.SOURCE_SHARD, EXPERT_AUTHORITIES[96][0])
        self.assertEqual(base.SOURCE_SHARD_SHA256, EXPERT_AUTHORITIES[96][1])
        self.assertEqual(base.EXPECTED_PLACEMENT_COUNT, EXPECTED_PLACEMENTS[96])
        self.assertIsNone(base.EXPECTED_PROJECTION_HASHES)

    def test_expert64_enables_immutable_projection_control(self):
        configure(64)
        self.assertEqual(base.EXPECTED_PROJECTION_HASHES, EXPERT64_CONTROL)

    def test_configure_rejects_unplanned_identity(self):
        with self.assertRaisesRegex(ValueError, "expert must be one of"):
            configure(9)


if __name__ == "__main__":
    unittest.main()
