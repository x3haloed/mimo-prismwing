import unittest
from tools.analyze_prompt_lookup_one_tps import propose,simulate

class PromptLookupTest(unittest.TestCase):
    def test_longest_most_recent_and_target_bonus(self):
        self.assertEqual(propose([1,2,3,1,2],2,3,3),[3,1,2])
        row=simulate([1,2,3,1,2],[3,9],2,2)
        self.assertEqual(row["passes"],1)
        self.assertEqual(row["mean_A"],2)

if __name__=="__main__":unittest.main()
