import unittest
from tools.analyze_pw0322_causal_q64 import route_union

class Pw0322Tests(unittest.TestCase):
    def test_route_union_preserves_layer_identity(self):
        traces=[{'layer':0,'selected_experts_by_position':[]}]
        for layer in range(1,48):
            traces.append({'layer':layer,'selected_experts_by_position':[list(range(8)) for _ in range(64)],'route_weights_by_position':[[0.125]*8 for _ in range(64)]})
        ids=route_union({'verification_layer_traces':traces})
        self.assertEqual(len(ids),47*8)
        self.assertIn((1,0),ids); self.assertIn((47,7),ids)

if __name__=='__main__': unittest.main()
