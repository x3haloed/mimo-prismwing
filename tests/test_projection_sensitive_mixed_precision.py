import unittest
import mlx.core as mx
import numpy as np
from tools.run_projection_sensitive_mixed_precision import projection_configs, array_bytes
from tools.run_microscaling_fp4_real_expert import quantize_projection

class ProjectionSensitiveMixedPrecisionTest(unittest.TestCase):
    def test_projection_configs_and_bytes_are_explicit(self):
        configs=projection_configs((3,3,6));self.assertEqual([configs[x]["bits"] for x in ("gate","up","down")],[3,3,6])
        weights={"gate":np.zeros((2,128),dtype=np.float16),"up":np.zeros((2,128),dtype=np.float16),"down":np.zeros((4,128),dtype=np.float16)}
        arrays={name:quantize_projection(weight,configs[name]) for name,weight in weights.items()}
        self.assertEqual(sum(array_bytes(value) for value in arrays.values()),sum(int(x.nbytes) for value in arrays.values() for x in value if x is not None))

if __name__=="__main__":unittest.main()
