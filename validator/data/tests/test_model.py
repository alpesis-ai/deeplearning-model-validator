import ctypes
import unittest

import data.model
import data.tensor


class ModelTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):

        model = data.model.Model()
        model.layer_size = 2
        model.layers = (2 * data.model.ModelLayer)()
        model.layers[0].index = ctypes.c_int(0)
        model.layers[0].type = data.model.conv
        weights = data.tensor.TensorFloat()
        weights.shape.n = 1
        weights.shape.channels = 2
        weights.shape.height = 3
        weights.shape.width = 4
        weights.capacity = 1*2*3*4
        # weights.data = (model.layers[0].weights.capacity * ctypes.c_float)()

        model.layers[1].index = 1
        model.layers[1].type = data.model.relu
   
        self.assertEqual(model.layer_size, 2)
        self.assertEqual(model.layers[0].index, 0)
        self.assertEqual(model.layers[0].type, 0)
        self.assertEqual(model.layers[1].index, 1)
        self.assertEqual(model.layers[1].type, 2)


if __name__ == '__main__':

    unittest.main()
