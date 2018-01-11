import ctypes
import unittest

import data.tensor


class TensorTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_tensor_shape(self):
        tensor_shape = data.tensor.TensorShape()
        tensor_shape.n = ctypes.c_uint(4);
        tensor_shape.channels = ctypes.c_uint(100);
        tensor_shape.height = ctypes.c_uint(200);
        tensor_shape.width = ctypes.c_uint(150);

        self.assertEqual(tensor_shape.n, 4)
        self.assertEqual(tensor_shape.channels, 100)
        self.assertEqual(tensor_shape.height, 200)
        self.assertEqual(tensor_shape.width, 150)


    def test_tensor_float(self):
        tensor_shape = data.tensor.TensorShape()
        tensor_shape.n = ctypes.c_uint(1);
        tensor_shape.channels = ctypes.c_uint(2);
        tensor_shape.height = ctypes.c_uint(3);
        tensor_shape.width = ctypes.c_uint(4);
   
 
        tensor = data.tensor.TensorFloat()
        tensor.shape = tensor_shape
        capacity = tensor.shape.n * tensor.shape.channels * tensor.shape.height * tensor.shape.width
        tensor.capacity = ctypes.c_ulong(capacity)
        tensor.data = (ctypes.c_float * tensor.capacity)()
        for i in xrange(tensor.capacity): tensor.data[i] = i

        self.assertEqual(tensor.shape.n, 1)
        self.assertEqual(tensor.shape.channels, 2)
        self.assertEqual(tensor.shape.height, 3)
        self.assertEqual(tensor.shape.width, 4)
        self.assertEqual(tensor.capacity, 24)
        self.assertEqual(tensor.data[tensor.capacity/2], tensor.capacity/2)



if __name__ == '__main__':

    unittest.main()
