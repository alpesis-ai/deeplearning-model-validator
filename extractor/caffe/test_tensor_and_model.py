import ctypes

import tensor_and_model as TM


def test_tensor_shape():

    tensor_shape = TM.TensorShape()
    tensor_shape.n = ctypes.c_uint(4);
    tensor_shape.channels = ctypes.c_uint(100);
    tensor_shape.height = ctypes.c_uint(200);
    tensor_shape.width = ctypes.c_uint(150);

    print(tensor_shape.n)
    print(tensor_shape.channels)
    print(tensor_shape.height)
    print(tensor_shape.width)


def test_tensor_float():

    tensor_shape = TM.TensorShape()
    tensor_shape.n = ctypes.c_uint(1);
    tensor_shape.channels = ctypes.c_uint(2);
    tensor_shape.height = ctypes.c_uint(3);
    tensor_shape.width = ctypes.c_uint(4);
   
 
    tensor = TM.TensorFloat()
    tensor.shape = tensor_shape
    capacity = tensor.shape.n * tensor.shape.channels * tensor.shape.height * tensor.shape.width
    tensor.capacity = ctypes.c_ulong(capacity)
    tensor.data = (ctypes.c_float * tensor.capacity)()
    for i in xrange(tensor.capacity): tensor.data[i] = i

    print(tensor.shape.n)
    print(tensor.shape.channels)
    print(tensor.shape.height)
    print(tensor.shape.width)
    print(tensor.capacity)
    print(tensor.data[tensor.capacity/2])


def test_model():

    model = TM.Model()
    model.layer_size = 2
    model.layers = (2 * TM.ModelLayer)()
    model.layers[0].index = 0
    model.layers[0].type = TM.conv
    model.layers[0].weights = TM.TensorFloat()
    model.layers[0].weights.shape.n = 1
    model.layers[0].weights.shape.channels = 2
    model.layers[0].weights.shape.height = 3
    model.layers[0].weights.shape.width = 4
    model.layers[0].weights.capacity = 1*2*3*4
    model.layers[0].weights.data = (model.layers[0].weights.capacity * ctypes.c_float)()

    model.layers[1].index = 1
    model.layers[1].type = TM.relu
   

    print(model.layer_size)
    print(model.layers[0].index)
    print(model.layers[0].type)
    print(model.layers[1].index)
    print(model.layers[1].type)


if __name__ == '__main__':

    test_tensor_shape()
    test_tensor_float()
    test_model()
