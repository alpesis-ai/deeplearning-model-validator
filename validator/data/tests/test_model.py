import ctypes

import tensor_and_model as TM


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

    test_model()
