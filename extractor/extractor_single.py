import sys
import ctypes
import struct
import argparse

import tensor_and_model as TM


def extract_model(net, filepath):

    print "model"
    filename = filepath + "model_weights.model"

    model = TM.Model()
    model.layer_size = len(net.params)
    model.layers = (model.layer_size * TM.ModelLayer)()

    counter = 0
    layer_names = list(net._layer_names)
    for i in xrange(len(layer_names)):
        if (net.layers[i].type == "Convolution") or \
           (net.layers[i].type == "InnerProduct"):
            model.layers[counter].index = i
            if (net.layers[i].type == "Convolution"): model.layers[counter].type = TM.conv
            if (net.layers[i].type == "InnerProduct"): model.layers[counter].type = TM.fc
            counter += 1

    counter = 0
    for name, blobs in net.params.iteritems():
        model.layers[counter].weights = (1*TM.TensorFloat)()
        model.layers[counter].biases = (1*TM.TensorFloat)()

        if (blobs[0]):
            dim_w = len(blobs[0].data.shape)
            model.layers[counter].weights[0].capacity = 1
            for d in range(dim_w): model.layers[counter].weights[0].capacity *= blobs[0].data.shape[d]        
            model.layers[counter].weights[0].data = (model.layers[counter].weights[0].capacity * ctypes.c_float)()
            if dim_w == 4:
                model.layers[counter].weights[0].shape.n = blobs[0].data.shape[0]
                model.layers[counter].weights[0].shape.channels = blobs[0].data.shape[1]
                model.layers[counter].weights[0].shape.height = blobs[0].data.shape[2]
                model.layers[counter].weights[0].shape.width = blobs[0].data.shape[3]
            elif dim_w == 2:
                model.layers[counter].weights[0].shape.n = blobs[0].data.shape[0]
                model.layers[counter].weights[0].shape.channels = blobs[0].data.shape[1]
                model.layers[counter].weights[0].shape.height = 1
                model.layers[counter].weights[0].shape.width = 1
            else:
                pass

        if (blobs[1]):
            dim_b = len(blobs[1].data.shape)
            model.layers[counter].biases[0].capacity = 1
            for d in range(dim_b): model.layers[counter].biases[0].capacity *= blobs[1].data.shape[d]        
            model.layers[counter].biases[0].data = (model.layers[counter].biases[0].capacity * ctypes.c_float)()
            if dim_b == 4:
                model.layers[counter].biases[0].shape.n = blobs[1].data.shape[0]
                model.layers[counter].biases[0].shape.channels = blobs[1].data.shape[1]
                model.layers[counter].biases[0].shape.height = blobs[1].data.shape[2]
                model.layers[counter].biases[0].shape.width = blobs[1].data.shape[3]
            elif dim_b == 2:
                model.layers[counter].biases[0].shape.n = blobs[1].data.shape[0]
                model.layers[counter].biases[0].shape.channels = blobs[1].data.shape[1]
                model.layers[counter].biases[0].shape.height = 1
                model.layers[counter].biases[0].shape.width = 1
            else:
                pass

        counter += 1
 
    f = open(filename, mode="wb")
    f.write(ctypes.c_uint(model.layer_size))
    counter = 0
    for name, blobs in net.params.iteritems(): 
        f.write(ctypes.c_int(model.layers[counter].index))
        f.write(ctypes.c_int(model.layers[counter].type))
        f.write(ctypes.c_uint(model.layers[counter].weights[0].shape.n))
        f.write(ctypes.c_uint(model.layers[counter].weights[0].shape.channels))
        f.write(ctypes.c_uint(model.layers[counter].weights[0].shape.height))
        f.write(ctypes.c_uint(model.layers[counter].weights[0].shape.width))
        f.write(ctypes.c_ulong(model.layers[counter].weights[0].capacity))
        f.write(blobs[0].data[...])
        f.write(ctypes.c_uint(model.layers[counter].biases[0].shape.n))
        f.write(ctypes.c_uint(model.layers[counter].biases[0].shape.channels))
        f.write(ctypes.c_uint(model.layers[counter].biases[0].shape.height))
        f.write(ctypes.c_uint(model.layers[counter].biases[0].shape.width))
        f.write(ctypes.c_ulong(model.layers[counter].biases[0].capacity))
        f.write(blobs[1].data[...])
        counter += 1
    f.close()


def extract_weights(net, filepath):
    """
    """

    print "weights"

    # weights
    for name, blobs in net.params.iteritems():
        for index in range(len(blobs)):
            print "- ", name, blobs[index].data.shape
            this_name = str(name.replace("/", "_"))
            if index == 0:
                filename = filepath + "weights_" + this_name + "_w.bin" 
            elif index == 1:
                filename = filepath + "weights_" + this_name + "_b.bin"
            else:
                filename = filepath + "weights_" + this_name + ".bin"

            this_tensor = TM.TensorFloat()
            dim = len(blobs[index].data.shape)
            this_tensor.capacity = 1
            for d in range(dim):
                this_tensor.capacity *= blobs[index].data.shape[d]
            this_tensor.data = (this_tensor.capacity * ctypes.c_float)()

            if dim == 4:
                this_tensor.shape.n = blobs[index].data.shape[0]
                this_tensor.shape.channels = blobs[index].data.shape[1]
                this_tensor.shape.height = blobs[index].data.shape[2]
                this_tensor.shape.width = blobs[index].data.shape[3]
            elif dim == 2:
                this_tensor.shape.n = blobs[index].data.shape[0]
                this_tensor.shape.channels = blobs[index].data.shape[1]
                this_tensor.shape.height = 1
                this_tensor.shape.width = 1
            else:
                pass


            with open(filename, mode="wb") as f:
                f.write(this_tensor)
                f.write(blobs[index].data[...])




def extract_outputs(net, outpath):
    """
    """

    net.forward()

    print "out"


    for name, blobs in net.blobs.iteritems():
        print "- ", name, net.blobs[name].data.shape
        this_name = str(name.replace("/", "_"))
        filename = outpath + "outs_" + this_name + ".bin"

        this_tensor = TM.TensorFloat() 
        dim = len(net.blobs[name].data.shape)
        this_tensor.capacity = 1
        for d in range(dim):
            this_tensor.capacity *= net.blobs[name].data.shape[d]

        data = []
        if dim == 4:
            this_tensor.shape.n = net.blobs[name].data.shape[0]
            this_tensor.shape.channels = net.blobs[name].data.shape[1]
            this_tensor.shape.height = net.blobs[name].data.shape[2]
            this_tensor.shape.width = net.blobs[name].data.shape[3]
        if dim == 2:
            this_tensor.shape.n = net.blobs[name].data.shape[0]
            this_tensor.shape.channels = net.blobs[name].data.shape[1]
            this_tensor.shape.height = 1
            this_tensor.shape.width = 1
        
        with open(filename, mode="wb") as f:
            f.write(this_tensor)
            f.write(net.blobs[name].data[...])

