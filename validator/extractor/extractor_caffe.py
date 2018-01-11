import sys
import ctypes

import data.tensor
import data.model 


class CaffeExtractor:

    def __init__(self, net, netname, net_pt, model, out_path):
        self.net = net
        self.netname = netname
        self.net_pt = net_pt
        self.model = model
        self.outpath = out_path


    def extract_model(self):
        filename = self._create_model_filename("model_weights.model")

        model = data.model.Model()
        model.layer_size = len(self.net.params)
        model.layers = (model.layer_size * data.model.ModelLayer)()

        counter = 0
        layer_names = list(self.net._layer_names)
        for i in xrange(len(layer_names)):
            if (self.net.layers[i].type == "Convolution") or \
               (self.net.layers[i].type == "InnerProduct"):
                model.layers[counter].index = i
                if (self.net.layers[i].type == "Convolution"): model.layers[counter].type = data.model.conv
                if (self.net.layers[i].type == "InnerProduct"): model.layers[counter].type = data.model.fc
                counter += 1

        counter = 0
        for name, blobs in self.net.params.iteritems():
            model.layers[counter].weights = (1*data.tensor.TensorFloat)()
            model.layers[counter].biases = (1*data.tensor.TensorFloat)()

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
        for name, blobs in self.net.params.iteritems(): 
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



    def extract_weights(self):
        for name, blobs in self.net.params.iteritems():
            for index in range(len(blobs)):
                print "- ", name, blobs[index].data.shape
                filename = self._create_layer_filename(name, "weights_", index)

                this_tensor = data.tensor.TensorFloat()
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


                self._save_file(filename, this_tensor, blobs[index].data[...]) 


    def extract_outputs(self):
        self.net.forward()

        for name, blobs in self.net.blobs.iteritems():
            print "- ", name, self.net.blobs[name].data.shape
            filename = self._create_layer_filename(name, "outs_", 0)

            this_tensor = data.tensor.TensorFloat() 
            dim = len(self.net.blobs[name].data.shape)
            this_tensor.capacity = 1
            for d in range(dim):
                this_tensor.capacity *= self.net.blobs[name].data.shape[d]

            if dim == 4:
                this_tensor.shape.n = self.net.blobs[name].data.shape[0]
                this_tensor.shape.channels = self.net.blobs[name].data.shape[1]
                this_tensor.shape.height = self.net.blobs[name].data.shape[2]
                this_tensor.shape.width = self.net.blobs[name].data.shape[3]
            if dim == 2:
                this_tensor.shape.n = self.net.blobs[name].data.shape[0]
                this_tensor.shape.channels = self.net.blobs[name].data.shape[1]
                this_tensor.shape.height = 1
                this_tensor.shape.width = 1
       
            self._save_file(filename, this_tensor, self.net.blobs[name].data[...]) 


    def _create_model_filename(self, name):
        filename = "{0}{1}_{2}".format(self.outpath, self.netname, name)
        return filename


    def _create_layer_filename(self, name, filetype, index):
        filename = str(name.replace("/", "_"))
        filename = "{0}{1}_{2}{3}_{4}.data".format(self.outpath, self.netname, filetype, filename, str(index))
        return filename


    def _save_file(self, filename, tensor, data):
        with open(filename, mode="wb") as f:
            print "filepath: ", filename
            f.write(tensor)
            f.write(data)
