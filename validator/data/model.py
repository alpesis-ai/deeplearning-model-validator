import ctypes

from tensor import TensorFloat

(conv, 
 pool, 
 relu, 
 fc, 
 softmax, 
 rpn, 
 roi_pool, 
 reshape) = map(ctypes.c_int, xrange(8))


class ModelLayer(ctypes.Structure):

    _fields_ = [("index", ctypes.c_int),
                ("type", ctypes.c_int),
                ("weights", ctypes.POINTER(TensorFloat)),
                ("biases", ctypes.POINTER(TensorFloat))]


class Model(ctypes.Structure):

    _fields_ = [("layer_size", ctypes.c_uint),
                ("layers", ctypes.POINTER(ModelLayer))]
