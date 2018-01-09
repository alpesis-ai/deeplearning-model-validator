import ctypes


class TensorShape(ctypes.Structure):

    _fields_ = [("n", ctypes.c_uint),
                ("channels", ctypes.c_uint),
                ("height", ctypes.c_uint),
                ("width", ctypes.c_uint)]


class TensorFloat(ctypes.Structure):

    _fields_ = [("shape", TensorShape),
                ("capacity", ctypes.c_ulong),
                ("data", ctypes.POINTER(ctypes.c_float)),
                ("data_gpu", ctypes.POINTER(ctypes.c_float))]


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
