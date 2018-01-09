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
