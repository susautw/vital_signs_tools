import ctypes


class TLVHeader(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32),
        ('length', ctypes.c_uint32),
    ]

    type: int
    length: int

