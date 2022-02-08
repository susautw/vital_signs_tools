import ctypes


class FrameHeader(ctypes.Structure):
    _fields_ = [
        ('sync', ctypes.c_uint64),
        ('total_packet_len', ctypes.c_uint32),
        ("platform", ctypes.c_uint32),
        ('frame_number', ctypes.c_uint32),
        ('time_cpu_cycles', ctypes.c_uint32),
        ('num_detected_object', ctypes.c_uint32),
        ('num_tlvs', ctypes.c_uint32),
    ]

    sync: int
    total_packet_len: int
    platform: int
    frame_number: int
    time_cpu_cycles: int
    num_detected_object: int
    num_tlvs: int
