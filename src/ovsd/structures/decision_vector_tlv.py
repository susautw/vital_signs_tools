import ctypes


def decision_vector_tlv(n_zones: int = 2):
    vec_type = ctypes.c_bool * n_zones
    return vec_type
