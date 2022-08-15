import ctypes


def vital_signs_vector_tlv(n_zones: int = 2):
    return VitalSignsVectorTLV * n_zones


class VitalSignsVectorTLV(ctypes.Structure):
    _fields_ = [
        ('unwrapped_waveform', ctypes.c_float),
        ('heart_waveform', ctypes.c_float),
        ('breathing_waveform', ctypes.c_float),
        ('heart_rate', ctypes.c_float),
        ('breathing_rate', ctypes.c_float)
    ]

    unwrapped_waveform: float
    heart_waveform: float
    breathing_waveform: float
    heart_rate: float
    breathing_rate: float

    def __repr__(self):
        show = {name: getattr(self, name) for name, _ in self._fields_}
        return f'<{type(self).__name__} {show}>'
