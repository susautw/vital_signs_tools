import ctypes

SUPPORTED_DTYPES = [ctypes.c_uint16, ctypes.c_uint8, ctypes.c_float]


def range_azimuth_heatmap_tlv(n_range_bins: int = 64, n_azimuth_bins: int = 48, dtype=ctypes.c_uint16):
    if n_range_bins <= 0 or n_azimuth_bins < 0 or dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"invalid arguments {n_range_bins=}, {n_azimuth_bins=}, {dtype=}")

    map_type = dtype * n_azimuth_bins * n_range_bins

    return map_type
