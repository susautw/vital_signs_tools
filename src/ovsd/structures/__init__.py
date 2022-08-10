import ctypes
from typing import Type, Union

from tlv import TLVFrameParser
from .decision_vector_tlv import decision_vector_tlv
from .range_azimuth_heatmap_tlv import range_azimuth_heatmap_tlv
from .vital_signs_vector_tlv import vital_signs_vector_tlv, VitalSignsVectorTLV
from ..configs import OVSDConfig

heatmap_type: Type[ctypes.Array[ctypes.Array[Union[ctypes.c_uint8, ctypes.c_uint16, ctypes.c_float]]]]
decision_type: Type[ctypes.Array[ctypes.c_bool]]
vital_signs_type: Type[ctypes.Array[VitalSignsVectorTLV]]

_initialized = False

__all__ = [
    'decision_vector_tlv', 'range_azimuth_heatmap_tlv', 'vital_signs_vector_tlv', 'VitalSignsVectorTLV',
    'heatmap_type', 'decision_type', 'vital_signs_type', 'init_structures', 'get_parser'
]


def init_structures(config: OVSDConfig):
    global heatmap_type, decision_type, vital_signs_type, _initialized
    heatmap_type = range_azimuth_heatmap_tlv(
        config.num_range_bins, config.num_angle_bins, config.gui_monitor.heatmap_dtype
    )
    decision_type = decision_vector_tlv(config.zone_def.number_of_zones)
    vital_signs_type = vital_signs_vector_tlv(config.zone_def.number_of_zones)
    _initialized = True


def get_parser(config: OVSDConfig = None):
    if config is None:
        if not _initialized:
            raise RuntimeError("no config provided. please provide a config or call init_structures first.")
    else:
        init_structures(config)
    tlv_frame_parser = TLVFrameParser()
    # noinspection PyTypeChecker
    tlv_frame_parser.register_type(8, heatmap_type)
    # noinspection PyTypeChecker
    tlv_frame_parser.register_type(9, decision_type)
    # noinspection PyTypeChecker
    tlv_frame_parser.register_type(10, vital_signs_type)
    return tlv_frame_parser
