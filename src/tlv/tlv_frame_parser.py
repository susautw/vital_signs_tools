import ctypes
from typing import Union

from tlv.typing import CData, WriteableBuffer
from tlv import FrameHeader, TLVHeader, logger, TLVFrame


class TLVFrameParser:
    registery: dict[int, CData]

    def __init__(self):
        self.registery = {}

    def register_type(self, type_number: int, c_type: Union[CData]):
        self.registery[type_number] = c_type

    def parse(self, packet_buffer: WriteableBuffer) -> TLVFrame:
        offset = 0
        frame_header = FrameHeader.from_buffer(packet_buffer, offset)
        offset += ctypes.sizeof(FrameHeader)

        tlvs = []
        for i_tlv in range(frame_header.num_tlvs):
            header = TLVHeader.from_buffer(packet_buffer, offset)
            offset += ctypes.sizeof(TLVHeader)
            if header.type not in self.registery:
                logger.warning(f"Skip the unregistered TLV type {header.type} with length {header.length}")
                offset += header.length
                continue
            tlv_type = self.registery[header.type]
            if header.length != (cdata_size := ctypes.sizeof(tlv_type)):
                logger.warning(f"Skip {i_tlv} TLV by wrong length: ({header.length=}) != ({cdata_size=})")
                offset += header.length
                continue

            tlvs.append(tlv_type.from_buffer(packet_buffer, offset))
            offset += header.length
        return TLVFrame(frame_header, tlvs, packet_buffer)
