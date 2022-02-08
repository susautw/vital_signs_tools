from dataclasses import dataclass
from typing import Sequence, overload

from tlv import FrameHeader
from tlv.typing import CData


@dataclass
class TLVFrame(Sequence[CData]):
    frame_header: FrameHeader
    tlvs: Sequence[CData]

    def __len__(self):
        return self.frame_header.num_tlvs

    @overload
    def __getitem__(self, i: slice) -> Sequence[CData]: ...
    @overload
    def __getitem__(self, i: int) -> CData: ...

    def __getitem__(self, i):
        return self.tlvs[i]
