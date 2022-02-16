import ctypes
import logging
import time
from io import RawIOBase
from typing import Iterator

from tlv import TLVFrameParser, FrameHeader, TLVFrame

MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
READ_BYTES = 1024
FRAME_SIZE = ctypes.sizeof(FrameHeader)


def from_stream(
        stream: RawIOBase,
        parser: TLVFrameParser,
        delay: float = None,
        max_buffer=2 ** 16,
        magic_word=MAGIC
) -> Iterator[TLVFrame]:
    buffer = bytearray(max_buffer)
    buffer_len = 0
    current_frame = None

    while not stream.closed:
        if buffer_len + READ_BYTES > len(buffer):
            buffer_len = 0  # clear buffer
            current_frame = None

        tmp = stream.read(READ_BYTES)
        if tmp == b'':
            break
        buffer[buffer_len: buffer_len + len(tmp)] = tmp
        buffer_len += len(tmp)

        if current_frame is None:
            magic_pos = buffer.find(magic_word, 0, buffer_len)
            if magic_pos == -1:
                continue
            if magic_pos > 0:  # strip data before magic word
                packet_len = buffer_len - magic_pos
                buffer[:packet_len] = buffer[magic_pos:buffer_len]
                buffer_len = packet_len

            if buffer_len >= FRAME_SIZE:
                current_frame = FrameHeader.from_buffer(buffer[:FRAME_SIZE])

        # previous if block MAY create current_frame, so MUST NOT use elif for following block
        if current_frame is not None and buffer_len >= current_frame.total_packet_len:
            try:
                yield parser.parse(buffer[:current_frame.total_packet_len])
            except ValueError as e:
                logging.exception(e)
                logging.warning("skipped this frame")

            new_buffer_len = max(0, buffer_len - current_frame.total_packet_len)
            buffer[:new_buffer_len] = buffer[current_frame.total_packet_len: buffer_len]
            buffer_len = new_buffer_len
            current_frame = None

            if delay:
                time.sleep(delay)
