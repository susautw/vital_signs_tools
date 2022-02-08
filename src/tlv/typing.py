import array
import mmap
from typing import Union, Any, TypeVar, Type, TYPE_CHECKING
import ctypes

ReadOnlyBuffer = bytes
WriteableBuffer = Union[bytearray, memoryview, array.array, mmap.mmap, "_CData"]
ReadableBuffer = Union[ReadOnlyBuffer, WriteableBuffer]


_CT = TypeVar("_CT", bound="_CData")


if TYPE_CHECKING:
    class _CData:
        @classmethod
        def from_buffer(cls: Type[_CT], source: WriteableBuffer, offset: int = ...) -> _CT: ...

        @classmethod
        def from_buffer_copy(cls: Type[_CT], source: ReadableBuffer, offset: int = ...) -> _CT: ...

        @classmethod
        def from_address(cls: Type[_CT], address: int) -> _CT: ...

        @classmethod
        def from_param(cls: Type[_CT], obj: Any) -> "_CT | _CArgObject": ...

        @classmethod
        def in_dll(cls: Type[_CT], library: ctypes.CDLL, name: str) -> _CT: ...


CData = Union["_CData"]
