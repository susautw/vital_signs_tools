from typing import TypeVar, Iterable, Iterator

T = TypeVar("T")


def pack(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """
    list(pack("ABCDE", n=2)) => [["A", "B"], ["C", "D"]]
    :param iterable:
    :param n:
    :return:
    """
    cache = []
    for t in iterable:
        cache.append(t)
        if len(cache) == n:
            yield cache
            cache = []
