from iter_utils import pack


def test_pack():
    assert list(pack("ABCDE", n=2)) == [["A", "B"], ["C", "D"]]
    assert list(pack("ABCDEF", n=2)) == [["A", "B"], ["C", "D"], ["E", "F"]]
