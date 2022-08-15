from utility import RollingAverage
from .mmw_info import MMWInfo


def rolling_average_factory():
    return RollingAverage(window_size=4, low=1000, init=0)
