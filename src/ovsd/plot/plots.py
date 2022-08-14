from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING, Generic, TypeVar, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.figure import FigureBase
from matplotlib.patches import Rectangle

from . import AbstractZone

if TYPE_CHECKING:
    from .plot_visitors import IPlotVisitor

T = TypeVar("T")


class IPlot(ABC, Generic[T]):
    @abstractmethod
    def draw(self) -> None: ...

    @abstractmethod
    def accept(self, visitor: "IPlotVisitor"): ...

    @abstractmethod
    def set_data(self, data: T) -> None: ...


class ZoneInfoPlot(IPlot): ...


class PlotGroup(IPlot[None]):
    _children: list[IPlot]

    def __init__(self):
        self._children = []

    def draw(self) -> None:
        for p in self._children:
            p.draw()

    def accept(self, visitor: "IPlotVisitor"):
        visitor.visit_plot_group(self)

    def get_plots(self) -> list[IPlot]:
        return self._children

    def add_plot(self, *p: IPlot) -> None:
        self._children.extend(p)
    
    def set_data(self, data: None) -> None:
        raise RuntimeError("group does not accept data")


class AbstractHMapPlot(IPlot[T], ABC):
    fig: FigureBase
    x_axis: np.ndarray
    y_axis: np.ndarray
    mesh: QuadMesh

    def __init__(self, fig: FigureBase, x: np.ndarray, y: np.ndarray):
        if x.shape != y.shape:
            raise ValueError(f"the shape of x and y must be the same")

        self.fig = fig
        self.x_axis = x
        self.y_axis = y

        self.mesh = self.get_axes().pcolormesh(x, y, np.zeros_like(x))
        self.mesh.set_animated(True)

    @abstractmethod
    def get_axes(self) -> plt.Axes: ...

    def set_clim(self, cmin: float, cmax: float) -> None:
        self.mesh.set_clim(cmin, cmax)

    def draw(self):
        self.get_axes().draw_artist(self.mesh)

    def accept(self, visitor: "IPlotVisitor"):
        visitor.visit_abstract_hmap_plot(self)


class AbstractFullHMapPlot(AbstractHMapPlot[Tuple[np.ndarray, dict[AbstractZone, bool]]], ABC):
    rect_zones: Sequence[AbstractZone]

    def __init__(self, fig: FigureBase, x: np.ndarray, y: np.ndarray, rect_zones: Sequence[AbstractZone]):
        super().__init__(fig, x, y)
        self.rect_zones = rect_zones
        self._zone_rect_patch = {}
        x_space = x[0]
        y_space = y[:, 0]
        for zone in rect_zones:
            self._zone_rect_patch[zone] = Rectangle(
                (
                    self.mesh.convert_xunits(x_space[zone.azimuth_start]),
                    self.mesh.convert_yunits(y_space[zone.range_start])
                ),
                x_space[zone.azimuth_length] - x_space[0],
                y_space[zone.range_length] - y_space[0],
                fill=False,
                linewidth=1
            )
            self._zone_rect_patch[zone].set_animated(True)
            self.get_axes().add_patch(self._zone_rect_patch[zone])

    def set_data(self, data: Tuple[np.ndarray, dict[AbstractZone, bool]]) -> None:
        arr, decision = data
        self.mesh.set_array(arr)
        for zone, rect in self._zone_rect_patch.items():
            if zone in decision:
                rect.set_edgecolor("g" if decision[zone] else "r")

    def draw(self):
        super().draw()
        for artist in sorted(self._zone_rect_patch.values(), key=lambda x: x.get_zorder()):
            self.get_axes().draw_artist(artist)

    def accept(self, visitor: "IPlotVisitor"):
        super().accept(visitor)
        visitor.visit_abstract_full_hmap_plot(self)


class ZoneHMapPlot(AbstractHMapPlot[np.ndarray]):
    zone: AbstractZone

    def __init__(self, fig: FigureBase, x: np.ndarray, y: np.ndarray, zone: AbstractZone):
        super().__init__(fig, x, y)
        self.zone = zone

    @cache
    def get_axes(self) -> plt.Axes:
        return self.fig.add_subplot(111)

    def set_data(self, data: np.ndarray) -> None:
        self.mesh.set_array(data)

    def accept(self, visitor: "IPlotVisitor"):
        super().accept(visitor)
        visitor.visit_zone_hmap_plot(self)


class PolarHMapPlot(AbstractFullHMapPlot):
    @cache
    def get_axes(self) -> plt.Axes:
        ax: plt.PolarAxes = self.fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_thetalim(*np.deg2rad([-60, 60]))
        ax.set_xlabel("Azimuth")
        ax.yaxis.set_label_text("Range", y=0.4, verticalalignment="center")
        ax.grid(False)
        return ax

    def accept(self, visitor: "IPlotVisitor"):
        super().accept(visitor)
        visitor.visit_polar_hmap_plot(self)


class FullHMapPlot(AbstractFullHMapPlot):
    @cache
    def get_axes(self) -> plt.Axes:
        return self.fig.add_subplot(111)

    def accept(self, visitor: "IPlotVisitor"):
        super().accept(visitor)
        visitor.visit_full_hmap_plot(self)
