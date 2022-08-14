from enum import IntEnum
from typing import Optional

import numpy as np
from matplotlib.figure import FigureBase

from . import AbstractZone, plots
from ovsd.configs import OVSDConfig


class PlotType(IntEnum):
    ZONE_INFO = 0
    ZONE_HMAP = 1
    POLAR_HMAP = 2
    FULL_HMAP = 3


class PlotGroupBuilder:
    x_axis: np.ndarray = None
    y_axis: np.ndarray = None
    type_fig_map: dict[tuple[PlotType, Optional[AbstractZone]], FigureBase]
    zones: list[AbstractZone]
    show_rect_in_hmap: bool = True

    def __init__(self):
        self.type_fig_map = {}
        self.zones = []

    def with_config(self, config: OVSDConfig) -> "PlotGroupBuilder":
        azm_space_deg = np.linspace(-60, 60, config.num_angle_bins)
        rad_space = np.linspace(0, 3, config.num_range_bins)
        self.x_axis, self.y_axis = np.meshgrid(np.deg2rad(azm_space_deg), rad_space)
        return self

    def add_plot_type(self, plot_type: PlotType, fig: FigureBase, zone: AbstractZone = None) -> "PlotGroupBuilder":
        if not (0 <= plot_type <= 3):
            raise ValueError(f"invalid plot_type {plot_type}")
        if plot_type <= PlotType.ZONE_HMAP:
            if zone is None:
                raise ValueError("Zone plot types must specify a zone")
            self.type_fig_map[plot_type, zone] = fig
            self.add_zone(zone)
        else:
            self.type_fig_map[plot_type, None] = fig
        return self

    def add_zone(self, zone: AbstractZone) -> "PlotGroupBuilder":
        if zone not in self.zones:
            self.zones.append(zone)
        return self

    def set_show_rect_in_hmap(self, val: bool) -> "PlotGroupBuilder":
        self.show_rect_in_hmap = val
        return self

    def build(self) -> plots.PlotGroup:
        pg = plots.PlotGroup()
        for (plot_type, zone), fig in self.type_fig_map.items():
            plot: plots.IPlot
            if plot_type == PlotType.ZONE_INFO:
                raise RuntimeError("This type is not implemented.")
            elif plot_type == PlotType.ZONE_HMAP:
                plot = plots.ZoneHMapPlot(fig, self.x_axis[zone.idx_slice], self.y_axis[zone.idx_slice], zone)
            elif plot_type == PlotType.FULL_HMAP:
                plot = plots.FullHMapPlot(
                    fig, self.x_axis, self.y_axis, rect_zones=self.zones if self.show_rect_in_hmap else []
                )
            elif plot_type == PlotType.POLAR_HMAP:
                plot = plots.PolarHMapPlot(
                    fig, self.x_axis, self.y_axis, rect_zones=self.zones if self.show_rect_in_hmap else []
                )
            else:
                raise RuntimeError("This statement is never executed.")
            pg.add_plot(plot)
        return pg
