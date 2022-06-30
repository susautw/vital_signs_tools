from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle


if TYPE_CHECKING:
    from occupancy_and_vital_signs_detection.main import Config


class HeatmapPlotter:
    def __init__(
            self,
            config: "Config", *,
            full: plt.Axes = None,
            polar_full: plt.PolarAxes = None,
            zones: dict[int, plt.Axes] = None
    ):
        self.config: "Config" = config
        self.full: plt.Axes = full
        self.polar_full: plt.PolarAxes = polar_full
        self.show_rect: bool = self.polar_full is not None
        self.zones: dict[int, plt.Axes] = {} if zones is None else zones

        self._initialized = False
        self._full_mesh: Optional[QuadMesh] = None
        self._polar_full_mesh: Optional[QuadMesh] = None
        self._zone_plots: dict[int, ZonePlot] = {}

    def set_show_rect(self, val: bool) -> None:
        self.show_rect = val

    def initialize(self):
        azm_space_deg = np.linspace(-60, 60, self.config.num_angle_bins)
        rad_space = np.linspace(0, 3, self.config.num_range_bins)
        azm, rad = np.meshgrid(azm_space := np.deg2rad(azm_space_deg), rad_space)
        azm_deg, rad_deg = np.meshgrid(azm_space_deg, rad_space)
        azm_block_size = azm_space[1] - azm_space[0]
        rad_block_size = rad_space[1] - rad_space[0]

        if self.polar_full is not None:
            self.polar_full.set_theta_zero_location("N")
            self.polar_full.set_thetalim(*np.deg2rad([-60, 60]))
            self.polar_full.set_xlabel("Azimuth")
            self.polar_full.yaxis.set_label_text("Range", y=0.4, verticalalignment="center")
            self.polar_full.grid(False)
            self._polar_full_mesh = self.polar_full.pcolormesh(
                azm,
                rad,
                np.zeros((self.config.num_range_bins, self.config.num_angle_bins))
            )

        if self.full is not None:
            self._full_mesh = self.full.pcolormesh(
                azm,
                rad,
                np.zeros((self.config.num_range_bins, self.config.num_angle_bins))
            )
        should_show_rect = self._should_show_rect()
        for zone_idx, zone_ax in self.zones.items():
            zone = self.config.zone_def.zones[zone_idx]
            zone_plot = self._zone_plots.setdefault(zone_idx, ZonePlot())
            if should_show_rect:
                rect = Rectangle(
                    (
                        self._polar_full_mesh.convert_xunits(azm_space[zone.azimuth_start]),
                        self._polar_full_mesh.convert_yunits(rad_space[zone.range_start])
                    ),
                    zone.azimuth_length * azm_block_size,
                    zone.range_length * rad_block_size,
                    fill=False,
                    linewidth=1
                )
                self.polar_full.add_patch(rect)
                zone_plot.rect = rect

            zone_plot.mesh = zone_ax.pcolormesh(
                azm_deg[zone.idx_slice], rad_deg[zone.idx_slice], np.zeros((zone.range_length, zone.azimuth_length))
            )
        self._initialized = True

    def _should_show_rect(self) -> bool:
        return self.polar_full is not None and self.show_rect

    @property
    def full_mesh(self) -> Optional[QuadMesh]:
        self.raise_if_not_initialized()
        return self._full_mesh

    @property
    def polar_full_mesh(self) -> Optional[QuadMesh]:
        self.raise_if_not_initialized()
        return self._polar_full_mesh

    @property
    def zone_plots(self) -> Optional[dict[int, "ZonePlot"]]:
        self.raise_if_not_initialized()
        return self._zone_plots

    def raise_if_not_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("cannot be used before initialization.")


@dataclass
class ZonePlot:
    mesh: Optional[QuadMesh] = None
    rect: Optional[Rectangle] = None
