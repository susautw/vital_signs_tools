import ctypes
from typing import Type

from fancy import config as cfg

from config_loader import SequentialLoader
from sdk_configs import ChannelConfig, ProfileConfig, ChirpConfig, FrameConfig
from .plot import AbstractZone


class GUIConfig(cfg.BaseConfig):
    decision: bool = cfg.Option(type=bool)
    heatmap: int = cfg.Option(type=int)
    vital_signs: bool = cfg.Option(type=bool)

    heatmap_dtype: Type = cfg.PlaceHolder()

    heatmap_dtype_map = {
        8: ctypes.c_uint8,
        16: ctypes.c_uint16,
        32: ctypes.c_float
    }

    def post_load(self):
        self.heatmap_dtype = self.heatmap_dtype_map[self.heatmap]


class Zone(cfg.BaseConfig, AbstractZone):
    range_start: int = cfg.Option(type=int)
    range_length: int = cfg.Option(type=int)
    azimuth_start: int = cfg.Option(type=int)
    azimuth_length: int = cfg.Option(type=int)


class ZoneDef(cfg.BaseConfig):
    number_of_zones: int = cfg.Option(type=int)
    _zone_defs: list[int] = cfg.Option(name="zone_defs", type=[int])

    zones: list[Zone] = cfg.PlaceHolder()
    geo_sorted_ids: list[int] = cfg.PlaceHolder()

    def post_load(self):
        self.zones = []
        for i in range(self.number_of_zones):
            loader = SequentialLoader(self._zone_defs[i * 4: i * 4 + 4])
            self.zones.append(Zone(loader))
        geo_sorted_pair = sorted(enumerate(self.zones), key=lambda x: x[1].order(), reverse=True)
        self.geo_sorted_ids = [0] * len(geo_sorted_pair)
        for geo_sorted_idx, (idx, zone) in enumerate(geo_sorted_pair):
            self.geo_sorted_ids[idx] = geo_sorted_idx


class OVSDConfig(cfg.BaseConfig):
    channel_cfg: ChannelConfig = cfg.Option(type=ChannelConfig)
    profile_cfg: ProfileConfig = cfg.Option(type=ProfileConfig)
    chirp_cfg: list[ChirpConfig] = cfg.Option(type=[ChirpConfig])
    frame_cfg: FrameConfig = cfg.Option(type=FrameConfig)
    gui_monitor: GUIConfig = cfg.Option(type=GUIConfig)
    zone_def: ZoneDef = cfg.Option(type=ZoneDef)

    num_chirps_per_frame: int = cfg.PlaceHolder()
    num_doppler_bins: int = cfg.PlaceHolder()
    num_range_bins: int = cfg.PlaceHolder()
    num_angle_bins: int = cfg.Lazy(lambda c: 48)
    range_resolution_meters: float = cfg.PlaceHolder()
    range_idx_to_meter: float = cfg.PlaceHolder()
    doppler_resolution_mps: float = cfg.PlaceHolder()

    def post_load(self):
        self.load_lazies()
        self.num_chirps_per_frame = (self.frame_cfg.chirp_end_index
                                     - self.frame_cfg.chirp_start_index + 1) * self.frame_cfg.number_of_loops

        self.num_doppler_bins = self.num_chirps_per_frame // self.channel_cfg.num_tx_ant
        self.num_range_bins = self._power_2_roundup(self.profile_cfg.num_adc_samples)
        self.range_resolution_meters = 3e8 * self.profile_cfg.dig_out_sample_rate * 1e3 / (
                2 * self.profile_cfg.freq_slope_const * 1e12 * self.profile_cfg.num_adc_samples
        )
        self.range_idx_to_meter = 3e8 * self.profile_cfg.dig_out_sample_rate * 1e3 / (
                2 * self.profile_cfg.freq_slope_const * 1e12 * self.num_range_bins
        )
        self.doppler_resolution_mps = 3e8 / (2 * self.profile_cfg.start_freq * 1e9 * (
                self.profile_cfg.idle_time + self.profile_cfg.ramp_end_time
        ) * 1e-6 * self.num_doppler_bins * self.channel_cfg.num_tx_ant)

    @staticmethod
    def _power_2_roundup(x: int) -> int:
        y = 1
        while y < x:
            y *= 2
        return y
