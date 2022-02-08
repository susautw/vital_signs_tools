import argparse
import ctypes
import logging
import sys
from pathlib import Path
from typing import Type, Union

import serial
from fancy import config as cfg

import structures
from config_loader import SequentialLoader, MMWaveConfigLoader
from sdk_configs import ChannelConfig, FrameConfig, ProfileConfig, ChirpConfig
from tlv import from_stream, TLVFrameParser, TLVFrame

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

heatmap_type: Type[ctypes.Array[ctypes.Array[Union[ctypes.c_uint8, ctypes.c_uint16, ctypes.c_float]]]]
decision_type: Type[ctypes.Array[ctypes.c_bool]]
vital_signs_type: Type[ctypes.Array[structures.VitalSignsVectorTLV]]


def main():
    global heatmap_type, decision_type, vital_signs_type

    args = get_arg_parser().parse_args()
    cli_port = serial.Serial(args.cli_port, 115200)
    data_port = serial.Serial(args.data_port, 921600)

    with args.config.open() as fp:
        lines: list[str] = list(fp.readlines())
        config = Config(MMWaveConfigLoader(lines))
        for line in lines:
            line = line.strip()
            if len(line.replace("\t", "").replace(" ", "")) == 0:
                continue
            if line.startswith("%"):
                continue
            cli_port.write(line.encode())
            logger.info(f"cli: {line}")
            logger.info(f"cli {cli_port.readline()}")

    tlv_frame_parser = TLVFrameParser()
    tlv_frame_parser.register_type(8, heatmap_type := structures.range_azimuth_heatmap_tlv(
        config.num_range_bins, config.num_angle_bins, config.gui_monitor.heatmap_dtype
    ))
    tlv_frame_parser.register_type(9, decision_type := structures.decision_vector_tlv(config.zone_def.number_of_zones))
    tlv_frame_parser.register_type(
        10, vital_signs_type := structures.vital_signs_vector_tlv(config.zone_def.number_of_zones)
    )

    try:
        for frame in from_stream(data_port, tlv_frame_parser):
            accept_frame(frame)
    except KeyboardInterrupt:
        cli_port.write(b"SensorStop\n")


def accept_frame(frame: TLVFrame):
    logger.info(f"get {len(frame)} tlvs from a frame")
    for tlv in frame:
        if isinstance(tlv, heatmap_type):
            logger.debug("updating heatmap")

        elif isinstance(tlv, decision_type):
            logger.debug("updating decision")
            ...  # update decision
        elif isinstance(tlv, vital_signs_type):
            logger.debug("updating vital_signs")
            ...  # update vital_signs


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


class Zone(cfg.BaseConfig):
    range_start: int = cfg.Option(type=int)
    range_length: int = cfg.Option(type=int)
    azimuth_start: int = cfg.Option(type=int)
    azimuth_length: int = cfg.Option(type=int)


class ZoneDef(cfg.BaseConfig):
    number_of_zones: int = cfg.Option(type=int)
    _zone_defs: list[int] = cfg.Option(name="zone_defs", type=[int])

    zones: list[Zone] = cfg.PlaceHolder()

    def post_load(self):
        self.zones = []
        for i in range(self.number_of_zones):
            loader = SequentialLoader(self._zone_defs[i * 4: i * 4 + 4])
            self.zones.append(Zone(loader))


class Config(cfg.BaseConfig):
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


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cli_port", type=str)
    parser.add_argument("data_port", type=str)
    parser.add_argument("-c", "--config", type=Path, required=True)
    return parser


if __name__ == '__main__':
    main()
