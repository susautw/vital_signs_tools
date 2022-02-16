import argparse
import ctypes
import logging
import sys
from collections import deque
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Type, Union, IO, Optional

import serial
from fancy import config as cfg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle

import structures
from config_loader import SequentialLoader, MMWaveConfigLoader
from sdk_configs import ChannelConfig, FrameConfig, ProfileConfig, ChirpConfig
from tlv import from_stream, TLVFrameParser, TLVFrame
from utils import test_time

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

args: "ArgConfig"
config: "Config"
visualizer: "Visualizer"

heatmap_type: Type[ctypes.Array[ctypes.Array[Union[ctypes.c_uint8, ctypes.c_uint16, ctypes.c_float]]]]
decision_type: Type[ctypes.Array[ctypes.c_bool]]
vital_signs_type: Type[ctypes.Array[structures.VitalSignsVectorTLV]]


def main():
    global args, heatmap_type, decision_type, vital_signs_type, visualizer

    args = ArgConfig(cfg.DictConfigLoader(vars(get_arg_parser().parse_args())))

    if args.mode == "serial":
        cli_port = serial.Serial(args.cli_port, baudrate=115200, timeout=10)
        data_stream = serial.Serial(args.data_port, baudrate=921600, timeout=10)
    elif args.mode == "file":
        cli_port = None
        data_stream = args.file.open("rb")
    else:
        logger.error(f"Unknown mode {args.mode}")
        return 1

    load_and_send_config(args.config, cli_port)

    visualizer = Visualizer(
        100, config.zone_def.number_of_zones
    )

    tlv_frame_parser = TLVFrameParser()
    tlv_frame_parser.register_type(8, heatmap_type := structures.range_azimuth_heatmap_tlv(
        config.num_range_bins, config.num_angle_bins, config.gui_monitor.heatmap_dtype
    ))
    tlv_frame_parser.register_type(9, decision_type := structures.decision_vector_tlv(config.zone_def.number_of_zones))
    tlv_frame_parser.register_type(
        10, vital_signs_type := structures.vital_signs_vector_tlv(config.zone_def.number_of_zones)
    )

    delay = config.frame_cfg.frame_periodicity_in_ms / 1000 if args.mode == "file" else None
    visualizer.start()
    queue = Queue()
    read_thread = Thread(target=read_packet_to_queue, args=(data_stream, tlv_frame_parser, delay, queue))
    read_thread.start()
    while read_thread.is_alive():
        frame = queue.get()
        accept_frame(frame)


def read_packet_to_queue(data_stream, tlv_frame_parser, delay, queue: Queue):
    with args.binary_packet_file:
        for frame in from_stream(data_stream, tlv_frame_parser, delay=delay):
            queue.put(frame)


def accept_frame(frame: TLVFrame):
    logger.info(f"get {len(frame)} tlvs from a frame")
    if args.should_store_packets:
        args.binary_packet_file.write(frame.raw_data)
    decision = [False, False]
    for tlv in frame:
        if isinstance(tlv, heatmap_type):
            logger.debug("updating heatmap")
            visualizer.set_heatmap(tlv)

        elif isinstance(tlv, decision_type):
            decision = list(tlv)
            logger.debug(f"updating decision {decision}")
            visualizer.set_decision(tlv)
        elif isinstance(tlv, vital_signs_type):
            logger.debug(f"updating vital_signs {[v for i, v in enumerate(tlv) if decision[i]]}")
            visualizer.set_vital_signs(tlv)
    with test_time("update"):
        visualizer.update()


class Visualizer:
    vital_signs_info: np.ndarray

    max_signal_len: int
    num_zones: int

    decision: "decision_type" = None
    vital_signs: "vital_signs_type" = None
    heatmap: np.ndarray = None

    plot_names = ["heart_rate", "heart_waveform", "breathing_rate", "breathing_waveform"]

    fig: plt.Figure
    fig_info: plt.Figure
    fig_info_plots: dict[str, list["PlotUpdater"]]

    fig_heatmap: plt.Figure
    heatmap_updater: "HeatmapUpdater"

    def __init__(self, max_signal_len: int, num_zones: int):
        self.max_signal_len = max_signal_len
        self.num_zones = num_zones
        self.fig = plt.figure(figsize=(13, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2])
        self.fig_info = self.fig.add_subfigure(gs[0])
        self.fig_heatmap = self.fig.add_subfigure(gs[1])
        self._init_info_plots()
        self._init_heatmap()

    def _init_info_plots(self):
        self.fig_info_plots = {}
        gs: plt.GridSpec = self.fig_info.add_gridspec(
            5, self.num_zones,
            height_ratios=[0.2, 1, 1, 1, 1],
            hspace=0.5, wspace=0.3
        )
        for i in range(self.num_zones):
            text_ax: plt.Axes = self.fig_info.add_subplot(gs[0, i])
            text_ax.set_axis_off()
            text_ax.text(
                .5, .5, f"Zone {i}",
                verticalalignment="center",
                horizontalalignment="center",
                fontsize="x-large"
            )
            for j, plot_name in enumerate(self.plot_names):
                ax: plt.Axes = self.fig_info.add_subplot(gs[j + 1, i])
                ax.set_title(plot_name)
                ax.set_xticks([])
                line, = ax.plot(np.zeros(self.max_signal_len))
                self.fig_info_plots.setdefault(plot_name, []).append(
                    PlotUpdater(plot_name, line)
                )

    def _init_heatmap(self) -> None:
        canvas: plt.FigureCanvasBase = self.fig_heatmap.canvas
        motion_notify_event = canvas.callbacks.callbacks["motion_notify_event"]
        if len(motion_notify_event) > 0:
            canvas.mpl_disconnect(list(motion_notify_event.keys())[0])
        gs = self.fig_heatmap.add_gridspec(1, 1, left=0, right=0.85)
        ax: plt.PolarAxes = self.fig_heatmap.add_subplot(gs[0], polar=True)
        ax.set_theta_zero_location("N")
        ax.set_thetalim(*np.deg2rad([-60, 60]))
        ax.set_xlabel("Azimuth")
        ax.yaxis.set_label_text("Range", y=0.4, verticalalignment="center")
        azm, rad = np.meshgrid(
            azm_space := np.linspace(np.deg2rad(-60), np.deg2rad(60), config.num_angle_bins),
            rad_space := np.linspace(0, 3, config.num_range_bins)
        )
        azm_block_size = azm_space[1] - azm_space[0]
        rad_block_size = rad_space[1] - rad_space[0]

        zone_rects = []

        mesh: QuadMesh = ax.pcolormesh(azm, rad, np.zeros((config.num_range_bins, config.num_angle_bins)))
        for zone in config.zone_def.zones:
            rect = Rectangle(
                (mesh.convert_xunits(azm_space[zone.azimuth_start]), mesh.convert_yunits(rad_space[zone.range_start])),
                zone.azimuth_length * azm_block_size,
                zone.range_length * rad_block_size,
                fill=False,
                linewidth=1
            )
            ax.add_patch(rect)
            zone_rects.append(rect)

        self.heatmap_updater = HeatmapUpdater(
            mesh,
            zone_rects,
            self.max_signal_len,
            1000
        )

    def start(self) -> None:
        self.fig.show()

    def set_decision(self, decision: "decision_type") -> None:
        self.decision = decision

    def set_vital_signs(self, vital_signs: "vital_signs_type") -> None:
        self.vital_signs = vital_signs

    def set_heatmap(self, heatmap: "heatmap_type") -> None:
        self.heatmap = np.asarray(heatmap)

    def update(self):
        for zone, (zone_decision, zone_vital_signs) in enumerate(zip(self.decision, self.vital_signs)):
            for plot_name in self.plot_names:
                self.fig_info_plots[plot_name][zone].update(
                    getattr(zone_vital_signs, plot_name) if zone_decision else 0,
                    bool(zone_decision)
                )

        self.heatmap_updater.update(self)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class PlotUpdater:
    name: str
    line: plt.Line2D
    axes: plt.Axes

    def __init__(self, name: str, line: plt.Line2D):
        self.name = name
        self.line = line
        self.axes = line.axes

    def update(self, new_value: float, decision: bool) -> None:
        data = self.line.get_ydata().copy()
        data[0: -1] = data[1:]
        data[-1] = new_value
        self.line.set_ydata(data)
        self.line.set_label(f'{self.name}:{new_value:.2f}')
        self.axes.set_title(self.name, {'color': "g" if decision else "r"})
        self.axes.legend(loc="upper left")
        self.axes.relim()
        self.axes.autoscale_view(scalex=False)


class HeatmapUpdater:
    heatmap_artist: QuadMesh
    zone_rects: list[Rectangle]
    axes: plt.Axes
    max_values: deque
    default_vmax: float

    def __init__(self, heatmap_artist: QuadMesh, zone_rects: list[Rectangle], max_signal_len: int, default_vmax: float):
        self.heatmap_artist = heatmap_artist
        self.zone_rects = zone_rects
        self.axes = heatmap_artist.axes
        self.max_values = deque([0], maxlen=max_signal_len)
        self.default_vmax = default_vmax

    def update(self, context: "Visualizer"):
        self.max_values.popleft()
        self.max_values.append(np.max(context.heatmap))
        rolling_avg = sum(self.max_values) / self.max_values.maxlen
        self.heatmap_artist.set_array(context.heatmap)
        self.heatmap_artist.set_clim(0, rolling_avg if rolling_avg > self.default_vmax else self.default_vmax)
        for decision, zone_rect in zip(context.decision, self.zone_rects):
            zone_rect.set_edgecolor("g" if decision else "r")


def load_and_send_config(config_file: Path, cli_port: Optional[serial.Serial]):
    global config
    logger.info("Sending config...")
    with config_file.open() as fp:
        lines: list[str] = list(fp.readlines())
        config = Config(MMWaveConfigLoader(lines))
        if cli_port is None:
            return

        for line in lines:
            line = line.strip()
            if len(line.replace("\t", "").replace(" ", "")) == 0:
                continue
            if line.startswith("%"):
                continue
            cli_port.write(line.encode() + b'\n')
            for _ in range(3):
                response = cli_port.readline()
                logger.info(f"cli: {response.strip().decode()}")
                if b'Done' in response:
                    break
                if b'not recognized as a CLI command' in response:
                    exit(1)
                if b'Error' in response:
                    exit(1)
    return config


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


class ArgConfig(cfg.BaseConfig):
    mode: str = cfg.Lazy(lambda c: "serial" if c.file is None else "file")
    cli_port: Optional[str] = cfg.Option(nullable=True, type=str)
    data_port: Optional[str] = cfg.Option(nullable=True, type=str)
    _store_packets: Optional[Path] = cfg.Option(name="store_packets", nullable=True, type=Path)

    should_store_packets: bool = cfg.PlaceHolder()
    binary_packet_file: Optional[IO] = cfg.PlaceHolder()

    file: Path = cfg.Option(nullable=True, type=Path)
    config: Path = cfg.Option(required=True, type=Path)

    def post_load(self):
        self.should_store_packets = self._store_packets is not None
        if self.should_store_packets:
            self._store_packets.parent.mkdir(parents=True, exist_ok=True)
            self.binary_packet_file = self._store_packets.open("wb")
        else:
            self.binary_packet_file = BytesIO()  # dummy


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=Path)
    mode = parser.add_subparsers(title="mode", required=True)
    serial_mode = mode.add_parser("serial")
    serial_mode.add_argument("cli_port", type=str)
    serial_mode.add_argument("data_port", type=str)
    serial_mode.add_argument("--store-packets", type=Path)

    file_mode = mode.add_parser("file")
    file_mode.add_argument("file", type=Path)

    return parser


if __name__ == '__main__':
    exit(main())
