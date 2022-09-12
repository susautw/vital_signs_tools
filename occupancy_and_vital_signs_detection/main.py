import argparse
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import IO, Optional, Sequence, Any

import serial
from fancy import config as cfg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle
import matplotlib as mpl

from config_loader import MMWaveConfigLoader
from ovsd.configs import OVSDConfig
from ovsd import structures
from occupancy_and_vital_signs_detection.plots import HeatmapPlotter
from tlv import from_stream, TLVFrame, FrameHeader
from utility import RollingAverage

args: "ArgConfig"
config: "OVSDConfig"
visualizer: "Visualizer"

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

mpl.rcParams['font.family'] = ['Microsoft YaHei', 'sans-serif']
QUEUE_WARNING_RATIO = 0.8
finalized = False


def main():
    global args, visualizer

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
    logger.info(f"config has loaded.")

    tlv_frame_parser = structures.get_parser(config)
    delay = config.frame_cfg.frame_periodicity_in_ms / 1000 if args.mode == "file" else None

    queue = Queue(maxsize=args.queue_size)
    read_thread = Thread(target=read_packet_to_queue, args=(data_stream, tlv_frame_parser, delay, queue), daemon=True)

    visualizer = Visualizer(
        100, config.zone_def.number_of_zones,
    )

    finalizer = get_finalizer(data_stream, cli_port, visualizer.fig.canvas)

    visualizer.start()
    logger.info("visualizer has start")
    read_thread.start()
    logger.info("read_thread has start")

    try:
        while read_thread.is_alive() and not finalized:
            try:
                frame, skip = queue.get(timeout=1)
                accept_frame(frame, skip)
                if (ratio := queue.qsize() / queue.maxsize) > QUEUE_WARNING_RATIO:
                    logger.warning(f'frame {frame.frame_header.frame_number} / high queue_size_ratio: {ratio:.2%}')
                else:
                    logger.info(f'frame {frame.frame_header.frame_number}')
            except Empty:
                pass
    except KeyboardInterrupt:
        logger.info("Interrupted!")
    finally:
        finalizer(None)
    logger.info("Done.")


def read_packet_to_queue(data_stream, tlv_frame_parser, delay, q: Queue):
    skip = 0
    with args.binary_packet_file:
        for frame in from_stream(data_stream, tlv_frame_parser, delay=delay):
            if skip > 0:
                skip -= 1
                q.put((frame, True))
                continue
            queue_size_ratio = q.qsize() / args.queue_size
            for ratio in args.packet_loss_threshold:
                if queue_size_ratio > ratio:
                    skip += args.packet_loss_threshold[ratio]
                    break
            q.put((frame, False))


def accept_frame(frame: TLVFrame, skip: bool):
    logger.debug(f"get {len(frame)} tlvs from a frame")
    if args.should_store_packets:
        args.binary_packet_file.write(frame.raw_data)
    if skip:
        return
    visualizer.set_frame_header(frame.frame_header)
    for tlv in frame:
        if isinstance(tlv, structures.heatmap_type):
            logger.debug("updating heatmap")
            visualizer.set_heatmap(tlv)
        elif isinstance(tlv, structures.decision_type):
            logger.debug(f"updating decision")
            visualizer.set_decision(tlv)
        elif isinstance(tlv, structures.vital_signs_type):
            logger.debug(f"updating vital_signs")
            visualizer.set_vital_signs(tlv)
    visualizer.update()


def get_finalizer(data_stream: IO, cli_port, canvas: plt.FigureCanvasBase):
    def finalize(_event):
        global finalized
        if finalized:
            return
        finalized = True
        canvas.close_event()
        if cli_port is not None:
            cli_port.write(b"sensorStop\n")
            cli_port.close()
        data_stream.close()

    canvas.mpl_connect("close_event", finalize)
    return finalize


class Visualizer:
    vital_signs_info: np.ndarray

    max_signal_len: int
    num_zones: int

    frame_header: FrameHeader = None
    decision: "structures.decision_type" = None
    vital_signs: "structures.vital_signs_type" = None
    heatmap: np.ndarray = None

    plot_names = ["heart_rate", "heart_waveform", "breathing_rate", "breathing_waveform"]

    fig: plt.Figure
    fig_info: plt.Figure
    fig_heatmap: plt.Figure
    fig_frame_info: plt.Figure
    plot_updaters: list["IPlotUpdater"]

    bg_cache: Any

    def __init__(self, max_signal_len: int, num_zones: int):
        self.max_signal_len = max_signal_len
        self.num_zones = num_zones
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[3, 2], height_ratios=[0.05, 1, 0.05])

        title_ax = self.fig.add_subfigure(gs[0, :]).add_subplot(111)
        title_ax.set_axis_off()
        title_ax.text(0.5, 0, "靜宜大學人工智慧實驗室", fontsize=22, horizontalalignment="center")

        self.fig_info = self.fig.add_subfigure(gs[1, 0])
        self.fig_heatmap = self.fig.add_subfigure(gs[1, 1])
        self.fig_frame_info = self.fig.add_subfigure(gs[2, :])

        self.plot_updaters = []
        self._init_info_plots()
        self._init_heatmap()
        self._init_frame_info()

        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _init_info_plots(self):
        gs: plt.GridSpec = self.fig_info.add_gridspec(
            5, self.num_zones,
            height_ratios=[0.2, 1, 1, 1, 1],
            hspace=0.5, wspace=0.3
        )
        for i in range(self.num_zones):
            text_ax: plt.Axes = self.fig_info.add_subplot(gs[0, i])
            text_ax.set_axis_off()
            geo_sorted_idx = config.zone_def.geo_sorted_ids[i]
            text_ax.text(
                .5, .5, f"Zone {geo_sorted_idx}",
                verticalalignment="center",
                horizontalalignment="center",
                fontsize="x-large"
            )
            for j, plot_name in enumerate(self.plot_names):
                ax: plt.Axes = self.fig_info.add_subplot(gs[j + 1, i])
                ax.set_title(plot_name)
                ax.set_xticks([])
                line, = ax.plot(np.zeros(self.max_signal_len))
                ax.set_animated(True)
                self.plot_updaters.append(LinePlotUpdater(plot_name, geo_sorted_idx, line))

    def _init_heatmap(self) -> None:
        canvas: plt.FigureCanvasBase = self.fig_heatmap.canvas
        motion_notify_event = canvas.callbacks.callbacks["motion_notify_event"]
        if len(motion_notify_event) > 0:
            canvas.mpl_disconnect(list(motion_notify_event.keys())[0])

        gs = self.fig_heatmap.add_gridspec(2, self.num_zones, left=0, right=0.85, height_ratios=[5, 2], wspace=0.3)
        polar_ax: plt.PolarAxes = self.fig_heatmap.add_subplot(gs[0, :], polar=True)

        zone_axs = {
            zone_idx: self.fig_heatmap.add_subplot(gs[1, config.zone_def.geo_sorted_ids[zone_idx]])
            for zone_idx, zone in enumerate(config.zone_def.zones)
        }

        plotter = HeatmapPlotter(config, polar_full=polar_ax, zones=zone_axs)
        plotter.initialize()

        zone_rects = []
        zone_meshes = []

        for zone_plot in plotter.zone_plots.values():
            zone_plot.rect.set_animated(True)
            zone_plot.mesh.set_animated(True)
            zone_rects.append(zone_plot.rect)
            zone_meshes.append(zone_plot.mesh)

        self.plot_updaters.append(HeatmapUpdater(
            plotter.polar_full_mesh,
            zone_meshes,
            zone_rects,
            rolling_avg_window_size=4,
            default_vmax=1000
        ))

    def _init_frame_info(self) -> None:
        ax = self.fig_frame_info.add_subplot(111)
        ax.set_axis_off()
        text = ax.text(1, 0.5, "", fontsize=16, ha="right", va="center")
        text.set_animated(True)
        self.plot_updaters.append(FrameInfoUpdater(text))

    def start(self) -> None:
        self.fig.show()

    def close(self):
        self.fig.canvas.close_event()

    def set_frame_header(self, frame_header: FrameHeader):
        self.frame_header = frame_header

    def set_decision(self, decision: "structures.decision_type") -> None:
        self.decision = decision

    def set_vital_signs(self, vital_signs: "structures.vital_signs_type") -> None:
        self.vital_signs = vital_signs

    def set_heatmap(self, heatmap: "structures.heatmap_type") -> None:
        self.heatmap = np.asarray(heatmap)

    def update(self):
        artists = []
        for plot_updater in self.plot_updaters:
            artists.extend(plot_updater.update(self))

        self.fig.canvas.restore_region(self.bg_cache)
        artists.sort(key=lambda x: x.get_zorder())
        for a in artists:
            a.axes.draw_artist(a)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        return artists


class IPlotUpdater:
    def update(self, context: Visualizer) -> Sequence[plt.Artist]:
        raise NotImplementedError()


class FrameInfoUpdater(IPlotUpdater):
    def __init__(self, text: plt.Text, time_format: str = None):
        self.time_format = time_format
        self.text = text

    def update(self, context: Visualizer) -> Sequence[plt.Artist]:
        now_time = datetime.now().time()
        if self.time_format is None:
            time_info = f"time={now_time.isoformat('milliseconds')}"
        else:
            time_info = now_time.strftime(self.time_format)
        frame_info = f'frame_number={context.frame_header.frame_number}'
        self.text.set_text(', '.join([time_info, frame_info]))
        return self.text,


class LinePlotUpdater(IPlotUpdater):
    name: str
    line: plt.Line2D
    axes: plt.Axes
    idx: int

    def __init__(self, name: str, idx: int, line: plt.Line2D):
        self.name = name
        self.idx = idx
        self.line = line
        self.axes = line.axes

    def update(self, context: Visualizer) -> Sequence[plt.Artist]:
        decision = context.decision[self.idx]
        new_value = getattr(context.vital_signs[self.idx], self.name) if decision else 0
        data = self.line.get_ydata().copy()
        data[0: -1] = data[1:]
        data[-1] = new_value
        self.line.set_ydata(data)
        self.line.set_label(f'{self.name}:{new_value:.2f}')
        self.axes.set_title(self.name, {'color': "g" if decision else "r"})
        self.axes.legend(loc="upper left")
        self.axes.relim()
        self.axes.autoscale_view(scalex=False)
        return self.axes,


class HeatmapUpdater(IPlotUpdater):
    heatmap_artist: QuadMesh
    zone_heatmap_artists: list[QuadMesh]
    zone_heatmaps_vmax: list[RollingAverage]
    zone_rects: list[Rectangle]
    axes: plt.Axes
    heatmap_vmax: RollingAverage
    artists: list[plt.Artist]

    def __init__(
            self,
            heatmap_artist: QuadMesh,
            zone_heatmap_artists: list[QuadMesh],
            zone_rects: list[Rectangle],
            rolling_avg_window_size: int,
            default_vmax: float
    ):
        self.heatmap_artist = heatmap_artist
        self.zone_heatmap_artists = zone_heatmap_artists
        self.zone_rects = zone_rects
        self.axes = heatmap_artist.axes
        self.heatmap_vmax = RollingAverage(rolling_avg_window_size, low=default_vmax, init=0)
        self.zone_heatmaps_vmax = [
            RollingAverage(rolling_avg_window_size, low=default_vmax, init=0)
            for _ in range(config.zone_def.number_of_zones)
        ]
        self.artists = [self.heatmap_artist, *self.zone_heatmap_artists, *self.zone_rects]

    def update(self, context: Visualizer) -> Sequence[plt.Artist]:
        self.heatmap_artist.set_array(context.heatmap)
        self.heatmap_artist.set_clim(0, self.heatmap_vmax.next(np.max(context.heatmap)))
        for zone, decision, zone_rect, zone_mesh, zone_vmax in zip(
                config.zone_def.zones,
                context.decision,
                self.zone_rects,
                self.zone_heatmap_artists,
                self.zone_heatmaps_vmax
        ):
            zone_heatmap = context.heatmap[zone.idx_slice]
            zone_mesh.set_array(zone_heatmap)
            zone_mesh.set_clim(0, zone_vmax.next(np.max(zone_heatmap)))
            zone_rect.set_edgecolor("g" if decision else "r")

        return self.artists


def load_and_send_config(config_file: Path, cli_port: Optional[serial.Serial]):
    global config
    logger.info("Sending config...")
    with config_file.open() as fp:
        lines: list[str] = list(fp.readlines())
        config = OVSDConfig(MMWaveConfigLoader(lines))
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


class ArgConfig(cfg.BaseConfig):
    mode: str = cfg.Lazy(lambda c: "serial" if c.file is None else "file")
    cli_port: Optional[str] = cfg.Option(nullable=True, type=str)
    data_port: Optional[str] = cfg.Option(nullable=True, type=str)
    _store_packets: Optional[Path] = cfg.Option(name="store_packets", nullable=True, type=Path)
    packet_file_path = cfg.PlaceHolder()

    should_store_packets: bool = cfg.PlaceHolder()
    binary_packet_file: Optional[IO] = cfg.PlaceHolder()

    queue_size: int = cfg.Lazy(lambda c: 10)
    packet_loss_threshold: dict[float: int] = {0.5: 1, 0.7: 2, 0.9: 4}

    file: Path = cfg.Option(nullable=True, type=Path)
    config: Path = cfg.Option(required=True, type=Path)

    def post_load(self):
        self.should_store_packets = self._store_packets is not None
        if self.should_store_packets:
            self._store_packets.parent.mkdir(parents=True, exist_ok=True)
            self.packet_file_path = self._store_packets.with_suffix(
                f'.{datetime.now().strftime("%Y%m%dT%H%M%S")}{self._store_packets.suffix}'
            )
            self.binary_packet_file = self.packet_file_path.open("wb")
        else:
            self.binary_packet_file = BytesIO()  # dummy
        # noinspection PyTypeChecker
        self.packet_loss_threshold = dict(sorted(self.packet_loss_threshold.items(), reverse=True))


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
