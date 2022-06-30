import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.main import Config, get_parser
from occupancy_and_vital_signs_detection.plots import HeatmapPlotter

from tlv import from_stream, TLVFrame
from utility import RollingAverage

configs: dict[str, Config] = {}
args: argparse.Namespace
DEFAULT_CONFIG = "30cm"


def main():
    """
    get data from packet binary file then store it to hdf5 file
    """
    global args, configs

    args = get_arg_parser().parse_args()
    source: Path = args.source
    config_files = {
        '30cm': Path('exp/x49-az-heatmap-data/vod_vs_68xx_10fps_center_30.cfg'),
        '60cm': Path('exp/x49-az-heatmap-data/vod_vs_68xx_10fps_center_60.cfg'),
        '90cm': Path('exp/x49-az-heatmap-data/vod_vs_68xx_10fps_center_90.cfg'),
    }

    for profile, config_file in config_files.items():
        with config_file.open() as fp:
            configs[profile] = Config(MMWaveConfigLoader(fp.readlines()))
    plots = init_plots()
    args.output.mkdir(parents=True, exist_ok=True)

    default_config = configs[DEFAULT_CONFIG]
    parser = get_parser(default_config)

    num_skip_frames = int(6000 / default_config.frame_cfg.frame_periodicity_in_ms)  # skip 0~6 second
    num_frame_store = 49

    file_paths = sorted(source.glob("**/*.bin"))
    acceptor = FrameAcceptor(plots, num_skip_frames, num_frame_store)

    for file_path in file_paths:
        acceptor.set_file_path(file_path)
        print(file_path)
        with file_path.open("rb") as fp:
            for frame in from_stream(fp, parser):
                if not acceptor.accept(frame):
                    break
        acceptor.reset()


class FrameAcceptor:
    file_path = None
    zone_profile = None
    output_path = None
    count = 0
    heatmap_vmax: RollingAverage
    num_skip_frames: int
    num_frame_store: int

    def __init__(
            self,
            plots: dict[str, tuple[Figure, QuadMesh]],
            num_skip_frames: int,
            num_frame_store: int
    ):
        self.plots = plots
        self.heatmap_vmax = RollingAverage(4, low=1000, init=0)
        self.num_skip_frames = num_skip_frames
        self.num_frame_store = num_frame_store

    def set_file_path(self, file_path: Path):
        self.file_path = file_path
        self.zone_profile = file_path.stem[:4]

    def step(self):
        self.count += 1

    def reset(self):
        self.count = 0
        self.heatmap_vmax.reset()

    def accept(self, frame: TLVFrame) -> bool:
        from occupancy_and_vital_signs_detection.main import heatmap_type
        if frame.frame_header.frame_number < self.num_skip_frames:
            return True
        for tlv in frame:
            if isinstance(tlv, heatmap_type):
                heatmap = np.asarray(tlv)
                self.heatmap_vmax.next(np.max(heatmap))
                process_plots(heatmap, self.plots, self.file_path.name[:4], self)
        self.count += 1
        if self.count > self.num_frame_store:
            return False
        return True


def init_plots() -> dict[str, tuple[Figure, QuadMesh]]:
    result = {}
    angular_full_fig: Figure = plt.figure()
    full_fig: Figure = plt.figure()
    zone_figs = {profile: plt.figure() for profile in configs}
    config = configs[DEFAULT_CONFIG]

    angular_full_ax: plt.PolarAxes = angular_full_fig.add_subplot(111, polar=True)
    full_ax: plt.Axes = full_fig.add_subplot(111)

    plotter = HeatmapPlotter(config, full=full_ax, polar_full=angular_full_ax)
    plotter.set_show_rect(False)
    plotter.initialize()

    result['angular_full'] = (angular_full_fig, plotter.polar_full_mesh)
    result['full'] = (full_fig, plotter.full_mesh)

    for profile, fig in zone_figs.items():
        rect_ax = fig.add_subplot(111)
        zone_plotter = HeatmapPlotter(config[profile], zones={0: rect_ax})
        result[profile] = (fig, zone_plotter.zone_plots[0].mesh)
    return result


def process_plots(
        heatmap: np.ndarray,
        plots: dict[str, tuple[Figure, QuadMesh]],
        zone_name: str,
        context: FrameAcceptor
):
    af_fig, af_mesh = plots["angular_full"]
    af_mesh.set_array(heatmap)
    af_mesh.set_clim(0, context.heatmap_vmax.current())
    f_fig, f_mesh = plots["full"]
    f_mesh.set_array(heatmap)
    f_mesh.set_clim(0, context.heatmap_vmax.current())
    zone_fig, zone_mesh = plots[zone_name]
    zone = configs[zone_name].zone_def.zones[0]
    zone_mesh.set_array(heatmap[zone.idx_slice])
    zone_mesh.set_clim(0, context.heatmap_vmax.current())

    figs = {
        "angular_full": af_fig,
        "full": f_fig,
        zone_name: zone_fig
    }

    for profile, fig in figs.items():
        out_path = context.file_path.relative_to(args.source)
        out_path = args.output / profile / out_path.with_stem(
            f'{out_path.stem}_{context.count:04d}'
        ).with_suffix(".png")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path))


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="binary file source, can be file or directory")
    parser.add_argument("output", type=Path, help="output directory")
    return parser


if __name__ == '__main__':
    main()
