import argparse
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from config_loader import MMWaveConfigLoader
from utility import structure_to_dict
from .main import Config, get_parser

from tlv import from_stream, TLVFrame

config: Config
args: argparse.Namespace


def packet_to_h5(_args: Sequence[str] = None):
    """
    get data from packet binary file then store it to hdf5 file
    """
    global args, config

    args = get_arg_parser().parse_args(_args if len(_args) > 0 else None)
    source: Path = args.source
    with args.config.open() as fp:
        config = Config(MMWaveConfigLoader(fp.readlines()))
    args.output.mkdir(parents=True, exist_ok=True)

    parser = get_parser(config)

    file_paths = sorted(source.glob(args.pattern))

    for file_path in file_paths:
        print(file_path)
        with FrameAcceptor(file_path) as acceptor, file_path.open("rb") as fp:
            for frame in from_stream(fp, parser):
                acceptor.accept(frame)


class FrameAcceptor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        output_path = (args.output / file_path.relative_to(args.source)).with_suffix(".h5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_ds = h5py.File(output_path, "w")
        self.heatmap_group = self.output_ds.require_group("heatmap")
        self.metadata_group = self.output_ds.require_group("metadata")
        self.full_heatmap = None
        self.zone_heatmaps = {}

    def accept(self, frame: TLVFrame):
        from .main import heatmap_type
        metadata = self.metadata_group.create_dataset(
            f"{frame.frame_header.frame_number}",
            shape=()
        )
        for name, value in structure_to_dict(frame.frame_header).items():
            metadata.attrs[f'frame.{name}'] = value
        for tlv in frame:
            if isinstance(tlv, heatmap_type):
                heatmap = np.asarray(tlv)
                if self.full_heatmap is None:
                    self.full_heatmap = heatmap.reshape(1, *heatmap.shape)
                else:
                    self.full_heatmap = np.vstack([self.full_heatmap, heatmap.reshape(1, *heatmap.shape)])
                for i, zone in enumerate(config.zone_def.zones):
                    cropped = heatmap[zone.idx_slice]
                    cropped = cropped.reshape(1, *cropped.shape)
                    if i not in self.zone_heatmaps:
                        self.zone_heatmaps[i] = cropped
                    else:
                        self.zone_heatmaps[i] = np.vstack([self.zone_heatmaps[i], cropped])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return
        self.heatmap_group.create_dataset(
            name="full",
            data=self.full_heatmap,
            compression=9
        )
        for zone_idx, zone_heatmap in self.zone_heatmaps.items():
            self.heatmap_group.create_dataset(
                name=f'zone{zone_idx}',
                data=zone_heatmap,
                compression=9
            )
        self.output_ds.close()


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="binary file source, can be file or directory")
    parser.add_argument("-p", "--pattern", type=str, default="**/*.bin")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("config", type=Path, help="path to config file")
    return parser


if __name__ == '__main__':
    packet_to_h5()
