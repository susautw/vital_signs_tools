import argparse
from pathlib import Path

import h5py
import numpy as np

from config_loader import MMWaveConfigLoader
from main import Config, get_parser

from tlv import from_stream, TLVFrame

config: Config
args: argparse.Namespace


def main():
    """
    get data from packet binary file then store it to hdf5 file
    """
    global args, config

    args = get_arg_parser().parse_args()
    source: Path = args.source
    with args.config.open() as fp:
        config = Config(MMWaveConfigLoader(fp.readlines()))
    args.output.mkdir(parents=True, exist_ok=True)

    parser = get_parser(config)

    file_paths = sorted(source.glob("**/*.bin"))

    for file_path in file_paths:
        print(file_path)
        with FrameAcceptor(file_path) as acceptor, file_path.open("rb") as fp:
            for frame in from_stream(fp, parser):
                acceptor.accept(frame)


class FrameAcceptor:

    def __init__(self, file_path: Path):
        self.file_path = file_path
        output_path = (args.output / file_path.relative_to(args.source)).with_suffix(".h5")
        self.output_ds = h5py.File(output_path, "w")
        self.heatmap_group = self.output_ds.require_group("heatmap")
        self.zone_groups = [
            self.heatmap_group.require_group(f"zone{i}") for i in range(config.zone_def.number_of_zones)
        ]

    def accept(self, frame: TLVFrame):
        from main import heatmap_type
        for tlv in frame:
            if isinstance(tlv, heatmap_type):
                heatmap = np.asarray(tlv)
                for i, zone in enumerate(config.zone_def.zones):
                    cropped = heatmap[zone.idx_slice]
                    self.zone_groups[i].create_dataset(
                        name=str(frame.frame_header.frame_number),
                        data=cropped,
                        compression=9
                    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output_ds.close()


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="binary file source, can be file or directory")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("config", type=Path, help="path to config file")
    return parser


if __name__ == '__main__':
    main()
