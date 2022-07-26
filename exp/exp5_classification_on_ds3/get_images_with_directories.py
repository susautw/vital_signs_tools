from itertools import tee
from pathlib import Path
from typing import Iterator

import cv2
import h5py
import numpy as np

import iter_utils
from datasets.ds3 import modify_path_stem

from config_loader import MMWaveConfigLoader
from exp.exp4.exposes import RemoveNoiseDecorator, HFInitHook
from occupancy_and_vital_signs_detection.h5_to_image import MapType, MapSourceType, HeatmapFigureIterator, \
    MeshUpdater, AveragingLimitDecorator, rolling_average_factory, get_figure_collections, FigureCollection
from occupancy_and_vital_signs_detection.main import Config
from utility import convert_fig_to_image, combine_images

BASE_DIR = Path(__file__).parent
DS_D3_DIR = BASE_DIR.parent.parent / "datasets" / "ds3"

SOURCE_BASE_DIR = DS_D3_DIR / "h5_out"
OUT_BASE_DIR = BASE_DIR / "outs"

SKIP = 50
SHAPE = (3, 3)
SEARCH_RANGE = 10
SEARCH_INTERVAL = int(np.asarray(SHAPE).prod())


def main():
    config = Config(MMWaveConfigLoader((DS_D3_DIR / 'vod_vs_68xx_10fps_center.cfg').read_text().split("\n")))
    h5_paths = sorted(SOURCE_BASE_DIR.glob("**/*.h5"))
    processor = FileProcessor(config)
    for h5_path in h5_paths:
        print(h5_path)
        processor.process(h5_path)
    print("Done")


class FileProcessor:
    def __init__(self, config: Config):
        self.configurators = {
            "normal": AveragingLimitDecorator(MeshUpdater(), rolling_average_factory),
            "noise_removed": AveragingLimitDecorator(RemoveNoiseDecorator(MeshUpdater()), rolling_average_factory)
        }

        self.figure_collections = {
            key: get_figure_collections({MapType.Full}, config, False) for key in self.configurators
        }

        self.profiles = self.configurators.keys()

    def process(self, path):
        with h5py.File(path) as fp:
            sources_iter = get_peak_aligned_sources_iter(fp, SKIP, SEARCH_RANGE, SEARCH_INTERVAL)
            hf_its = [
                HeatmapFigureIterator(
                    self.configurators[profile], self.figure_collections[profile], source_it
                )
                for profile, source_it in zip(self.profiles, tee(sources_iter, len(self.profiles)))
            ]
            hooks = []
            combined_image_its = []
            for hf_it in hf_its:
                hook = HFInitHook(apply_noise_remove=False)  # noise removal has been applied before.
                hook(hf_it)
                hooks.append(hook)
                combined_image_its.append(map(
                    lambda images: combine_images(hook.size[MapType.Full], SHAPE, images, 3),
                    map(
                        lambda images: [image[hook.content_range_idx[MapType.Full]] for image in images],
                        iter_utils.pack(map(to_full_image, hf_it), SEARCH_INTERVAL)
                    )
                ))

            save_dirs = self.get_and_create_save_dirs(path)

            for i, combined_images in enumerate(zip(*combined_image_its)):
                for profile, combined_image in zip(self.profiles, combined_images):
                    cv2.imwrite(str(save_dirs[profile] / f"{save_dirs[profile].name}_{i}.png"), combined_image)

    def get_and_create_save_dirs(self, path: Path) -> dict[str, Path]:
        save_paths = {
            profile: (
                    OUT_BASE_DIR / profile / path.relative_to(SOURCE_BASE_DIR)
            ).with_name(modify_path_stem(path.stem))
            for profile in self.profiles
        }
        for profile, save_path in save_paths.items():
            save_path = save_path.parents[1] / save_path.name
            save_path.mkdir(exist_ok=True, parents=True)
            save_paths[profile] = save_path
        return save_paths


def to_full_image(figure_collections: dict[MapType, FigureCollection]) -> np.ndarray:
    return convert_fig_to_image(figure_collections[MapType.Full].figure, draw=True)


def get_peak_aligned_sources_iter(
        h5_heatmaps: h5py.File,
        skip: int,
        search_range: int,  # range of peak finding
        search_interval: int  # interval between two searching
) -> Iterator[dict[MapSourceType, np.ndarray]]:
    full_heatmaps = np.asarray(h5_heatmaps[MapSourceType.Full.value])
    sum_of_heatmaps = np.sum(full_heatmaps, axis=(1, 2))
    len_heatmap = len(full_heatmaps)
    idx = skip
    interval_count = 0
    while idx < len_heatmap:
        if interval_count == 0:
            idx = idx + sum_of_heatmaps[idx: idx + search_range].argmax()
            if idx + search_interval >= len_heatmap:
                break
        yield {MapSourceType.Full: full_heatmaps[idx]}
        idx += 1
        interval_count = (interval_count + 1) % search_interval


if __name__ == '__main__':
    main()
