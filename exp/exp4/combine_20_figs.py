from pathlib import Path

import cv2
import h5py
import numpy as np

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.h5_to_image import MapType, h5_to_images, TYPE_NAME_SUFFIX_MAP, \
    MapSourceType, HeatmapFigureIterator
from occupancy_and_vital_signs_detection.main import Config

BASE_DIR = Path(__file__).parent
SOURCE_BASE_DIR = BASE_DIR / "out"
OUT_BASE_DIR = BASE_DIR / "out_20_fig"

MAP_TYP = MapType.Full | MapType.PolarFull

SKIP = 50
LENGTH = 20
SHAPE = (4, 5)
TYPE_FIG_SIZE_MAP = {
    MapType.Full: (6, 6),
    MapType.PolarFull: (6, 6)
}

TYPE_FIG_CONTENT_RANGE_MAP = {  # (W_start, W_end, H_start, H_end)
    MapType.Full: np.array([0, 1, 0, 1]),
    MapType.PolarFull: np.array([0, 1, 0.21, 0.79])
}

H: dict[MapType, int] = {}
W: dict[MapType, int] = {}
idx_slice: dict[MapType, tuple[slice, ...]] = {}


def main():
    """
    Combine 20 figures into a 5x4 grids in an image
    """
    config = Config(MMWaveConfigLoader((BASE_DIR / 'vod_vs_68xx_10fps_center.cfg').read_text().split("\n")))
    h5_paths = sorted(SOURCE_BASE_DIR.glob("**/*.h5"))
    for path in h5_paths:
        print(path)
        process_one(path, config)
    print("Done!")


def process_one(path: Path, config: Config) -> None:
    combined_images = {}
    with h5py.File(path) as fp:
        full_heatmaps = np.asarray(fp[MapSourceType.Full.value][SKIP: SKIP + LENGTH])
        skip = SKIP + full_heatmaps.sum(axis=(1, 2)).argmax()
        print(f"{skip=}")
        for i, images in enumerate(h5_to_images(
                fp, config, MAP_TYP, False, False,
                initialize_hook=fig_init_hook,
                skip=skip

        )):
            if i >= LENGTH:
                break
            for typ, image in images.items():
                if typ not in combined_images:
                    combined_images[typ] = np.zeros((H[typ] * SHAPE[0], W[typ] * SHAPE[1], 3), dtype=np.uint8)
                y, x = np.unravel_index(i, SHAPE)
                x *= W[typ]
                y *= H[typ]
                combined_images[typ][y: y + H[typ], x: x + W[typ]] = image[idx_slice[typ]]
        out_path = (OUT_BASE_DIR / path.relative_to(SOURCE_BASE_DIR)).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for typ, combined_image in combined_images.items():
            cv2.imwrite(
                str(out_path.with_stem(f'{out_path.stem}_{TYPE_NAME_SUFFIX_MAP[typ]}_combined')),
                combined_image
            )


def fig_init_hook(hf_iter: HeatmapFigureIterator) -> None:
    global W, H

    for typ, figure_collection in hf_iter.figure_collections.items():
        figure_collection.ax.set_axis_off()
        fig = figure_collection.figure
        fig.set_size_inches(*TYPE_FIG_SIZE_MAP[typ])
        size = np.asarray(fig.canvas.get_width_height())
        w_s, w_e, h_s, h_e = np.int32(TYPE_FIG_CONTENT_RANGE_MAP[typ] * size[[0, 0, 1, 1]])
        idx_slice[typ] = slice(h_s, h_e), slice(w_s, w_e)
        W[typ] = w_e - w_s
        H[typ] = h_e - h_s
        fig.tight_layout(pad=0)


if __name__ == '__main__':
    main()
