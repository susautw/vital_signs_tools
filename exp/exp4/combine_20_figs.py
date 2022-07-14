from pathlib import Path

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.h5_to_image import MapType, h5_to_images, TYPE_NAME_SUFFIX_MAP
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
    MapType.Full: (0, 1, 0, 1),
    MapType.PolarFull: (0, 1, 0.25, 0.75)
}

H: dict[MapType, int] = {}
W: dict[MapType, int] = {}
idx_slice: dict[MapType, tuple[slice]] = {}


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
        for i, images in enumerate(h5_to_images(
                fp, config, MAP_TYP, False, False,
                initialize_hook=fig_init_hook,
                skip=SKIP

        )):
            if i >= LENGTH:
                break
            for typ, image in images.items():
                if typ not in combined_images:
                    w_start = w_end = h_start = h_end = 0  # TODO limit every fig's w and h
                    combined_images[typ] = np.zeros((H[typ] * SHAPE[0], W[typ] * SHAPE[1], 3), dtype=np.uint8)
                y, x = np.unravel_index(i, SHAPE)
                x *= W[typ]
                y *= H[typ]
                combined_images[typ][y: y + H[typ], x: x + W[typ]] = image
        out_path = (OUT_BASE_DIR / path.relative_to(SOURCE_BASE_DIR)).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for typ, combined_image in combined_images.items():
            cv2.imwrite(
                str(out_path.with_stem(f'{out_path.stem}_{TYPE_NAME_SUFFIX_MAP[typ]}_combined')),
                combined_image
            )


def fig_init_hook(figs: dict[MapType, plt.Figure], axs: dict[MapType, plt.Axes]) -> None:
    global W, H
    for ax in axs.values():
        ax.set_axis_off()
    for typ, fig in figs.items():
        fig.set_size_inches(*TYPE_FIG_SIZE_MAP[typ])
        w, h = fig.canvas.get_width_height()
        w_s, w_e, h_s, w_e = TYPE_FIG_CONTENT_RANGE_MAP[typ]
        # TODO init idx_slice here
        fig.tight_layout(pad=0)


if __name__ == '__main__':
    main()
