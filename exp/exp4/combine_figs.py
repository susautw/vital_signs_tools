from pathlib import Path

import cv2
import h5py
import numpy as np

from config_loader import MMWaveConfigLoader
from exposes import HFInitHook
from occupancy_and_vital_signs_detection.h5_to_image import MapType, h5_to_images, TYPE_NAME_SUFFIX_MAP, MapSourceType
from occupancy_and_vital_signs_detection.main import Config

BASE_DIR = Path(__file__).parent
DS_DIR = BASE_DIR.parent.parent / "datasets"

SOURCE_BASE_DIR = DS_DIR / "ds3" / "h5_out"
OUT_BASE_DIR = BASE_DIR / "out_9_fig"
NOISE_REMOVED_OUT_BASE_DIR = BASE_DIR / "out_9_noise_removed_fig"

MAP_TYP = MapType.Full | MapType.PolarFull

SKIP = 50
SEARCH_MAX_RANGE = 10
SHAPE = (3, 3)
LENGTH = np.array(SHAPE).prod()


def main():
    """
    Combine 20 figures into a 5x4 grids in an image
    """
    config = Config(MMWaveConfigLoader((BASE_DIR / 'vod_vs_68xx_10fps_center.cfg').read_text().split("\n")))
    h5_paths = sorted(SOURCE_BASE_DIR.glob("**/*.h5"))
    for path in h5_paths:
        print(path)
        with h5py.File(path) as fp:
            process_one(path, config, fp, True)
            process_one(path, config, fp, False)
    print("Done!")


def process_one(path: Path, config: Config, fp: h5py.File, remove_noise: bool) -> None:
    combined_images = {}
    full_heatmaps = np.asarray(fp[MapSourceType.Full.value][SKIP: SKIP + LENGTH])
    skip = SKIP + full_heatmaps.sum(axis=(1, 2)).argmax()
    print(f"{skip=}")
    hook = HFInitHook(remove_noise)
    for i, images in enumerate(h5_to_images(
            fp, config, MAP_TYP, False, False,
            initialize_hook=hook,
            skip=skip
    )):
        if i >= LENGTH:
            break
        for typ, image in images.items():
            w, h = hook.size[typ]
            if typ not in combined_images:
                combined_images[typ] = np.zeros((h * SHAPE[0], w * SHAPE[1], 3), dtype=np.uint8)
            x, y = hook.size[typ] * np.unravel_index(i, SHAPE)[::-1]
            combined_images[typ][y: y + h, x: x + w] = image[hook.content_range_idx[typ]]

    base_dir = NOISE_REMOVED_OUT_BASE_DIR if remove_noise else OUT_BASE_DIR
    out_path = (base_dir / path.relative_to(SOURCE_BASE_DIR)).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for typ, combined_image in combined_images.items():
        cv2.imwrite(
            str(out_path.with_stem(f'{modify_path(out_path.stem)}_{TYPE_NAME_SUFFIX_MAP[typ]}_combined')),
            combined_image
        )


def modify_path(stem: str) -> str:
    stem = stem.split(".", maxsplit=1)[0]
    angle = "0"
    distance = "90"
    if stem.startswith(("+", "-")):  # angle
        angle = stem[:-1]
    else:  # distance
        distance = stem[:-1]

    if stem[-1] == "h":
        direction = "horizontal"
    elif stem[-1] == "v":
        direction = "vertical"
    else:
        raise ValueError("invalid path stem")

    return f'{direction}_{distance}_{angle}'


if __name__ == '__main__':
    main()
