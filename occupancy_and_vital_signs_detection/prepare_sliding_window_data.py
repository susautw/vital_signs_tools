import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    """
    prepare the data (two of arrays).
    data: zone size * window size
    label put in attribute
    """
    args = get_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    h5_file_paths = sorted(args.root.glob("**/*.h5"))

    print({i: path for i, path in enumerate(h5_file_paths)})

    h5_file_fps = [h5py.File(path) for path in h5_file_paths]

    modes = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
    output_h5s = {mode_name: args.output / f'{mode_name}.h5' for mode_name in modes}
    output_h5_fps = {mode_name: h5py.File(path, "w") for mode_name, path in output_h5s.items()}
    output_frame_count = {mode_name: 0 for mode_name in modes}

    for i, h5_file_fp in enumerate(h5_file_fps):
        arr = get_array_from_h5(h5_file_fp['heatmap/zone0'])
        for j in range(arr.shape[0] - args.window_size + 1):
            data = arr[j:j + args.window_size]
            target = i
            proportional_store_to_h5(output_frame_count, data, target, modes, output_h5_fps)


def get_array_from_h5(group: h5py.Group) -> np.ndarray:
    return np.array([data[1] for data in sorted(group.items(), key=lambda x: int(x[0]))])


def proportional_store_to_h5(
        frame_count: dict[str, int],
        data: np.ndarray,
        target: int,
        ratios: dict[str, float],
        fps: dict[str, h5py.File]
) -> str:
    mode = np.random.choice(list(ratios.keys()), p=list(ratios.values()))
    ds = fps[mode].create_dataset(f'{frame_count[mode]:06d}', data=data, compression=9)
    ds.attrs['target'] = target

    frame_count[mode] += 1
    return mode


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="root directory of H5s")
    parser.add_argument("output", type=Path, help="output directory of train / validation / test file")
    parser.add_argument("-w", "--window-size", type=int)
    return parser


if __name__ == '__main__':
    main()
