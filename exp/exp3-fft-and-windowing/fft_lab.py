from pathlib import Path
from typing import Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.windows import get_window
from scipy.signal import stft

BASE_DIR = Path(__file__).parent
EXP_DIR = BASE_DIR.parent


def main():
    fft_size = 64
    source_root = EXP_DIR / Path("exp2/outs")
    output_base = BASE_DIR / "stft_out_64"
    fs = 1 / 0.107
    frame_to_peek = int(np.ceil((5 / 0.107) / fft_size))

    window_params: list[Union[tuple, str]] = [
        "unwindowed",
        "hann",
        "hamming",
        "blackman",
        "cosine"
    ]

    windows: dict[str, np.ndarray] = {}
    for window_param in window_params:
        name = window_param
        if isinstance(window_param, tuple):
            name = window_param[0]
        if window_param == "unwindowed":
            windows[name] = np.ones(fft_size)
        else:
            windows[name] = get_window(window_param, Nx=fft_size)

    paths = sorted(source_root.glob("**/*.h5"))

    for path in paths:
        print(path)
        ds = h5py.File(path)
        output = output_base / path.relative_to(source_root)
        heatmaps = ds['heatmap/zone0']
        signal = np.asarray(heatmaps).sum(axis=(1, 2))
        plot_stft(signal, windows, fs, fft_size, frame_to_peek, output)


def plot_stft(
        signal: np.ndarray,
        windows: dict[str, np.ndarray],
        fs: float,
        fft_size: int,
        frame_to_peek: int,
        output: Path
):
    fig: plt.Figure = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(len(windows), 3, hspace=0.5, wspace=0.5)
    signal = signal - signal.mean()  # remove DC component

    for i, (name, window) in enumerate(windows.items()):
        f, t, z = stft(signal, fs, window, nperseg=fft_size, detrend=lambda x: x-x.mean())
        window_ax: plt.Axes = fig.add_subplot(gs[i, 0])
        ax: plt.Axes = fig.add_subplot(gs[i, 1])
        peek_ax: plt.Axes = fig.add_subplot(gs[i, 2])

        window_ax.set_title(f"{name}")
        ax.set_title(f'STFT ({name})')
        peek_ax.set_title(f'fft at frame:{frame_to_peek}')

        f *= 60  # sec to min

        magnitude = np.abs(z)

        mesh = ax.pcolormesh(t, f, magnitude, vmin=0, vmax=np.max(magnitude))

        window_ax.plot(window)

        ax.set_ylabel('Frequency [Hz*60]')
        ax.set_yticks([0, 60, 80, 120, 200])
        ax.set_xlabel('Time [sec]')

        peek_ax.plot(f, magnitude[:, frame_to_peek])
        fig.colorbar(mesh, ax=ax)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_stem(f'{output.stem}_stft').with_suffix(".png"))
    fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    main()
