from pathlib import Path

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.main import Config
from occupancy_and_vital_signs_detection.plots import HeatmapPlotter
from utility import RollingAverage, convert_artist_to_image

BASE_DIR = Path(__file__).parent

FRAME_DURATION = 0.107
FPS = 1 / FRAME_DURATION


def main():
    source_root = BASE_DIR / "outs"
    paths = sorted(source_root.glob("**/*.h5"))
    fft_out_base = BASE_DIR / "fft_figs"
    heatmap_fig_out_base = BASE_DIR / "heatmap_figs"

    start = 50
    size = 128

    configs_paths = {
        "30cm": BASE_DIR / "vod_vs_68xx_10fps_center_30.cfg",
        "60cm": BASE_DIR / "vod_vs_68xx_10fps_center_60.cfg",
        "90cm": BASE_DIR / "vod_vs_68xx_10fps_center_90.cfg",
    }

    configs = {
        name: Config(MMWaveConfigLoader(path.read_text().split("\n")))
        for name, path in configs_paths.items()
    }

    for path in paths:
        print(path)
        ds = h5py.File(path)

        full_heatmaps: np.ndarray = np.asarray(ds["heatmap/full"])
        heatmaps: np.ndarray = np.asarray(ds["heatmap/zone0"])

        heatmap_fig_out = (heatmap_fig_out_base / path.relative_to(source_root)).with_suffix(".png")
        fft_fig_out = (fft_out_base / path.relative_to(source_root)).with_suffix(".png")

        heatmap_fig_out.parent.mkdir(parents=True, exist_ok=True)
        fft_fig_out.parent.mkdir(parents=True, exist_ok=True)

        plot_heatmap(heatmaps, full_heatmaps, heatmap_fig_out, configs[path.stem[:4]])
        # find_frequency_via_fft(heatmaps, start, size, fft_fig_out)


def plot_heatmap(heatmaps: np.ndarray, full_heatmaps: np.ndarray, output: Path, config) -> None:
    rolling_average = RollingAverage(4, low=1000, init=0)
    fig: plt.Figure = plt.figure()
    zone_ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(0, 0, 1, 1)
    plotter = HeatmapPlotter(config, zones={0: zone_ax})
    plotter.initialize()
    zone_mesh = plotter.zone_plots[0].mesh
    max_idx = np.argmax(heatmaps.sum(axis=(1, 2))[50: 50 + 20]) + 50
    w, h = fig.canvas.get_width_height()
    size = np.array([h, w])
    combined = np.zeros((h * 7, w * 7, 3), dtype=np.uint8)

    for i, (heatmap, full_heatmap) in enumerate(zip(heatmaps, full_heatmaps)):
        zone_mesh.set_array(heatmap)
        zone_mesh.set_clim(0, rolling_average.next(np.max(full_heatmap)))
        fig.savefig(output.with_stem(f'{output.stem}_{i:04d}'))
        if max_idx <= i < max_idx + 49:
            start = np.array(np.unravel_index(i - max_idx, shape=(7, 7))) * size
            combined[start[0]: start[0] + h, start[1]: start[1] + w] = convert_artist_to_image(fig)

    cv2.imwrite(str(output.with_stem(f'{output.stem}_combined')), combined)
    fig.clf()
    plt.close(fig)


def find_frequency_via_fft(heatmaps: np.ndarray, start: int, size: int, output: Path) -> None:
    sum_ = heatmaps.sum(axis=(1, 2))[start:start + size]

    fig: plt.Figure = plt.figure()
    gs = fig.add_gridspec(3, 1, hspace=0.7)

    sum_ax = fig.add_subplot(gs[0])
    sum_ax.plot(np.arange(size) * FRAME_DURATION, sum_)
    sum_ax.set_title("Sum of heatmap")
    f = np.fft.fft(sum_)

    shifted = np.fft.fftshift(f)
    magnitude = np.sqrt(np.real(shifted) ** 2 + np.imag(shifted) ** 2)

    peak_idx = np.argmax(magnitude[size // 2 + 1:]) + 1
    peak = peak_idx * (10 / size) * 60
    print(peak_idx, peak)

    magnitude[np.argmax(magnitude)] = 0
    angle = np.angle(shifted)

    half_size = size // 2
    x = np.arange(-half_size, -half_size + size) * (FPS / size) * 60

    mag_ax: plt.Axes = fig.add_subplot(gs[1])
    mag_ax.plot(x, magnitude)
    shifted_peak_idx = peak_idx + half_size
    mag_ax.scatter(x[shifted_peak_idx], magnitude[shifted_peak_idx], c="r", label=f"peak({peak})")
    mag_ax.legend()
    mag_ax.set_title("Magnitude")
    angle_ax = fig.add_subplot(gs[2])
    angle_ax.plot(x, angle)
    angle_ax.set_title("Angle")

    fig.savefig(output)
    fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    main()
