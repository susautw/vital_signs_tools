import numpy as np

from occupancy_and_vital_signs_detection.h5_to_image import HeatmapFigureIterator, MapType, \
    HeatmapConfiguratorDecorator, MapSourceType, FigureCollection

TYPE_FIG_SIZE_MAP = {
    MapType.Full: (6, 6),
    MapType.PolarFull: (6, 6)
}

TYPE_FIG_CONTENT_RANGE_MAP = {  # (W_start, W_end, H_start, H_end)
    MapType.Full: np.array([0, 1, 0, 1]),
    MapType.PolarFull: np.array([0, 1, 0.21, 0.79])
}


class HFInitHook:
    fig_sizes: dict[MapType, tuple[int, int]]
    fig_content_range: dict[MapType, np.ndarray]  # [width_start, width_end, height_start, height_end]
    remove_noise: bool

    size: dict[MapType, np.ndarray]  # [width, height]
    content_range_idx: dict[MapType, tuple[slice, ...]]

    def __init__(
            self,
            remove_noise: bool,
            fig_sizes: dict[MapType, tuple[int, int]] = None,
            fig_content_range: dict[MapType, np.ndarray] = None,
    ):
        self.fig_sizes = TYPE_FIG_SIZE_MAP if fig_sizes is None else fig_sizes
        self.fig_content_range = TYPE_FIG_CONTENT_RANGE_MAP if fig_content_range is None else fig_content_range
        self.remove_noise = remove_noise

        self.size = {}
        self.content_range_idx = {}

    def __call__(self, hf_iter: HeatmapFigureIterator):
        for typ, figure_collection in hf_iter.figure_collections.items():
            figure_collection.ax.set_axis_off()
            fig = figure_collection.figure
            fig.set_size_inches(*self.fig_sizes[typ])
            size = np.asarray(fig.canvas.get_width_height())
            w_s, w_e, h_s, h_e = np.int32(self.fig_content_range[typ] * size[[0, 0, 1, 1]])
            self.content_range_idx[typ] = slice(h_s, h_e), slice(w_s, w_e)

            self.size[typ] = np.array([w_e - w_s, h_e - h_s], np.int32)
            fig.tight_layout(pad=0)

        if self.remove_noise:
            prev_configurator = None
            configurator = hf_iter.configurator

            while isinstance(configurator, HeatmapConfiguratorDecorator):
                prev_configurator = configurator
                configurator = configurator.component

            new_configurator = RemoveNoiseDecorator(configurator)
            if prev_configurator is not None:
                prev_configurator.component = new_configurator
            else:
                hf_iter.configurator = new_configurator


class RemoveNoiseDecorator(HeatmapConfiguratorDecorator):

    def configure(
            self,
            sources: dict[MapSourceType, np.ndarray],
            figure_collections: dict[MapType, FigureCollection]
    ) -> None:
        full = sources[MapSourceType.Full]
        leave = full[5:]  # fill valve should not calculate by removing part.
        full[:5] = leave[leave < leave.mean()].mean()
        super().configure(sources, figure_collections)
