from typing import Any, Union

from fancy import config as cfg
from fancy.config.process import auto_process_typ


def union(*types):
    types = [auto_process_typ(t) for t in types]

    def _inner(val: Any) -> Any:
        for t in types:
            try:
                return t(val)
            except ValueError:
                pass
        return types[0](val)

    return _inner


class ChannelConfig(cfg.BaseConfig):
    rx_channel_en: int = cfg.Option(type=int)
    tx_channel_en: int = cfg.Option(type=int)
    cascading: int = cfg.Option(type=int)

    num_rx_ant: int = cfg.PlaceHolder()
    num_tx_ant: int = cfg.PlaceHolder()

    def post_load(self):
        self.num_rx_ant = self.decode_mask(self.rx_channel_en)
        self.num_tx_ant = self.decode_mask(self.tx_channel_en)

    def decode_mask(self, mask: int):
        sum_ = 0
        while mask > 0:
            sum_ += mask & 1
            mask >>= 1
        return sum_


class ProfileConfig(cfg.BaseConfig):
    profile_id: int = cfg.Option(type=int)
    start_freq: Union[int, float] = cfg.Option(type=union(int, float))
    idle_time: Union[int, float] = cfg.Option(type=union(int, float))
    adc_start_time: Union[int, float] = cfg.Option(type=union(int, float))
    ramp_end_time: Union[int, float] = cfg.Option(type=union(int, float))
    tx_out_power: int = cfg.Option(type=int)
    tx_phase_shifter: int = cfg.Option(type=int)
    freq_slope_const: Union[int, float] = cfg.Option(type=union(int, float))
    tx_start_time: Union[int, float] = cfg.Option(type=union(int, float))
    num_adc_samples: int = cfg.Option(type=int)
    dig_out_sample_rate: int = cfg.Option(type=int)
    hpf_corner_freq1: int = cfg.Option(type=int)
    hpf_corner_freq2: int = cfg.Option(type=int)
    rx_gain: int = cfg.Option(type=int)


class ChirpConfig(cfg.BaseConfig):
    chirp_start_index: int = cfg.Option(type=int)
    chirp_end_index: int = cfg.Option(type=int)
    profile_identifier: int = cfg.Option(type=int)
    start_frequency_variation: Union[int, float] = cfg.Option(type=union(int, float), description="in Hz")
    frequency_slope_variation: Union[int, float] = cfg.Option(type=union(int, float), description="in kHz/us")
    idle_time_variation: Union[int, float] = cfg.Option(type=union(int, float), description="in u-sec")
    adc_start_time_variation: Union[int, float] = cfg.Option(type=union(int, float), description="in u-sec")
    tx_antenna_enable_mask: int = cfg.Option(type=int)


class FrameConfig(cfg.BaseConfig):
    chirp_start_index: int = cfg.Option(type=int)
    chirp_end_index: int = cfg.Option(type=int)
    number_of_loops: int = cfg.Option(type=int)
    number_of_frames: int = cfg.Option(type=int)
    frame_periodicity_in_ms: Union[int, float] = cfg.Option(type=union(int, float))
    trigger_select: int = cfg.Option(type=int)
    frame_trigger_delay_in_ms: Union[int, float] = cfg.Option(type=union(int, float))




