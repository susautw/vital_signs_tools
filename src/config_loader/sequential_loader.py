from typing import Sequence, Union

from fancy import config as cfg


class SequentialLoader(cfg.BaseConfigLoader):
    def __init__(self, sequence: Sequence, setter: Union[cfg.attribute_setters.AttributeSetter, str] = None):
        super().__init__(setter=setter)
        self.sequence = sequence

    def load(self, config: 'cfg.BaseConfig'):
        options = list(config.get_all_options().values())
        if len(self.sequence) > len(options):
            if not isinstance(options[-1].raw_type, list):
                raise ValueError(
                    f"{type(config).__name__}: Sequence length greater"
                    f" than config option and the last option ({options[-1].name}) isn't list "
                )
            sequence = self.sequence[:len(options) - 1] + [self.sequence[len(options) - 1:]]
        elif len(self.sequence) == len(options):
            sequence = self.sequence
        else:
            raise ValueError(
                f"{type(config).__name__}: "
                f"insufficient sequence length to load config (sequence: {len(self.sequence)}, need: {len(options)})"
            )

        for option, val in zip(options, sequence):
            # noinspection PyProtectedMember
            self.get_setter().set(config, option.name, val)

    def get_sub_loader(self, val) -> "cfg.BaseConfigLoader":
        raise RuntimeError("SequentialLoader should not have next layer.")
