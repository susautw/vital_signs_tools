import re
from typing import Union, Iterable, Sequence

from fancy import config as cfg

from config_loader import SequentialLoader


class MMWaveConfigLoader(cfg.BaseConfigLoader):
    pattern = r'[a-z]([A-Z])'

    def __init__(self, text_lines: Iterable[str]):
        super().__init__(setter="ignore")
        self.cfg = {}
        for line in text_lines:
            if line.startswith("%"):
                continue
            tmp = line.strip().split(maxsplit=1)
            if len(tmp) == 0:
                continue
            if len(tmp) == 1:
                tmp = (tmp[0], True)
            title, val = tmp
            title = re.sub(self.pattern, lambda m: (t := m.group().lower())[0] + "_" + t[1], title)
            if title in self.cfg:
                cfg_val = self.cfg[title]
                if isinstance(cfg_val, list):
                    cfg_val.append(val)
                else:
                    self.cfg[title] = [cfg_val, val]
            else:
                self.cfg[title] = val

    def load(self, config: 'cfg.BaseConfig'):
        config_name_option_map = {option.name: option for option in config.get_all_options().values()}
        for title, val in self.cfg.items():
            if title not in config_name_option_map:
                continue
            val_is_sequence = not isinstance(val, str) and isinstance(val, Sequence)
            if isinstance(config_name_option_map[title].raw_type, list):
                if not val_is_sequence:
                    val = [val]
            elif val_is_sequence:
                raise ValueError(f"{title} shouldn't be sequence: {val}")
            self.get_setter().set(config, title, val)

    def get_sub_loader(self, val) -> "cfg.BaseConfigLoader":
        return SequentialLoader(str(val).split())
